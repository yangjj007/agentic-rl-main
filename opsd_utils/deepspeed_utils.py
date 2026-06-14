"""Helpers for DeepSpeed + Accelerate launch detection."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_accelerate_config_path(config_name: Optional[str] = None) -> Optional[Path]:
    candidates: list[str] = []
    if config_name:
        candidates.append(str(config_name).strip())
    for env_key in ("ACCELERATE_CONFIG", "ACCELERATE_CONFIG_FILE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            candidates.append(val)
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_file():
            path = _project_root() / raw
        if path.is_file():
            return path
    return None


def uses_deepspeed_json_file(config_name: Optional[str] = None) -> bool:
    """True when Accelerate loads DeepSpeed settings from an external JSON file."""
    path = resolve_accelerate_config_path(config_name)
    if path is None:
        return False
    return "deepspeed_config_file" in path.read_text(encoding="utf-8")


def _yaml_get_str(path: Path, key: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            return m.group(1).strip().strip("'\"")
    return None


def is_deepspeed_accelerate_config(config_name: Optional[str] = None) -> bool:
    path = resolve_accelerate_config_path(config_name)
    if path is None:
        return False
    dist = (_yaml_get_str(path, "distributed_type") or "").upper()
    return dist == "DEEPSPEED"


def deepspeed_zero_stage(config_name: Optional[str] = None) -> Optional[int]:
    path = resolve_accelerate_config_path(config_name)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"zero_stage\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"deepspeed_config_file\s*:\s*(\S+)", text, re.IGNORECASE)
    if not m:
        return None
    json_path = Path(m.group(1).strip().strip("'\""))
    if not json_path.is_file():
        json_path = _project_root() / json_path
    if not json_path.is_file():
        return None
    ds_json = json.loads(json_path.read_text(encoding="utf-8"))
    stage = (ds_json.get("zero_optimization") or {}).get("stage")
    return int(stage) if stage is not None else None


def should_colocate_teacher_with_student(device_map: Optional[str] = None) -> bool:
    """True when frozen teacher should sit on the same GPU as the trainable student."""
    raw = (device_map or os.environ.get("DYME_TEACHER_DEVICE_MAP", "")).strip().lower()
    if raw in ("same", "colocate", "local"):
        return True
    if os.environ.get("DYME_DEEPSPEED_COLOCATE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if is_deepspeed_accelerate_config() and raw in ("", "auto"):
        return True
    return False


def gradient_checkpointing_enable_kwargs(config_name: Optional[str] = None) -> Optional[dict]:
    """
    Kwargs for ``model.gradient_checkpointing_enable``.

    DeepSpeed ZeRO-1/2 + reentrant checkpointing runs backward twice per segment and
    hits: "parameter ... has already been reduced".
    """
    if not is_deepspeed_accelerate_config(config_name):
        return None
    override = os.environ.get("DYME_GRADIENT_CHECKPOINTING_USE_REENTRANT", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return {"use_reentrant": True}
    if override in ("0", "false", "no", "off"):
        return {"use_reentrant": False}
    return {"use_reentrant": False}


def deepspeed_requires_single_student_forward(config_name: Optional[str] = None) -> bool:
    """
    DeepSpeed ZeRO-1/2 cannot reduce gradients when the student runs multiple
    forwards in one backward (GRPO micro-chunks + OPSD loop).
    """
    stage = deepspeed_zero_stage(config_name)
    return stage is not None and stage <= 2


def should_disable_gradient_checkpointing(config_name: Optional[str] = None) -> bool:
    """Gradient checkpointing also triggers double reduction under ZeRO-1/2."""
    return deepspeed_requires_single_student_forward(config_name)


def student_forward_chunk_size(
    batch_size: int,
    has_vision: bool,
    config_name: Optional[str] = None,
) -> int:
    """
    Micro-batch size for student forwards in ``_get_per_token_logps``.

    Under ZeRO-1/2 we must use one forward per backward (full local batch by default).
    Override with ``DYME_STUDENT_FORWARD_CHUNK`` only if you accept ZeRO-3+ or OOM risk.
    """
    if not has_vision:
        return batch_size
    if not deepspeed_requires_single_student_forward(config_name):
        return 1
    override = os.environ.get("DYME_STUDENT_FORWARD_CHUNK", "").strip()
    if override.isdigit():
        return max(1, min(batch_size, int(override)))
    return batch_size
