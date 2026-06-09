"""Verbose debug logging for the OPSD / TriMode training pipeline."""
from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

_DEBUG_ENABLED = False
_RANK = 0
_WORLD_SIZE = 1
_STEP_LABEL = "init"
_CALL_COUNTER = 0

MODE_NAMES = {0: "GRPO", 1: "OPSD", 2: "SFT"}


def _env_debug_enabled() -> bool:
    return os.environ.get("DYME_OPSD_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def configure(
    *,
    enabled: Optional[bool] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> bool:
    """Configure global OPSD debug logging. Returns whether debug is enabled."""
    global _DEBUG_ENABLED, _RANK, _WORLD_SIZE
    if enabled is None:
        enabled = _env_debug_enabled()
    _DEBUG_ENABLED = bool(enabled)
    if rank is not None:
        _RANK = rank
    if world_size is not None:
        _WORLD_SIZE = world_size
    return _DEBUG_ENABLED


def is_enabled() -> bool:
    return _DEBUG_ENABLED


def set_step_label(label: str) -> None:
    global _STEP_LABEL
    _STEP_LABEL = label


def _next_call_id(stage: str) -> str:
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    return f"{_CALL_COUNTER}:{stage}"


def _fmt(value: Any, max_len: int = 240) -> str:
    if value is None:
        return "None"
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device}, numel={value.numel()})"
            )
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            head = ", ".join(_fmt(v, max_len=40) for v in value[:6])
            return f"[{head}, ... +{len(value) - 6} more, total={len(value)}]"
        return "[" + ", ".join(_fmt(v, max_len=40) for v in value) + "]"
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _prefix(stage: str, call_id: Optional[str] = None) -> str:
    cid = call_id or _next_call_id(stage)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"[OPSD-DEBUG][{ts}][rank={_RANK}/{_WORLD_SIZE}][step={_STEP_LABEL}][{cid}]"


def log(stage: str, msg: str, **fields: Any) -> None:
    if not _DEBUG_ENABLED:
        return
    call_id = _next_call_id(stage)
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    print(f"{_prefix(stage, call_id)} {msg}{extra}", flush=True)


def log_config(stage: str, title: str, config: dict[str, Any]) -> None:
    if not _DEBUG_ENABLED:
        return
    log(stage, title, config=_fmt(config, max_len=2000))


def log_sync_point(stage: str, msg: str, **fields: Any) -> None:
    """Mark a point where all ranks must reach before a collective call."""
    log(stage, f"[SYNC_POINT] {msg}", **fields)


def log_mode_summary(stage: str, prompt_modes: list[int], completion_modes: Optional[list[int]] = None) -> None:
    if not _DEBUG_ENABLED:
        return
    prompt_names = [MODE_NAMES.get(m, str(m)) for m in prompt_modes]
    fields: dict[str, Any] = {"prompt_modes": prompt_names}
    if completion_modes is not None:
        counts = {name: completion_modes.count(code) for code, name in MODE_NAMES.items()}
        fields["completion_mode_counts"] = counts
    log(stage, "mode routing summary", **fields)


def log_tensor(stage: str, name: str, tensor: Any) -> None:
    if not _DEBUG_ENABLED:
        return
    log(stage, f"tensor `{name}`", value=_fmt(tensor))


def log_exception(stage: str, msg: str, exc: BaseException) -> None:
    if not _DEBUG_ENABLED:
        return
    tb = traceback.format_exc()
    log(stage, msg, error=repr(exc), traceback=_fmt(tb, max_len=4000))


@contextmanager
def timed(stage: str, msg: str = "", **fields: Any):
    if not _DEBUG_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    log(stage, f"START {msg}", **fields)
    try:
        yield
    except Exception as exc:
        log_exception(stage, f"FAILED {msg}", exc)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        log(stage, f"END {msg}", elapsed_s=f"{elapsed:.4f}")
