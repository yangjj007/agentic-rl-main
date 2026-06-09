"""Verbose debug logging for the OPSD / TriMode training pipeline."""
from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

_DEBUG_ENABLED = False
_DETAIL_EVERY = 10
_RANK = 0
_WORLD_SIZE = 1
_STEP_LABEL = "init"
_DETAIL_STEP: Optional[int] = None
_CALL_COUNTER = 0

MODE_NAMES = {0: "GRPO", 1: "OPSD", 2: "SFT"}


def _env_debug_enabled() -> bool:
    return os.environ.get("DYME_OPSD_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _env_detail_every() -> int:
    raw = os.environ.get("DYME_OPSD_DETAIL_EVERY", "").strip()
    if not raw:
        return 10
    try:
        return max(0, int(raw))
    except ValueError:
        return 10


def configure(
    *,
    enabled: Optional[bool] = None,
    detail_every: Optional[int] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> bool:
    """Configure global OPSD debug logging. Returns whether debug is enabled."""
    global _DEBUG_ENABLED, _DETAIL_EVERY, _RANK, _WORLD_SIZE
    if enabled is None:
        enabled = _env_debug_enabled()
    _DEBUG_ENABLED = bool(enabled)
    if detail_every is not None:
        _DETAIL_EVERY = max(0, int(detail_every))
    elif _env_detail_every() != 10 or os.environ.get("DYME_OPSD_DETAIL_EVERY"):
        _DETAIL_EVERY = _env_detail_every()
    if rank is not None:
        _RANK = rank
    if world_size is not None:
        _WORLD_SIZE = world_size
    return _DEBUG_ENABLED


def detail_every() -> int:
    return _DETAIL_EVERY


def should_log_detail(global_step: Optional[int]) -> bool:
    """True when a full diagnostic bundle should be emitted (rank 0 only)."""
    if _DETAIL_EVERY <= 0 or _RANK != 0:
        return False
    if global_step is None:
        return False
    return int(global_step) % _DETAIL_EVERY == 0


def is_enabled() -> bool:
    return _DEBUG_ENABLED


def set_step_label(label: str) -> None:
    global _STEP_LABEL
    _STEP_LABEL = label


def set_detail_step(global_step: Optional[int]) -> None:
    global _DETAIL_STEP
    _DETAIL_STEP = global_step


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


def _detail_prefix(global_step: int, section: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[OPSD-DETAIL][{ts}][rank={_RANK}/{_WORLD_SIZE}]"
        f"[step={global_step}][every={_DETAIL_EVERY}][{section}]"
    )


def log_detail_banner(global_step: int, title: str) -> None:
    if not should_log_detail(global_step):
        return
    bar = "=" * 20
    print(f"{_detail_prefix(global_step, 'BANNER')} {bar} {title} {bar}", flush=True)


def log_detail(section: str, msg: str, global_step: Optional[int] = None, **fields: Any) -> None:
    """Full-detail diagnostic line (periodic, rank 0). Independent of verbose OPSD-DEBUG."""
    step = global_step if global_step is not None else _DETAIL_STEP
    if step is None or isinstance(step, str):
        return
    if not should_log_detail(step):
        return
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v, max_len=800)}" for k, v in fields.items())
    print(f"{_detail_prefix(step, section)} {msg}{extra}", flush=True)


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
