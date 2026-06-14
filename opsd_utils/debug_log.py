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
_PROBE_ON_GENERATE = False
_PROBE_FIRST_TOKEN_LOGITS = True
_PROBE_PROMPT_TAIL_TOKENS = 16
_PROBE_LOG_MODEL_CONTEXT = True
_HEALTH_MONITOR_ENABLED = True
_HEALTH_LOG_ON_GENERATE = True
_HEALTH_LOG_EVERY_STEP = True
_HEALTH_LOG_DETAIL_BUNDLE = True
_HEALTH_LOG_ALERTS_IMMEDIATELY = True
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


def _env_probe_on_generate() -> Optional[bool]:
    raw = os.environ.get("DYME_OPSD_PROBE_ON_GENERATE", "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def _env_probe_first_token_logits() -> Optional[bool]:
    raw = os.environ.get("DYME_OPSD_PROBE_FIRST_TOKEN_LOGITS", "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def _env_probe_prompt_tail_tokens() -> Optional[int]:
    raw = os.environ.get("DYME_OPSD_PROBE_PROMPT_TAIL_TOKENS", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _env_probe_log_model_context() -> Optional[bool]:
    raw = os.environ.get("DYME_OPSD_PROBE_LOG_MODEL_CONTEXT", "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def _env_health_monitor_enabled() -> Optional[bool]:
    raw = os.environ.get("DYME_OPSD_HEALTH_MONITOR", "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def configure(
    *,
    enabled: Optional[bool] = None,
    detail_every: Optional[int] = None,
    probe_on_generate: Optional[bool] = None,
    probe_first_token_logits: Optional[bool] = None,
    probe_prompt_tail_tokens: Optional[int] = None,
    probe_log_model_context: Optional[bool] = None,
    health_monitor_enabled: Optional[bool] = None,
    health_log_on_generate: Optional[bool] = None,
    health_log_every_step: Optional[bool] = None,
    health_log_detail_bundle: Optional[bool] = None,
    health_log_alerts_immediately: Optional[bool] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> bool:
    """Configure global OPSD debug logging. Returns whether debug is enabled."""
    global _DEBUG_ENABLED, _DETAIL_EVERY, _PROBE_ON_GENERATE
    global _PROBE_FIRST_TOKEN_LOGITS, _PROBE_PROMPT_TAIL_TOKENS, _PROBE_LOG_MODEL_CONTEXT
    global _HEALTH_MONITOR_ENABLED, _HEALTH_LOG_ON_GENERATE, _HEALTH_LOG_EVERY_STEP
    global _HEALTH_LOG_DETAIL_BUNDLE, _HEALTH_LOG_ALERTS_IMMEDIATELY
    global _RANK, _WORLD_SIZE
    if enabled is None:
        enabled = _env_debug_enabled()
    _DEBUG_ENABLED = bool(enabled)
    if detail_every is not None:
        _DETAIL_EVERY = max(0, int(detail_every))
    elif _env_detail_every() != 10 or os.environ.get("DYME_OPSD_DETAIL_EVERY"):
        _DETAIL_EVERY = _env_detail_every()
    env_probe = _env_probe_on_generate()
    if probe_on_generate is not None:
        _PROBE_ON_GENERATE = bool(probe_on_generate)
    elif env_probe is not None:
        _PROBE_ON_GENERATE = env_probe
    env_first_logits = _env_probe_first_token_logits()
    if probe_first_token_logits is not None:
        _PROBE_FIRST_TOKEN_LOGITS = bool(probe_first_token_logits)
    elif env_first_logits is not None:
        _PROBE_FIRST_TOKEN_LOGITS = env_first_logits
    env_tail = _env_probe_prompt_tail_tokens()
    if probe_prompt_tail_tokens is not None:
        _PROBE_PROMPT_TAIL_TOKENS = max(1, int(probe_prompt_tail_tokens))
    elif env_tail is not None:
        _PROBE_PROMPT_TAIL_TOKENS = env_tail
    env_model_ctx = _env_probe_log_model_context()
    if probe_log_model_context is not None:
        _PROBE_LOG_MODEL_CONTEXT = bool(probe_log_model_context)
    elif env_model_ctx is not None:
        _PROBE_LOG_MODEL_CONTEXT = env_model_ctx
    env_health = _env_health_monitor_enabled()
    if health_monitor_enabled is not None:
        _HEALTH_MONITOR_ENABLED = bool(health_monitor_enabled)
    elif env_health is not None:
        _HEALTH_MONITOR_ENABLED = env_health
    if health_log_on_generate is not None:
        _HEALTH_LOG_ON_GENERATE = bool(health_log_on_generate)
    if health_log_every_step is not None:
        _HEALTH_LOG_EVERY_STEP = bool(health_log_every_step)
    if health_log_detail_bundle is not None:
        _HEALTH_LOG_DETAIL_BUNDLE = bool(health_log_detail_bundle)
    if health_log_alerts_immediately is not None:
        _HEALTH_LOG_ALERTS_IMMEDIATELY = bool(health_log_alerts_immediately)
    if rank is not None:
        _RANK = rank
    if world_size is not None:
        _WORLD_SIZE = world_size
    return _DEBUG_ENABLED


def detail_every() -> int:
    return _DETAIL_EVERY


def probe_on_generate() -> bool:
    return _PROBE_ON_GENERATE


def probe_first_token_logits() -> bool:
    return _PROBE_FIRST_TOKEN_LOGITS


def probe_prompt_tail_tokens() -> int:
    return _PROBE_PROMPT_TAIL_TOKENS


def probe_log_model_context() -> bool:
    return _PROBE_LOG_MODEL_CONTEXT


def should_log_probe() -> bool:
    """True when lightweight per-generate probe should run (rank 0 only)."""
    return _PROBE_ON_GENERATE and _RANK == 0


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


def get_detail_step() -> Optional[int]:
    return _DETAIL_STEP


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


def _probe_prefix(section: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    step = _DETAIL_STEP if _DETAIL_STEP is not None else "?"
    return (
        f"[OPSD-PROBE][{ts}][rank={_RANK}/{_WORLD_SIZE}]"
        f"[global_step={step}][{_STEP_LABEL}][{section}]"
    )


def log_probe(section: str, msg: str, **fields: Any) -> None:
    """Lightweight per-generate diagnostic (rank 0). Independent of OPSD-DEBUG verbosity."""
    if not should_log_probe():
        return
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v, max_len=1200)}" for k, v in fields.items())
    print(f"{_probe_prefix(section)} {msg}{extra}", flush=True)


def _gendbg_prefix(section: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    step = _DETAIL_STEP if _DETAIL_STEP is not None else "?"
    return (
        f"[OPSD-GENDBG][{ts}][rank={_RANK}/{_WORLD_SIZE}]"
        f"[global_step={step}][{_STEP_LABEL}][{section}]"
    )


def should_log_gendbg() -> bool:
    """True when deep generate diagnostics should run (rank 0 only)."""
    return _PROBE_ON_GENERATE and _RANK == 0


def health_monitor_enabled() -> bool:
    return _HEALTH_MONITOR_ENABLED and _RANK == 0


def should_log_health_on_generate() -> bool:
    return health_monitor_enabled() and _HEALTH_LOG_ON_GENERATE and should_log_probe()


def should_log_health_every_step() -> bool:
    return health_monitor_enabled() and _HEALTH_LOG_EVERY_STEP


def should_log_health_detail_bundle() -> bool:
    return health_monitor_enabled() and _HEALTH_LOG_DETAIL_BUNDLE


def should_log_health_alerts_immediately() -> bool:
    return health_monitor_enabled() and _HEALTH_LOG_ALERTS_IMMEDIATELY


def _health_prefix(section: str, global_step: Optional[int] = None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    step = global_step if global_step is not None else (_DETAIL_STEP if _DETAIL_STEP is not None else "?")
    return f"[OPSD-HEALTH][{ts}][rank={_RANK}/{_WORLD_SIZE}][global_step={step}][{section}]"


def log_health(section: str, msg: str, global_step: Optional[int] = None, **fields: Any) -> None:
    """L1/L2/L4 health lines (rank 0)."""
    if not health_monitor_enabled():
        return
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v, max_len=1200)}" for k, v in fields.items())
    print(f"{_health_prefix(section, global_step)} {msg}{extra}", flush=True)


def log_health_detail_banner(global_step: int, title: str) -> None:
    if not should_log_health_detail_bundle() or not should_log_detail(global_step):
        return
    bar = "=" * 20
    print(f"{_detail_prefix(global_step, 'health')} {bar} {title} {bar}", flush=True)


def log_health_detail(section: str, msg: str, global_step: int, **fields: Any) -> None:
    """L3 periodic health bundle (rank 0, same cadence as OPSD-DETAIL)."""
    if not should_log_health_detail_bundle() or not should_log_detail(global_step):
        return
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v, max_len=1200)}" for k, v in fields.items())
    print(f"{_detail_prefix(global_step, section)} {msg}{extra}", flush=True)


def log_gendbg(section: str, msg: str, **fields: Any) -> None:
    """Deep per-generate diagnostic (rank 0). Uses [OPSD-GENDBG] prefix."""
    if not should_log_gendbg():
        return
    extra = ""
    if fields:
        extra = " | " + " | ".join(f"{k}={_fmt(v, max_len=1200)}" for k, v in fields.items())
    print(f"{_gendbg_prefix(section)} {msg}{extra}", flush=True)


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
