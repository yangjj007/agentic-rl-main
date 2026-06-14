"""RLSD warmup gates for OPSD degenerate skip, denser online SFT, and embedded SFT cold start."""
from __future__ import annotations

from typing import Any, Mapping, Optional


def current_global_step(trainer: Any) -> int:
    return int(getattr(getattr(trainer, "state", None), "global_step", getattr(trainer, "_step", 0)) or 0)


def sft_cold_start_steps(opsd_config: Mapping[str, Any], max_steps: Optional[int]) -> int:
    """Steps at start of training devoted to embedded offline-style SFT (no generate / no OPSD)."""
    gate = opsd_config.get("gate", {})
    steps_env = gate.get("sft_cold_start_steps")
    if steps_env is not None:
        return max(0, int(steps_env))
    frac = float(gate.get("sft_cold_start_frac", 0.0) or 0.0)
    if frac <= 0.0 or max_steps is None or max_steps <= 0:
        return 0
    return max(1, int(max_steps * frac))


def in_sft_cold_start(
    opsd_config: Mapping[str, Any],
    global_step: int,
    max_steps: Optional[int],
) -> bool:
    cold_steps = sft_cold_start_steps(opsd_config, max_steps)
    return cold_steps > 0 and global_step < cold_steps


def resolve_skip_degenerate_opsd(
    opsd_config: Mapping[str, Any],
    global_step: int,
    max_steps: Optional[int] = None,
) -> bool:
    gate = opsd_config.get("gate", {})
    if not gate.get("skip_degenerate_for_opsd", False):
        return False
    cold_end = sft_cold_start_steps(opsd_config, max_steps)
    warmup = int(gate.get("degen_skip_warmup_steps", 200))
    # Do not skip degenerate OPSD during embedded SFT cold start or its degen warmup window.
    threshold = cold_end + warmup if cold_end > 0 else warmup
    return global_step >= threshold


def sft_slots_for_step(
    opsd_config: Mapping[str, Any],
    global_step: int,
    max_steps: Optional[int] = None,
) -> int:
    if in_sft_cold_start(opsd_config, global_step, max_steps):
        return 0
    gate = opsd_config.get("gate", {})
    warmup_steps = int(gate.get("sft_warmup_steps", 200))
    cold_end = sft_cold_start_steps(opsd_config, max_steps)
    effective_warmup_end = cold_end + warmup_steps if cold_end > 0 else warmup_steps
    if global_step < effective_warmup_end:
        return max(1, int(gate.get("sft_warmup_slots_per_group", 2)))
    return 1
