"""RLSD warmup gates for OPSD degenerate skip and denser online SFT."""
from __future__ import annotations

from typing import Any, Mapping


def current_global_step(trainer: Any) -> int:
    return int(getattr(getattr(trainer, "state", None), "global_step", getattr(trainer, "_step", 0)) or 0)


def resolve_skip_degenerate_opsd(opsd_config: Mapping[str, Any], global_step: int) -> bool:
    gate = opsd_config.get("gate", {})
    if not gate.get("skip_degenerate_for_opsd", False):
        return False
    warmup = int(gate.get("degen_skip_warmup_steps", 200))
    return global_step >= warmup


def sft_slots_for_step(opsd_config: Mapping[str, Any], global_step: int) -> int:
    gate = opsd_config.get("gate", {})
    warmup_steps = int(gate.get("sft_warmup_steps", 200))
    if global_step < warmup_steps:
        return max(1, int(gate.get("sft_warmup_slots_per_group", 2)))
    return 1
