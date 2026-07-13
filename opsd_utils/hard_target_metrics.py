from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def no_full_hint_hard_sft_enabled(gate: Mapping[str, Any]) -> bool:
    return bool(
        gate.get("disable_online_sft_slots", False)
        and not gate.get("online_sft_on_all_wrong", True)
        and str(gate.get("teacher_probe_failure_route", "sft")).lower()
        == "mixed_grpo_all_wrong_skip"
    )


@dataclass
class OnlineSftSourceCounts:
    slot: int = 0
    route: int = 0
    forced: int = 0

    def record(self, source: str) -> None:
        if source not in {"slot", "route", "forced"}:
            raise ValueError(f"unknown online SFT source: {source}")
        setattr(self, source, int(getattr(self, source)) + 1)

    def as_rates(self, *, total_completions: int) -> dict[str, float]:
        denominator = max(int(total_completions), 1)
        slot_rate = self.slot / denominator
        route_rate = self.route / denominator
        forced_rate = self.forced / denominator
        aggregate = slot_rate + route_rate + forced_rate
        return {
            "routing/legacy_online_sft_slot_rate": slot_rate,
            "routing/legacy_online_sft_route_rate": route_rate,
            "routing/legacy_online_sft_forced_rate": forced_rate,
            "routing/legacy_online_sft_rate": aggregate,
            "routing/full_hint_hard_target_rate": aggregate,
        }
