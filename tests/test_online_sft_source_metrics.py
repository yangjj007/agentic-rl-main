from __future__ import annotations

import pytest

from opsd_utils.hard_target_metrics import OnlineSftSourceCounts


def test_online_sft_source_counts_report_source_specific_rates() -> None:
    counts = OnlineSftSourceCounts()
    counts.record("slot")
    counts.record("route")
    counts.record("forced")

    assert counts.as_rates(total_completions=8) == {
        "routing/legacy_online_sft_slot_rate": 0.125,
        "routing/legacy_online_sft_route_rate": 0.125,
        "routing/legacy_online_sft_forced_rate": 0.125,
        "routing/legacy_online_sft_rate": 0.375,
        "routing/full_hint_hard_target_rate": 0.375,
    }


def test_online_sft_source_counts_are_exact_zero_without_replacements() -> None:
    counts = OnlineSftSourceCounts()
    assert all(value == 0.0 for value in counts.as_rates(total_completions=32).values())


def test_online_sft_source_counts_reject_unknown_source() -> None:
    counts = OnlineSftSourceCounts()
    with pytest.raises(ValueError, match="online SFT source"):
        counts.record("teacher_repair")
