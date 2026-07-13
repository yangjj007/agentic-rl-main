from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.chart_cot_quality_gate import (
    ChartCoTQualityGateConfig,
    append_quality_sample_records,
    aggregate_chart_cot_verifications,
    filter_quality_eligible_indices,
)
from reward_utils.chart_cot_verifier import verify_chart_cot_trajectory


TABLE = "Year | Value\n2019 | 70\n2020 | 77"


def _verification(quality: str):
    responses = {
        "Q3": (
            "Goal: Find max.\nObservation: 2019: 70 and 2020: 77.\n"
            "Reasoning: The maximum is 77.\nConclusion: The maximum is 77.\nAnswer: 77"
        ),
        "Q2": (
            "Goal: Find max.\nObservation: Inspect chart.\nReasoning: Use evidence.\n"
            "Conclusion: This is the result.\nAnswer: 77"
        ),
        "Q1": "Reasoning: The maximum is 77.\nAnswer: 77",
        "Q0": (
            "Goal: Find max.\nObservation: 2019: 71.\nReasoning: The maximum is 77.\n"
            "Conclusion: The maximum is 77.\nAnswer: 77"
        ),
    }
    return verify_chart_cot_trajectory(responses[quality], TABLE, answer_correct=True)


def test_off_and_diagnostic_preserve_candidates_but_gate_keeps_q3_only() -> None:
    verifications = {0: _verification("Q3"), 1: _verification("Q2"), 2: _verification("Q0")}

    off = filter_quality_eligible_indices(
        {0, 1, 2}, verifications, ChartCoTQualityGateConfig(enabled=False, mode="off")
    )
    diagnostic = filter_quality_eligible_indices(
        {0, 1, 2}, verifications, ChartCoTQualityGateConfig(enabled=True, mode="diagnostic")
    )
    gate = filter_quality_eligible_indices(
        {0, 1, 2}, verifications, ChartCoTQualityGateConfig(enabled=True, mode="gate")
    )

    assert off.eligible_indices == {0, 1, 2}
    assert diagnostic.eligible_indices == {0, 1, 2}
    assert diagnostic.rejected_indices == set()
    assert gate.eligible_indices == {0}
    assert gate.rejected_indices == {1, 2}


def test_gate_aggregation_is_zero_safe_and_reports_quality_rates() -> None:
    empty = aggregate_chart_cot_verifications({})
    metrics = aggregate_chart_cot_verifications(
        {0: _verification("Q3"), 1: _verification("Q2"), 2: _verification("Q1"), 3: _verification("Q0")}
    )

    assert empty["candidate_count"] == 0
    assert empty["q3_rate"] == 0.0
    assert metrics["candidate_count"] == 4
    assert metrics["q3_rate"] == 0.25
    assert metrics["q2_rate"] == 0.25
    assert metrics["q1_rate"] == 0.25
    assert metrics["q0_rate"] == 0.25
    assert metrics["conclusion_answer_consistent_rate"] == 0.5


def test_quality_sample_records_append_to_rank_jsonl(tmp_path) -> None:
    path = append_quality_sample_records(
        output_dir=str(tmp_path),
        rank=2,
        global_step=9,
        records=({"global_idx": 1, "quality": "Q0"},),
    )

    assert path == str(tmp_path / "chart_cot_quality" / "rank2.jsonl")
    row = json.loads((tmp_path / "chart_cot_quality" / "rank2.jsonl").read_text(encoding="utf-8"))
    assert row == {"global_idx": 1, "global_step": 9, "quality": "Q0"}
