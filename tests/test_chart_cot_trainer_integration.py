from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.chart_cot_quality_gate import (
    ChartCoTQualityGateConfig,
    evaluate_teacher_trajectory_quality,
)
import opsd_utils.chart_cot_quality_gate as gate_module


def _response(observation: str, conclusion: str = "The maximum is 77.") -> str:
    return (
        "Goal: Find the maximum.\n"
        f"Observation: {observation}\n"
        "Reasoning: The maximum is 77.\n"
        f"Conclusion: {conclusion}\n"
        "Answer: 77"
    )


def test_quality_evaluation_maps_completion_indices_to_prompt_samples() -> None:
    samples = [
        {"question": "first", "visual_fact_deplot": "Year | Value\n2019 | 70\n2020 | 77"},
        {"question": "second", "visual_fact_deplot": "Year | Value\n2019 | 55\n2020 | 77"},
    ]
    texts = {0: _response("2019: 70 and 2020: 77."), 3: _response("2019: 70 and 2020: 77.")}

    result = evaluate_teacher_trajectory_quality(
        teacher_traj_texts=texts,
        samples=samples,
        num_generations=2,
        config=ChartCoTQualityGateConfig(enabled=True, mode="gate", max_log_samples=1),
    )

    assert result.verifications[0].quality == "Q3"
    assert result.verifications[3].quality == "Q0"
    assert result.gate_result.eligible_indices == {0}
    assert result.gate_result.rejected_indices == {3}
    assert len(result.sample_records) == 1
    assert result.sample_records[0]["global_idx"] == 0


def test_diagnostic_mode_preserves_missing_deplot_candidate() -> None:
    result = evaluate_teacher_trajectory_quality(
        teacher_traj_texts={0: _response("Inspect the chart.")},
        samples=[{"question": "missing table"}],
        num_generations=8,
        config=ChartCoTQualityGateConfig(enabled=True, mode="diagnostic"),
    )

    assert result.verifications[0].deplot_available is False
    assert result.gate_result.eligible_indices == {0}
    assert result.gate_result.rejected_indices == set()


def test_verifier_exception_degrades_to_q2_and_records_error(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise ValueError("bad table")

    monkeypatch.setattr(gate_module, "verify_chart_cot_trajectory", fail)
    result = evaluate_teacher_trajectory_quality(
        teacher_traj_texts={0: _response("2019: 70.")},
        samples=[{"question": "q", "visual_fact_deplot": "broken"}],
        num_generations=8,
        config=ChartCoTQualityGateConfig(enabled=True, mode="diagnostic"),
    )

    assert result.verifications[0].quality == "Q2"
    assert result.verifications[0].verification_error is True
    assert "verifier_error" in result.verifications[0].reason_codes
    assert result.gate_result.metrics["verifier_error_rate"] == 1.0
