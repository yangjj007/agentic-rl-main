"""Tests for image-checker timing smoke analysis."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.image_checker_timing_report import build_report, format_report


def test_image_checker_timing_report_summarizes_log_and_route_artifacts(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "[VISUAL-BATCH][2026-07-24 10:00:00][rank=0/1][global_step=0] "
                "generate_summary | visual/ic_latency_ms=10.0 | "
                "visual/checker_latency_ms=100.0 | visual/refiner_latency_ms=30.0 | "
                "visual/ic_calls=2.0 | visual/checker_calls=4.0 | visual/refiner_calls=3.0 | "
                "visual/ic_batch_calls=1.0 | visual/checker_batch_calls=2.0 | "
                "visual/refiner_batch_calls=1.0 | visual/teacher_batch_calls=4.0 | "
                "visual/checker_high=1.0 | visual/checker_medium=1.0 | visual/checker_low=2.0 | "
                "visual/checker_parse_failure=0.0 | visual/checker_image_missing=0.0 | "
                "visual/checker_aux_evidence_used=0.0",
                "[VISUAL-BATCH][2026-07-24 10:00:01][rank=0/1][global_step=1] "
                "generate_summary | visual/checker_latency_ms=50.0 | "
                "visual/checker_calls=2.0 | visual/checker_batch_calls=1.0 | "
                "visual/teacher_batch_calls=1.0 | visual/checker_low=2.0",
                "[OPSD-DETAIL][2026-07-24 10:00:02][rank=0/1][step=1][every=1][routing] "
                "mode routing summary | completion_mode_counts={\"GRPO\": 3, \"SFT\": 29}",
            ]
        ),
        encoding="utf-8",
    )

    step_dir = tmp_path / "out" / "visual_supervision" / "step_0"
    step_dir.mkdir(parents=True)
    rows = [
        {"kind": "checker", "sample_idx": 0, "label": "high", "score": 1.0},
        {"kind": "checker", "sample_idx": 1, "label": "medium", "score": 0.5},
        {"kind": "checker", "sample_idx": 2, "label": "low", "score": 0.0},
        {"kind": "route", "sample_idx": 0, "route": "grpo", "checker_score": 1.0, "answer_reward": 0.0},
        {"kind": "route", "sample_idx": 1, "route": "opsd", "checker_score": 0.5, "answer_reward": 1.0},
        {"kind": "route", "sample_idx": 2, "route": "sft_replaced", "checker_score": 0.0, "answer_reward": 0.0},
    ]
    (step_dir / "rank0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_report(log_files=[log_path], output_dir=tmp_path / "out")

    assert report["log_batches"] == 2
    assert report["visual/checker_latency_ms"] == 150.0
    assert report["visual/checker_calls"] == 6.0
    assert report["visual/checker_batch_calls"] == 3.0
    assert report["derived/checker_ms_per_call"] == 25.0
    assert report["derived/checker_ms_per_batch"] == 50.0
    assert report["checker_labels"] == {"high": 1, "low": 1, "medium": 1}
    assert report["routes"] == {"grpo": 1, "opsd": 1, "sft_replaced": 1}
    assert report["trainer_completion_modes"] == {"GRPO": 3, "SFT": 29}
    assert report["route_x_checker"] == {"grpo:high": 1, "opsd:medium": 1, "sft_replaced:low": 1}
    assert report["false_high_count"] == 1

    text = format_report(report)
    assert "[IMAGE-CHECKER-TIMING]" in text
    assert "checker_ms_per_call=25.0" in text
    assert "trainer_completion_modes=GRPO=3,SFT=29" in text
    assert "route_x_checker=grpo:high=1,opsd:medium=1,sft_replaced:low=1" in text
