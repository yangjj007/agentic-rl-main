from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "pcd_low_score_forensics",
        ROOT / "scripts" / "analysis" / "pcd_low_score_forensics.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_controller_route_signal_prefers_global_route_snapshot() -> None:
    module = _load_module()

    value, source = module.controller_grpo_route_signal(
        {
            "global_signal/grpo_route_rate": 0.125,
            "routing/grpo_route_rate": 0.875,
        }
    )

    assert value == 0.125
    assert source == "global_signal/grpo_route_rate"


def test_controller_route_signal_falls_back_for_legacy_local_only_log() -> None:
    module = _load_module()

    value, source = module.controller_grpo_route_signal(
        {"routing/grpo_route_rate": 0.375}
    )

    assert value == 0.375
    assert source == "routing/grpo_route_rate"


def test_training_windows_expose_canonical_controller_signal_and_source_mix(
    tmp_path: Path,
) -> None:
    module = _load_module()
    log = tmp_path / "train_opd_7b_dyme_probe_20260713.log"
    log.write_text(
        "\n".join(
            [
                str(
                    {
                        "epoch": 3.95,
                        "rewards/accuracy/mean": 0.1,
                        "global_signal/grpo_route_rate": 0.125,
                        "routing/grpo_route_rate": 0.875,
                    }
                ),
                str(
                    {
                        "epoch": 4.0,
                        "rewards/accuracy/mean": 0.2,
                        "routing/grpo_route_rate": 0.375,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module.collect_training_windows("run", tmp_path)
    all_row = next(row for row in rows if row["window"] == "all")

    assert all_row["controller/grpo_route_rate"] == 0.25
    assert all_row["controller/grpo_route_global_fraction"] == 0.5
    assert all_row["controller/grpo_route_local_fallback_fraction"] == 0.5


def test_final_eval_keeps_valid_attempt_when_later_retries_fail(tmp_path: Path) -> None:
    module = _load_module()
    eval_dir = tmp_path / "eval_chartqa"
    eval_dir.mkdir()
    summary = eval_dir / "summary.csv"
    fields = [
        "label",
        "accuracy",
        "processed",
        "total",
        "exit_status",
        "errors",
        "output_types",
        "log_path",
    ]
    attempts = [
        {
            "label": "eval_final_checkpoint_bsz1_gpu0_20260709_192652",
            "accuracy": "0.5800",
            "processed": "2500",
            "total": "2500",
            "exit_status": "0",
            "errors": "",
        },
        {
            "label": "eval_final_checkpoint_bsz32_gpu0_20260709_190000",
            "accuracy": "",
            "processed": "",
            "total": "",
            "exit_status": "1",
            "errors": "CUDA out of memory",
        },
        {
            "label": "eval_final_checkpoint_bsz1_gpu0_20260709_200000",
            "accuracy": "",
            "processed": "",
            "total": "",
            "exit_status": "1",
            "errors": "Traceback",
        },
    ]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)

    rows = module.collect_eval_rows("student_hint", tmp_path)

    assert len(rows) == 1
    assert rows[0]["checkpoint"] == "final_checkpoint"
    assert rows[0]["accuracy"] == "0.5800"
    assert rows[0]["processed"] == "2500"
    assert rows[0]["source_label"] == attempts[0]["label"]


def test_final_eval_prefers_completeness_then_recency_without_accuracy_cherry_pick(
    tmp_path: Path,
) -> None:
    module = _load_module()
    eval_dir = tmp_path / "eval_chartqa"
    eval_dir.mkdir()
    summary = eval_dir / "summary.csv"
    fields = [
        "label",
        "accuracy",
        "processed",
        "total",
        "exit_status",
        "errors",
    ]
    attempts = [
        {
            "label": "eval_final_checkpoint_bsz1_gpu0_20260709_190000",
            "accuracy": "0.6000",
            "processed": "2496",
            "total": "2500",
            "exit_status": "0",
            "errors": "",
        },
        {
            "label": "eval_final_checkpoint_bsz1_gpu0_20260709_200000",
            "accuracy": "0.5000",
            "processed": "2496",
            "total": "2500",
            "exit_status": "0",
            "errors": "",
        },
        {
            "label": "eval_final_checkpoint_bsz1_gpu0_20260709_210000",
            "accuracy": "0.9900",
            "processed": "2495",
            "total": "2500",
            "exit_status": "0",
            "errors": "",
        },
    ]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)

    rows = module.collect_eval_rows("student_hint", tmp_path)

    assert len(rows) == 1
    assert rows[0]["accuracy"] == "0.5000"
    assert rows[0]["processed"] == "2496"
    assert rows[0]["source_label"] == attempts[1]["label"]
