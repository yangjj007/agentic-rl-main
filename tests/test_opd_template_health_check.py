from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/check_opd_template_health.py"


def _run(
    tmp_path: Path,
    rows: list[dict],
    candidate_rows: list[dict] | None = None,
    extra_args: list[str] | None = None,
) -> tuple[int, dict]:
    for row in rows:
        row.setdefault("loss", 0.0)
    log = tmp_path / "train.log"
    log.write_text("\n".join(repr(row) for row in rows), encoding="utf-8")
    command = [str(SCRIPT), str(log), "--window", "2"]
    if candidate_rows is not None:
        candidate_dir = tmp_path / "candidates"
        candidate_dir.mkdir()
        (candidate_dir / "rank0.jsonl").write_text(
            "\n".join(json.dumps(row) for row in candidate_rows),
            encoding="utf-8",
        )
        command.extend(["--candidate-dir", str(candidate_dir), "--candidate-window-steps", "2"])
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode, json.loads(result.stdout)


def test_health_check_reports_total_rows_and_writes_gate_snapshots(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "gates"
    rows = [{"rewards/accuracy/mean": 0.1} for _ in range(41)]

    code, payload = _run(
        tmp_path,
        rows,
        extra_args=["--snapshot-dir", str(snapshot_dir)],
    )

    assert code == 0
    assert payload["total_rows"] == 41
    assert json.loads((snapshot_dir / "gate_20.json").read_text())["gate"] == 20
    assert json.loads((snapshot_dir / "gate_40.json").read_text())["gate"] == 40
    assert not (snapshot_dir / "gate_60.json").exists()


def test_health_check_stops_failed_step60_joint_recovery_gate(tmp_path: Path) -> None:
    rows = [{"rewards/accuracy/mean": 0.1} for _ in range(50)]
    rows.extend(
        {
            "rewards/accuracy/mean": 0.01,
            "routing/grpo_route_rate": 0.01,
            "completions/degenerate_rate": 0.7,
        }
        for _ in range(10)
    )

    code, payload = _run(
        tmp_path,
        rows,
        extra_args=["--window", "10", "--recovery-gate-step", "60"],
    )

    assert code == 6
    assert payload["status"] == "recovery_failure"
    assert payload["recovery_gate"] == {
        "step": 60,
        "accuracy_max": 0.02,
        "grpo_route_max": 0.02,
        "degenerate_min": 0.6,
        "decision": "stop",
    }


def test_health_check_requires_all_step60_failure_conditions(tmp_path: Path) -> None:
    rows = [
        {
            "rewards/accuracy/mean": 0.03,
            "routing/grpo_route_rate": 0.01,
            "completions/degenerate_rate": 0.7,
        }
        for _ in range(60)
    ]

    code, payload = _run(
        tmp_path,
        rows,
        extra_args=["--window", "10", "--recovery-gate-step", "60"],
    )

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["recovery_gate"]["decision"] == "continue"


def test_health_check_recovery_gate_prefers_global_grpo_route(tmp_path: Path) -> None:
    rows = [{"rewards/accuracy/mean": 0.1} for _ in range(50)]
    rows.extend(
        {
            "rewards/accuracy/mean": 0.01,
            "global_signal/grpo_route_rate": 0.10,
            "routing/grpo_route_rate": 0.0,
            "completions/degenerate_rate": 0.7,
        }
        for _ in range(10)
    )

    code, payload = _run(
        tmp_path,
        rows,
        extra_args=["--window", "10", "--recovery-gate-step", "60"],
    )

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["grpo_route_rate_mean"] == pytest.approx(0.10)
    assert payload["grpo_route_global_fraction"] == 1.0
    assert payload["grpo_route_local_fallback_fraction"] == 0.0
    assert payload["recovery_gate"]["decision"] == "continue"


def test_health_check_rejects_hard_imitation_signal(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [
            {"loss/teacher_traj_effective_weight": 0.0, "routing/teacher_sft_repair_rate": 0.0},
            {"loss/teacher_traj_effective_weight": 0.1, "routing/teacher_sft_repair_rate": 0.0},
        ],
    )
    assert code == 2
    assert payload["status"] == "mechanism_violation"


@pytest.mark.parametrize(
    "metric",
    ("routing/legacy_online_sft_rate", "routing/full_hint_hard_target_rate"),
)
def test_health_check_rejects_legacy_full_hint_hard_targets(
    tmp_path: Path,
    metric: str,
) -> None:
    code, payload = _run(
        tmp_path,
        [
            {metric: 0.0},
            {metric: 0.03125},
        ],
    )
    assert code == 2
    assert payload["status"] == "mechanism_violation"
    assert payload[f"{metric}_max"] == pytest.approx(0.03125)


def test_health_check_detects_persistent_empty_template_collapse(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [
            {"completions/full_cot_template_rate": 0.95, "completions/empty_cot_skeleton_rate": 0.3},
            {"completions/full_cot_template_rate": 0.9, "completions/empty_cot_skeleton_rate": 0.25},
        ],
    )
    assert code == 3
    assert payload["status"] == "template_collapse"


def test_health_check_detects_persistent_malformed_answer_template_collapse(
    tmp_path: Path,
) -> None:
    code, payload = _run(
        tmp_path,
        [
            {
                "completions/full_cot_template_rate": 0.95,
                "completions/empty_cot_skeleton_rate": 0.05,
                "completions/malformed_answer_section_rate": 0.3,
            },
            {
                "completions/full_cot_template_rate": 0.9,
                "completions/empty_cot_skeleton_rate": 0.0,
                "completions/malformed_answer_section_rate": 0.25,
            },
        ],
    )
    assert code == 3
    assert payload["status"] == "template_collapse"
    assert payload["template_collapse_reason"] == "malformed_answer_section"


def test_health_check_accepts_substantive_full_cot(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [
            {"completions/full_cot_template_rate": 0.95, "completions/empty_cot_skeleton_rate": 0.0},
            {"completions/full_cot_template_rate": 0.9, "completions/empty_cot_skeleton_rate": 0.05},
        ],
    )
    assert code == 0
    assert payload["status"] == "ok"


def test_health_check_reports_partial_goal_without_answer_drift(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [{"completions/full_cot_template_rate": 0.0}],
        candidate_rows=[
            {"global_step": 10, "student_output": "Goal: compare\nObservation: inspect bars"},
            {"global_step": 10, "student_output": "Goal Statement:\nFind the average"},
            {"global_step": 11, "student_output": "Goal:\nReasoning: calculate"},
            {"global_step": 11, "student_output": "Answer: 7"},
        ],
    )
    assert code == 5
    assert payload["status"] == "template_drift"
    assert payload["candidate_goal_without_answer_rate"] == 0.75
    assert payload["candidate_partial_cot_template_rate"] == 0.75


def test_health_check_normalizes_escaped_multiline_candidate_previews(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [{"completions/full_cot_template_rate": 0.0}],
        candidate_rows=[
            {
                "global_step": 10,
                "student_output": (
                    "Goal: find the value\\nObservation: inspect the chart\\n"
                    "Reasoning: compare the bars\\nConclusion: the largest is seven\\nAnswer: 7"
                ),
            },
            {
                "global_step": 10,
                "student_output": (
                    "Goal: find the value\\nObservation: inspect the chart\\n"
                    "Reasoning: compare the bars\\nConclusion: choose the largest\\nAnswer:"
                ),
            },
            {
                "global_step": 11,
                "student_output": "Goal: find the value\\nReasoning: compare the bars",
            },
        ],
    )
    assert code == 0
    assert payload["candidate_full_cot_template_rate"] == pytest.approx(2 / 3)
    assert payload["candidate_partial_cot_template_rate"] == pytest.approx(1 / 3)
    assert payload["candidate_goal_without_answer_rate"] == pytest.approx(1 / 3)
    assert payload["candidate_canonical_answer_rate"] == pytest.approx(2 / 3)
    assert payload["candidate_malformed_answer_rate"] == pytest.approx(1 / 3)


def test_health_check_ignores_non_metric_json_lines(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        '{"completions/full_cot_template_rate": 1.0, "completions/empty_cot_skeleton_rate": 1.0}\n'
        + repr(
            {
                "loss": 0.0,
                "completions/full_cot_template_rate": 0.0,
                "completions/empty_cot_skeleton_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(SCRIPT), str(log), "--window", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["rows"] == 1


def test_health_check_reports_core_training_window_metrics(tmp_path: Path) -> None:
    code, payload = _run(
        tmp_path,
        [
            {
                "rewards/accuracy/mean": 0.1,
                "routing/grpo_route_rate": 0.2,
                "routing/opd_route_rate": 0.5,
                "routing/sft_route_rate": 0.3,
                "signal/grpo_zero_loss_rate": 1.0,
                "signal/group_all_wrong_rate": 0.9,
                "completions/degenerate_rate": 0.8,
                "completions/clipped_ratio": 0.7,
                "completions/eos_rate": 0.2,
            },
            {
                "rewards/accuracy/mean": 0.3,
                "routing/grpo_route_rate": 0.4,
                "routing/opd_route_rate": 0.4,
                "routing/sft_route_rate": 0.2,
                "signal/grpo_zero_loss_rate": 0.5,
                "signal/group_all_wrong_rate": 0.7,
                "completions/degenerate_rate": 0.4,
                "completions/clipped_ratio": 0.3,
                "completions/eos_rate": 0.6,
            },
        ],
    )
    assert code == 0
    assert payload["accuracy_reward_mean"] == pytest.approx(0.2)
    assert payload["grpo_route_rate_mean"] == pytest.approx(0.3)
    assert payload["opd_route_rate_mean"] == pytest.approx(0.45)
    assert payload["sft_route_rate_mean"] == pytest.approx(0.25)
    assert payload["grpo_zero_loss_rate_mean"] == pytest.approx(0.75)
    assert payload["all_wrong_rate_mean"] == pytest.approx(0.8)
    assert payload["degenerate_rate_mean"] == pytest.approx(0.6)
    assert payload["clipped_ratio_mean"] == pytest.approx(0.5)
    assert payload["eos_rate_mean"] == pytest.approx(0.4)
