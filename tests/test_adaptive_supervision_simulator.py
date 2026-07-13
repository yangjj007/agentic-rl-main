from __future__ import annotations

import json

import pytest

from scripts.simulate_adaptive_supervision import load_signals_jsonl, run_scenario


def test_cold_start_keeps_full_supervision() -> None:
    states = run_scenario("cold_start")

    assert states[-1]["mastery"] == 0.0
    assert states[-1]["supervision"] == 1.0
    assert states[-1]["opsd_weight"] == 1.5
    assert states[-1]["teacher_traj_weight"] == 0.5
    assert states[-1]["opd_max_per_prompt"] == 8


def test_gradual_learning_reduces_supervision_monotonically() -> None:
    states = run_scenario("gradual_learning")
    supervision = [state["supervision"] for state in states]

    assert supervision == sorted(supervision, reverse=True)
    assert supervision[-1] < supervision[0]
    assert states[-1]["mastery"] > 0.0


def test_regression_does_not_restore_teacher_or_opsd_supervision() -> None:
    states = run_scenario("regression")
    peak = min(state["supervision"] for state in states)

    assert states[-1]["supervision"] == peak
    assert states[-1]["mastery"] == max(state["mastery"] for state in states)


def test_single_spike_is_damped_by_conservative_ema() -> None:
    states = run_scenario("single_spike")

    assert states[0]["readiness"] < 0.02
    assert states[0]["supervision"] > 0.99
    assert states[-1]["opd_max_per_prompt"] == 8


def test_scenario_output_is_deterministic() -> None:
    assert run_scenario("gradual_learning") == run_scenario("gradual_learning")


def test_load_signals_jsonl(tmp_path) -> None:
    path = tmp_path / "signals.jsonl"
    rows = [
        {"mixed_rate": 0.1, "zero_loss_rate": 0.9},
        {"mixed_rate": 0.4, "zero_loss_rate": 0.2},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert load_signals_jsonl(path) == [(0.1, 0.9), (0.4, 0.2)]


@pytest.mark.parametrize(
    "row",
    [
        {"mixed_rate": 0.1},
        {"mixed_rate": float("nan"), "zero_loss_rate": 0.5},
    ],
)
def test_load_signals_jsonl_rejects_invalid_rows(tmp_path, row) -> None:
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        load_signals_jsonl(path)
