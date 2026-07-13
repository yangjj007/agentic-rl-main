import json
from pathlib import Path

from scripts.analysis.check_frozen_run_env import audit_run_env, main


EXPECTED_ENV = {
    "DYME_NUM_TRAIN_EPOCHS": "4",
    "DYME_FAST_NUM_TRAIN_EPOCHS": "4",
    "DYME_SAVE_STRATEGY": "steps",
    "DYME_SAVE_STEPS": "50",
    "DYME_SAVE_TOTAL_LIMIT": "3",
    "DYME_TEACHER_TRAJECTORY": "0",
    "DYME_DISABLE_ONLINE_SFT_SLOTS": "1",
    "DYME_ONLINE_SFT_ON_ALL_WRONG": "0",
    "DYME_TEACHER_PROBE_FAILURE_ROUTE": "mixed_grpo_all_wrong_skip",
    "DYME_ADAPTIVE_SUPERVISION": "1",
    "DYME_ADAPTIVE_READINESS_SOURCE": "global_grpo_route",
    "DYME_ADAPTIVE_EMA_ALPHA": "0.10",
    "DYME_ADAPTIVE_TARGET_READINESS": "0.30",
    "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT": "1.5",
    "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT": "0.5",
    "DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT": "0.0",
    "DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT": "0.0",
    "DYME_ADAPTIVE_OPSD_INITIAL_CAP": "8",
    "DYME_ADAPTIVE_OPSD_FINAL_CAP": "2",
    "DYME_EFFECTIVE_SAMPLING": "1",
    "DYME_EFFECTIVE_SAMPLING_AFTER_STEP": "0",
    "DYME_EFFECTIVE_SAMPLING_START_PROGRESS": "0.0",
    "DYME_OPSD_SKIP_DEGENERATE": "0",
    "DYME_GLOBAL_SIGNAL_LOGGING": "1",
    "DYME_EVAL_FORMAT_REWARD": "0",
}


def write_run_env(path: Path, env: dict[str, str]) -> None:
    path.write_text(json.dumps({"argv": [], "cwd": "/tmp", "env": env}), encoding="utf-8")


def test_audit_accepts_exact_frozen_environment(tmp_path: Path) -> None:
    path = tmp_path / "run_env.json"
    write_run_env(path, EXPECTED_ENV)

    result = audit_run_env(path)

    assert result["status"] == "ok"
    assert result["violations"] == []
    assert result["checked"] == len(EXPECTED_ENV)


def test_audit_reports_missing_and_mismatched_values(tmp_path: Path) -> None:
    path = tmp_path / "run_env.json"
    env = dict(EXPECTED_ENV)
    env.pop("DYME_DISABLE_ONLINE_SFT_SLOTS")
    env["DYME_EFFECTIVE_SAMPLING_AFTER_STEP"] = "294"
    write_run_env(path, env)

    result = audit_run_env(path)

    assert result["status"] == "violation"
    assert {
        (row["key"], row["expected"], row["actual"])
        for row in result["violations"]
    } == {
        ("DYME_DISABLE_ONLINE_SFT_SLOTS", "1", None),
        ("DYME_EFFECTIVE_SAMPLING_AFTER_STEP", "0", "294"),
    }


def test_cli_returns_nonzero_and_writes_json_on_violation(tmp_path: Path, capsys) -> None:
    path = tmp_path / "run_env.json"
    write_run_env(path, {})

    assert main([str(path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "violation"
