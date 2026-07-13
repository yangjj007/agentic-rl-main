from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/test/run_dyme_matched_4epoch.sh"
CONFIG = ROOT / "scripts/test/config/config_dyme_matched.py"
FULL_CONFIG = ROOT / "scripts/test/config/config_dyme_full_matched.py"


def test_matched_config_uses_clrc_optimizer_decode_and_dyme_routing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DYME_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("DYME_STUDENT_MODEL", "/tmp/local-llava")
    monkeypatch.setenv("DYME_NUM_TRAIN_EPOCHS", "4")
    monkeypatch.setenv("DYME_MAX_STEPS", "1")
    monkeypatch.setenv("DYME_SAVE_STRATEGY", "steps")
    monkeypatch.setenv("DYME_SAVE_STEPS", "50")
    monkeypatch.setenv("DYME_SAVE_TOTAL_LIMIT", "3")

    spec = importlib.util.spec_from_file_location("config_dyme_matched", CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = module.CONFIG
    args = config["training"]["dyme_args"]

    assert config["model"]["pretrained_model_path"] == "/tmp/local-llava"
    assert config["opsd"]["enabled"] is False
    assert config["opsd"]["gate"]["online_sft_on_all_wrong"] is True
    assert config["opsd"]["gate"].get("disable_online_sft_slots", False) is False
    assert config["opsd"]["visual_supervision"]["enabled"] is False
    assert args["output_dir"] == str(tmp_path / "out")
    assert args["num_train_epochs"] == 4
    assert args["max_steps"] == 1
    assert args["save_strategy"] == "steps"
    assert args["save_steps"] == 50
    assert args["save_total_limit"] == 3
    assert args["learning_rate"] == 5e-5
    assert args["warmup_steps"] == 50
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 16
    assert args["num_generations"] == 8
    assert args["max_completion_length"] == 96
    assert args["temperature"] == 0.5
    assert args["repetition_penalty"] == 1.5


def test_runner_dry_run_is_isolated_and_includes_automatic_eval(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_DYME_RUN_ID": "pytest_matched_dyme",
        "DYME_DYME_OUTPUT_ROOT": str(tmp_path / "out-root"),
        "DYME_DYME_LOG_ROOT": str(tmp_path / "log-root"),
    }
    result = subprocess.run(
        ["bash", str(RUNNER), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "run id: pytest_matched_dyme" in out
    assert str(tmp_path / "out-root" / "pytest_matched_dyme" / "pure_dyme_matched") in out
    assert "scripts/test/config/config_dyme_matched.py" in out
    assert "/home/deepseek_VG/.conda/envs/dyme/bin/python -m accelerate.commands.launch" in out
    assert "--num_processes 8" in out
    assert "DYME_NUM_TRAIN_EPOCHS=4" in out
    assert "DYME_SAVE_STRATEGY=steps" in out
    assert "DYME_SAVE_STEPS=50" in out
    assert "DYME_SAVE_TOTAL_LIMIT=3" in out
    assert f"DYME_STUDENT_MODEL={ROOT}/models/llava-0.5b-ov" in out
    assert "--opsd_enabled" not in out
    assert "DYME_EVAL_BATCH_SIZE=1" in out
    assert "-m eval.eval_chartqa" in out
    assert "parse_eval_chartqa_logs.py" in out


def test_runner_train_stage_dry_run_skips_eval_for_smoke(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_DYME_RUN_ID": "pytest_train_only",
        "DYME_DYME_OUTPUT_ROOT": str(tmp_path / "out-root"),
        "DYME_DYME_LOG_ROOT": str(tmp_path / "log-root"),
        "DYME_DYME_MAX_STEPS": "1",
    }
    result = subprocess.run(
        ["bash", str(RUNNER), "--dry-run", "--stages", "train", "--epochs", "10"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "stages: train" in out
    assert "DYME_NUM_TRAIN_EPOCHS=10" in out
    assert "DYME_MAX_STEPS=1" in out
    assert "TRAIN:" in out
    assert "EVAL:" not in out
    assert "PARSE:" not in out


def test_runner_eval_stage_dry_run_skips_train(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_DYME_RUN_ID": "pytest_eval_only",
        "DYME_DYME_OUTPUT_ROOT": str(tmp_path / "out-root"),
        "DYME_DYME_LOG_ROOT": str(tmp_path / "log-root"),
    }
    result = subprocess.run(
        ["bash", str(RUNNER), "--dry-run", "--stages", "eval"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "stages: eval" in out
    assert "TRAIN:" not in out
    assert "EVAL:" in out
    assert "PARSE:" in out


def test_full_config_only_adds_visual_supervision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DYME_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("DYME_VISUAL_CHECKER", "1")
    monkeypatch.setenv("DYME_VISUAL_REFINER", "1")
    monkeypatch.setenv("DYME_VISUAL_PREFETCH_IC", "1")

    spec = importlib.util.spec_from_file_location("config_dyme_full_matched", FULL_CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = module.CONFIG
    visual = config["opsd"]["visual_supervision"]

    assert config["opsd"]["enabled"] is False
    assert visual["enabled"] is True
    assert visual["checker"]["enabled"] is True
    assert visual["refiner"]["enabled"] is True
    assert visual["prefetch_ic"] is True
    assert config["training"]["dyme_args"]["learning_rate"] == 5e-5
    assert config["training"]["dyme_args"]["num_generations"] == 8

    expected = copy.deepcopy(module.pure.CONFIG)
    expected["opsd"]["visual_supervision"] = visual
    assert config == expected


def test_runner_full_variant_keeps_budget_and_enables_visual_supervision(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_DYME_RUN_ID": "pytest_full_dyme",
        "DYME_DYME_OUTPUT_ROOT": str(tmp_path / "out-root"),
        "DYME_DYME_LOG_ROOT": str(tmp_path / "log-root"),
    }
    result = subprocess.run(
        ["bash", str(RUNNER), "--variant", "full", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "Matched Full DyME 4epoch run" in out
    assert str(tmp_path / "out-root" / "pytest_full_dyme" / "full_dyme_matched") in out
    assert "scripts/test/config/config_dyme_full_matched.py" in out
    assert "DYME_VISUAL_CHECKER=1" in out
    assert "DYME_VISUAL_REFINER=1" in out
    assert "DYME_VISUAL_PREFETCH_IC=1" in out
    assert "DYME_SAVE_STEPS=50" in out
    assert "DYME_EVAL_BATCH_SIZE=1" in out
