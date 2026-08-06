from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_positive_replay_sft_warmup_dry_run_prints_replay_dataset_and_checkpoint(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "DYME_REPLAY_TRAIN_DATASET": str(replay_dataset),
        "DYME_REPLAY_SFT_RUN_ID": "pytest_replay_warmup",
        "DYME_REPLAY_SFT_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "DYME_REPLAY_SFT_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_REPLAY_SFT_EPOCHS": "0.5",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6",
    }

    result = subprocess.run(
        ["bash", "scripts/test/run_positive_replay_sft_warmup.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    out = result.stdout
    assert "Positive replay SFT warmup" in out
    assert f"DYME_REPLAY_TRAIN_DATASET={replay_dataset}" in out
    assert "DYME_REPLAY_SFT_EPOCHS=0.5" in out
    assert "Launch plan: --num_processes 7" in out
    assert "main_sft.py --config scripts/test/config/config_positive_replay_sft.py" in out
    assert "--pretrained_model_path models/llava-0.5b-ov" in out
    assert str(tmp_path / "outputs" / "pytest_replay_warmup" / "final_checkpoint") in out
    assert "Next DyME env:" in out
    assert "DYME_STUDENT_MODEL=" in out
    assert "DYME_PRETRAINED_MODEL=" not in out


def test_positive_replay_sft_warmup_dry_run_rejects_missing_dataset(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_REPLAY_TRAIN_DATASET": str(tmp_path / "missing.json"),
        "DYME_REPLAY_SFT_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "DYME_REPLAY_SFT_LOG_ROOT": str(tmp_path / "logs"),
    }

    result = subprocess.run(
        ["bash", "scripts/test/run_positive_replay_sft_warmup.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Replay train dataset not found" in result.stderr
