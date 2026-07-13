from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_replay_warmup_then_pcd_dry_run_threads_warmup_checkpoint_into_dyme(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6",
        "DYME_REPLAY_TRAIN_DATASET": str(replay_dataset),
        "DYME_CHAIN_RUN_ID": "pytest_replay_chain",
        "DYME_REPLAY_SFT_OUTPUT_ROOT": str(tmp_path / "warmup"),
        "DYME_REPLAY_SFT_LOG_ROOT": str(tmp_path / "warmup_logs"),
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "pcd_out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "pcd_logs"),
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_replay_warmup_then_pcd_4epoch.sh",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    out = result.stdout.replace("\\,", ",")
    warmup_ckpt = tmp_path / "warmup" / "pytest_replay_chain_warmup" / "final_checkpoint"
    assert "Replay warmup -> DyME PCD chain" in out
    assert "Stage 1: positive replay SFT warmup" in out
    assert "Stage 2: DyME PCD training" in out
    assert f"warmup checkpoint: {warmup_ckpt}" in out
    assert f"DYME_STUDENT_MODEL={warmup_ckpt}" in out
    assert "DYME_PCD_RUN_ID=pytest_replay_chain_dyme" in out
    assert (
        "variant: "
        "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition"
    ) in out
    assert "DYME_EFFECTIVE_GROUP_FILTER=1" in out
    assert "DYME_POSITIVE_REPLAY=0" in out
    assert "DYME_POSITIVE_REPLAY_WEIGHT=0.0" in out
    assert "DYME_POSITIVE_REPLAY_BATCH_SIZE=0" in out
    assert "DYME_OPSD_FINAL_WEIGHT=0.0" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP=0" in out


def test_replay_warmup_then_pcd_dry_run_rejects_missing_replay_dataset(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_REPLAY_TRAIN_DATASET": str(tmp_path / "missing.json"),
        "DYME_CHAIN_RUN_ID": "pytest_missing_replay",
    }

    result = subprocess.run(
        ["bash", "scripts/test/run_replay_warmup_then_pcd_4epoch.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Replay train dataset not found" in result.stderr
