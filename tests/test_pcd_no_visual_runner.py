from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pcd_no_visual_dry_run_defaults_to_local_model_paths(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_default_models",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "DYME_STUDENT_MODEL=/home/deepseek_VG/deepseek/models/llava-0.5b-ov" in out
    assert "DYME_TEACHER_MODEL=/home/deepseek_VG/deepseek/models/llava-7b-ov" in out


def test_pcd_no_visual_dry_run_canonical_speed_profile(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_canonical_speed",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run", "--speed-profile", "canonical"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "speed profile: canonical" in out
    assert "DYME_PERF_TIMING=1" in out
    assert "DYME_TEACHER_PROBE_BATCH_SIZE=8" in out
    assert "DYME_TEACHER_PROBE_MAX_PER_BATCH=0" in out
    assert "DYME_TEACHER_TRAJECTORY=1" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG=1" in out
    assert "DYME_ONLINE_SFT_TARGET=hint_answer" in out


def test_pcd_no_visual_dry_run_fast60_speed_profile(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_fast60_speed",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run", "--speed-profile", "fast60"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "speed profile: fast60" in out
    assert "DYME_PERF_TIMING=1" in out
    assert "DYME_TEACHER_PROBE_BATCH_SIZE=8" in out
    assert "DYME_TEACHER_PROBE_MAX_PER_BATCH=16" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG=0" in out
    assert "DYME_ONLINE_SFT_TARGET=answer_only" in out
