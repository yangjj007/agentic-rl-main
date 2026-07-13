from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_direction_eval_runner_dry_run_lists_variants_and_checkpoints(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "pcd-no-visual"),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/eval_pcd_oracle_new_directions.sh",
            "--dry-run",
            "--run-id",
            "pytest_new_directions_4epoch",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward" in out
    assert "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay" in out
    assert "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay" in out
    assert "checkpoint-147" in out
    assert "checkpoint-294" in out
    assert "checkpoint-441" in out
    assert "checkpoint-588" in out
    assert "final_checkpoint" in out
    assert "parse_eval_chartqa_logs.py" in out
    assert "would eval" in out
