from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_direction_smoke_runner_dry_run_lists_three_variants(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_oracle_new_directions_smoke.sh",
            "--dry-run",
            "--run-id",
            "pytest_new_directions",
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
    assert "DYME_PCD_MAX_STEPS=10" in out
    assert "check_pcd_variant_smoke.py" in out
    assert "would run" in out
