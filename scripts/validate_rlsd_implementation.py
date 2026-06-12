#!/usr/bin/env python3
"""Offline validation for RLSD / OPD implementation (no GPU required)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("== RLSD / OPD offline validation ==\n")

    subprocess.check_call(
        [sys.executable, "-m", "pytest", "-q", "tests/test_privileged.py",
         "tests/test_mode_router.py", "tests/test_mode_router_rlsd.py",
         "tests/test_opsd_no_gold_suffix.py", "tests/test_opsd_loss_teacher.py"],
        cwd=ROOT,
    )
    print("\n[ok] unit tests passed")

    sys.path.insert(0, str(ROOT))
    from config.loader import load_config

    rlsd = load_config("rlsd")
    assert rlsd["opsd"]["mode"] == "rlsd"
    assert "format_only" in rlsd["opsd"]["privileged_providers"] or not rlsd["opsd"]["privileged_providers"]
    print("[ok] config_rlsd_chartqa loads")

    opd = load_config("opd_7b")
    assert opd["model"].get("teacher_model_path")
    print(f"[ok] config_opd_7b_chartqa teacher={opd['model']['teacher_model_path']}")

    print("\nGPU short-run (200 steps):")
    print("  bash scripts/train_rlsd_shortrun.sh")
    print("  python scripts/compare_trimode_logs.py outputs/logs/train_trimode_*.log <rlsd_shortrun.log>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
