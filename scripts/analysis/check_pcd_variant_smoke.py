from __future__ import annotations

import argparse
from pathlib import Path


_VARIANT_METRICS = {
    "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward": [
        "reward/eval_format_mean",
    ],
    "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay": [
        "loss/teacher_traj_effective_weight",
    ],
    "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay": [
        "reward/eval_format_mean",
        "loss/teacher_traj_effective_weight",
    ],
}


def _latest_log(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _required_metrics(variant: str) -> list[str]:
    return list(_VARIANT_METRICS.get(variant, []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a PCD variant smoke log for required metrics.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--log-dir")
    args = parser.parse_args(argv)

    log_file = Path(args.log_file) if args.log_file else None
    if log_file is None and args.log_dir:
        log_file = _latest_log(Path(args.log_dir))
    if log_file is None:
        print("missing log file: pass --log-file or --log-dir")
        return 1
    if not log_file.exists():
        print(f"missing log file: {log_file}")
        return 1

    text = log_file.read_text(encoding="utf-8", errors="replace")
    missing = [metric for metric in _required_metrics(args.variant) if metric not in text]
    if ">>> Training finished OK" not in text:
        missing.append(">>> Training finished OK")

    if missing:
        print(f"smoke check failed for {args.variant}")
        print(f"log: {log_file}")
        for item in missing:
            print(f"missing: {item}")
        return 1

    print(f"smoke check passed for {args.variant}")
    print(f"log: {log_file}")
    for metric in _required_metrics(args.variant):
        print(f"found: {metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
