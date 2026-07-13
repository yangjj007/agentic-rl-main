#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def _checkpoint_step(label: str) -> int:
    match = re.search(r"checkpoint-(\d+)", label)
    if match:
        return int(match.group(1))
    return 10**9


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    acc = re.findall(r"Current Global Mean Accuracy:\s*([0-9.]+)", text)
    processed = re.findall(r"Global samples processed:\s*(\d+)\s*/\s*(\d+)", text)
    output_types = re.findall(r"Output type counts:\s*(\{[^\n]*\})", text)
    template_behavior = re.findall(r"Template behavior counts:\s*(\{[^\n]*\})", text)
    status = re.findall(r"\[eval-exit\].*status=(\d+)", text)
    errors = []
    for pattern in ("Traceback", "CUDA out of memory", "RuntimeError", "Failed to load dataset"):
        if pattern in text:
            errors.append(pattern)
    return {
        "label": path.stem,
        "accuracy": acc[-1] if acc else "",
        "processed": processed[-1][0] if processed else "",
        "total": processed[-1][1] if processed else "",
        "output_types": output_types[-1] if output_types else "",
        "template_behavior": template_behavior[-1] if template_behavior else "",
        "exit_status": status[-1] if status else "",
        "errors": ";".join(errors),
        "log_path": str(path),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: parse_eval_chartqa_logs.py LOG_DIR SUMMARY_CSV", file=sys.stderr)
        return 2
    log_dir = Path(sys.argv[1])
    summary_csv = Path(sys.argv[2])
    rows = [parse_log(path) for path in sorted(log_dir.glob("*.log"), key=lambda p: _checkpoint_step(p.stem))]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "accuracy",
        "processed",
        "total",
        "exit_status",
        "errors",
        "output_types",
        "template_behavior",
        "log_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["accuracy"]]
    if completed:
        best = max(completed, key=lambda row: float(row["accuracy"]))
        print(f"[summary] wrote {summary_csv}")
        print(
            f"[best] label={best['label']} accuracy={best['accuracy']} "
            f"processed={best['processed']}/{best['total']} log={best['log_path']}"
        )
    else:
        print(f"[summary] wrote {summary_csv}; no completed accuracy found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
