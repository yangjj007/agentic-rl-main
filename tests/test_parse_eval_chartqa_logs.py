from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/test/parse_eval_chartqa_logs.py"


def test_parse_eval_summary_preserves_template_behavior_counts(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "eval_final_checkpoint.log").write_text(
        "Global samples processed: 2500 / 2500\n"
        "Current Global Mean Accuracy: 0.6012\n"
        "Output type counts: {'full_cot': 100}\n"
        "Template behavior counts: {'total': 2500, 'partial_cot_template': 12}\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(log_dir), str(summary)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    row = next(csv.DictReader(summary.open(encoding="utf-8")))
    assert row["template_behavior"] == "{'total': 2500, 'partial_cot_template': 12}"
