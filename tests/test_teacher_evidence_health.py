import json
import os
import subprocess
import sys

from data_utils.chart.deplot_pipeline import build_deplot_visual_fact, placeholder_deplot_table
from scripts.check_chartqa_teacher_evidence import summarize_teacher_evidence_health


def test_summarize_teacher_evidence_health_counts_leakage_and_deplot_status():
    samples = [
        {
            "answer": "Answer: 70",
            "visual_fact": "The bar value is 70.",
            "visual_fact_deplot": build_deplot_visual_fact({"question": "q"}, "Year | Value\n2020 | 70"),
        },
        {
            "answer": "42",
            "visual_fact_hint": "Answer-derived hint says 42.",
            "visual_fact_deplot": placeholder_deplot_table({"question": "q"}),
        },
        {
            "answer": "9",
            "visual_fact": "No answer here.",
        },
    ]

    stats = summarize_teacher_evidence_health(samples)

    assert stats["total"] == 3
    assert stats["visual_fact_answer_substring"] == 2
    assert stats["deplot_real"] == 1
    assert stats["deplot_placeholder"] == 1
    assert stats["deplot_missing"] == 1
    assert stats["clean_evidence_present_rate"] == 1 / 3


def test_check_chartqa_teacher_evidence_cli_json(tmp_path):
    inp = tmp_path / "data.json"
    inp.write_text(
        json.dumps(
            [
                {
                    "answer": "70",
                    "visual_fact": "value 70",
                    "visual_fact_deplot": placeholder_deplot_table({"question": "q"}),
                }
            ]
        ),
        encoding="utf-8",
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(root, "scripts/check_chartqa_teacher_evidence.py"),
            "--input",
            str(inp),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stats = json.loads(proc.stdout)
    assert stats["visual_fact_answer_substring_rate"] == 1.0
    assert stats["deplot_placeholder_rate"] == 1.0
