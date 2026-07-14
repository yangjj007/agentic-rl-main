import json
import os
import subprocess
import sys

from scripts.repair_chartqa_visual_facts import repair_chartqa_visual_fact_records


def test_repair_nulls_hint_derived_visual_fact_fields_and_preserves_deplot():
    rows = [
        {
            "question": "What is the lowest value?",
            "hint": "Goal: Find the lowest value.\nReasoning: compare values.\nAnswer: 70",
            "answer": "70",
            "visual_fact": "Goal: Find the lowest value.\nReasoning: compare values.\nAnswer: 70",
            "visual_fact_hint": "Goal: Find the lowest value.\nReasoning: compare values.\nAnswer: 70",
            "visual_fact_deplot": '{"source": "deplot", "parsed_table": "Year | Value\\n2019 | 70"}',
        }
    ]

    stats = repair_chartqa_visual_fact_records(rows)

    assert rows[0]["visual_fact"] is None
    assert rows[0]["visual_fact_hint"] is None
    assert rows[0]["hint"].startswith("Goal:")
    assert rows[0]["visual_fact_deplot"]
    assert stats["records"] == 1
    assert stats["visual_fact_null"] == 1
    assert stats["visual_fact_hint_null"] == 1


def test_repair_adds_missing_visual_fact_fields_as_null():
    rows = [{"question": "q", "hint": "h", "answer": "a"}]

    repair_chartqa_visual_fact_records(rows)

    assert "visual_fact" in rows[0]
    assert rows[0]["visual_fact"] is None
    assert "visual_fact_hint" in rows[0]
    assert rows[0]["visual_fact_hint"] is None


def test_repair_cli_in_place(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inp = tmp_path / "chartqa.json"
    inp.write_text(
        json.dumps(
            [
                {
                    "question": "q",
                    "hint": "Goal: solve\nAnswer: 42",
                    "answer": "42",
                    "visual_fact": "Goal: solve\nAnswer: 42",
                    "visual_fact_hint": "Goal: solve\nAnswer: 42",
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            os.path.join(root, "scripts", "repair_chartqa_visual_facts.py"),
            "--input",
            str(inp),
            "--in-place",
        ],
        check=True,
    )

    data = json.loads(inp.read_text(encoding="utf-8"))
    assert data[0]["visual_fact"] is None
    assert data[0]["visual_fact_hint"] is None
