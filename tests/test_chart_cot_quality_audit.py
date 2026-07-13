from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_chart_cot_quality",
        ROOT / "scripts" / "audit_chart_cot_quality.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dataset(path: Path) -> None:
    rows = [
        {
            "question": "What is the maximum?",
            "answer": "77",
            "hint": (
                "Goal: Find the maximum.\n"
                "Observation: 2019: 70 and 2020: 77.\n"
                "Reasoning: Comparing the values, the maximum is 77.\n"
                "Conclusion: The maximum is 77."
            ),
            "visual_fact_deplot": json.dumps(
                {"source": "google/deplot", "parsed_table": "Year | Value\n2019 | 70\n2020 | 77"}
            ),
        },
        {
            "question": "What is the maximum in the second chart?",
            "answer": "77",
            "hint": (
                "Goal: Find the maximum.\n"
                "Observation: 2019: 70 and 2020: 77.\n"
                "Reasoning: Comparing the values, the maximum is 77.\n"
                "Conclusion: The maximum is 77."
            ),
            "visual_fact_deplot": json.dumps(
                {"source": "google/deplot", "parsed_table": "Year | Value\n2019 | 70\n2020 | 77"}
            ),
        },
        {
            "question": "What is the value in 2019?",
            "answer": "70",
            "hint": (
                "Goal: Read 2019.\n"
                "Observation: 2019: 71.\n"
                "Reasoning: Read the value.\n"
                "Conclusion: The value is 70."
            ),
            "visual_fact_deplot": json.dumps(
                {"source": "google/deplot", "parsed_table": "Year | Value\n2019 | 70"}
            ),
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_audit_writes_deterministic_quality_and_template_artifacts(tmp_path: Path) -> None:
    module = _load_audit_module()
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert module.main(
        ["--dataset", str(dataset), "--out-dir", str(first), "--max-samples", "3", "--seed", "7"]
    ) == 0
    assert module.main(
        ["--dataset", str(dataset), "--out-dir", str(second), "--max-samples", "3", "--seed", "7"]
    ) == 0

    summary = json.loads((first / "chart_cot_quality_summary.json").read_text(encoding="utf-8"))
    rows = (first / "chart_cot_quality_rows.jsonl").read_text(encoding="utf-8").splitlines()
    conflicts = (first / "chart_cot_quality_conflicts.csv").read_text(encoding="utf-8")

    assert summary["sample_count"] == 3
    assert summary["quality_counts"] == {"Q0": 1, "Q1": 0, "Q2": 0, "Q3": 2}
    assert summary["templates"]["q3"]["dominant_template_rate"] == 1.0
    assert len(rows) == 3
    assert "grounding_contradiction" in conflicts
    assert (first / "chart_cot_quality_rows.jsonl").read_bytes() == (
        second / "chart_cot_quality_rows.jsonl"
    ).read_bytes()


def test_teacher_jsonl_audit_joins_dataset_deplot_by_question(tmp_path: Path) -> None:
    module = _load_audit_module()
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "question": "What is the value in 2019?",
                "reference": "70",
                "teacher_correct": True,
                "teacher_output": (
                    "Goal: Read 2019.\nObservation: 2019: 71.\n"
                    "Reasoning: Read the value.\nConclusion: The value is 70.\nAnswer: 70"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "teacher"

    assert module.main(
        [
            "--dataset",
            str(dataset),
            "--teacher-jsonl",
            str(candidates),
            "--out-dir",
            str(out_dir),
        ]
    ) == 0

    row = json.loads((out_dir / "chart_cot_quality_rows.jsonl").read_text(encoding="utf-8"))
    assert row["deplot_available"] is True
    assert row["quality"] == "Q0"
    assert "grounding_contradiction" in row["reason_codes"]
