from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "teacher_sft_repair_target_smoke",
        ROOT / "scripts" / "analysis" / "teacher_sft_repair_target_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_teacher_sft_repair_target_smoke_shows_constrained_format_improves(tmp_path: Path) -> None:
    dataset = tmp_path / "train_medium_vf_full.json"
    records = [
        {
            "question": "What's the lowest value of red graph?",
            "image": "/chartqa_output/images/train_000001.png",
            "answer": "70",
            "hint": (
                "Goal: Find the lowest value of the red graph.\n"
                "Observation: The red values are 70, 72, and 77.\n"
                "Reasoning: Compare the values.\n"
                "Conclusion: The lowest value is 70."
            ),
        }
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")

    cand_dir = tmp_path / "teacher_probe_candidates"
    cand_dir.mkdir()
    candidate_log = cand_dir / "rank0.jsonl"
    candidate_log.write_text(
        json.dumps(
            {
                "question": records[0]["question"],
                "image": "/dev/shm/data/images/train_000001.png",
                "reference": "Answer: 70",
                "teacher_correct": True,
                "teacher_output": (
                    "[Verified Hint]\n"
                    "Goal: Find the lowest value.\n"
                    "Observation: DePlot says 999.\n"
                    "Reasoning: Compare.\n"
                    "Conclusion: The..."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "smoke"
    module = _load_smoke_module()

    rc = module.main(
        [
            "--candidate-glob",
            str(candidate_log),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "1",
        ]
    )

    assert rc == 0
    rows = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    raw = next(row for row in rows if row["kind"] == "raw_teacher")
    constrained = next(row for row in rows if row["kind"] == "constrained_target")
    assert raw["full_hint_format_rate"] == "0.0000"
    assert raw["privileged_tag_rate"] == "1.0000"
    assert constrained["full_hint_format_rate"] == "1.0000"
    assert constrained["exact_reference_answer_line_rate"] == "1.0000"
    assert constrained["privileged_tag_rate"] == "0.0000"
    assert constrained["fallback_hint_rate"] == "1.0000"


def test_teacher_sft_repair_target_smoke_reports_student_style_targets(tmp_path: Path) -> None:
    dataset = tmp_path / "train_medium_vf_full.json"
    records = [
        {
            "question": "What's the lowest value of red graph?",
            "image": "/chartqa_output/images/train_000001.png",
            "answer": "70",
            "hint": (
                "Goal: Find the lowest value of the red graph.\n"
                "Observation: The red values are 70, 72, and 77.\n"
                "Reasoning: Compare the values.\n"
                "Conclusion: The lowest value is 70."
            ),
        }
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")

    cand_dir = tmp_path / "teacher_probe_candidates"
    cand_dir.mkdir()
    candidate_log = cand_dir / "rank0.jsonl"
    candidate_log.write_text(
        json.dumps(
            {
                "question": records[0]["question"],
                "image": "/dev/shm/data/images/train_000001.png",
                "reference": "Answer: 70",
                "teacher_correct": True,
                "teacher_output": (
                    "[Verified Hint]\n"
                    "Goal: Find the lowest value.\n"
                    "Observation: The red values are 70, 72, and 77.\n"
                    "Reasoning: Compare the values and choose the smallest one.\n"
                    "Conclusion: The lowest value is 70.\n"
                    "Answer: 999"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "smoke"
    module = _load_smoke_module()

    rc = module.main(
        [
            "--candidate-glob",
            str(candidate_log),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "1",
            "--target-styles",
            "chartqa_hint,student_short,answer_only",
        ]
    )

    assert rc == 0
    rows = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    student_short = next(row for row in rows if row["kind"] == "target_student_short")
    answer_only = next(row for row in rows if row["kind"] == "target_answer_only")
    assert student_short["exact_reference_answer_line_rate"] == "1.0000"
    assert student_short["student_short_rate"] == "1.0000"
    assert student_short["full_hint_format_rate"] == "0.0000"
    assert student_short["privileged_tag_rate"] == "0.0000"
    assert answer_only["answer_only_rate"] == "1.0000"
    assert answer_only["exact_reference_answer_line_rate"] == "1.0000"

    records = [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["kind"] != "target_student_hint_short" for row in records)
