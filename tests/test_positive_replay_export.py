from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_positive_replay_buffer",
        ROOT / "scripts" / "analysis" / "export_positive_replay_buffer.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_positive_replay_export_uses_verified_hint_not_truncated_teacher_output(tmp_path: Path) -> None:
    dataset = tmp_path / "train_medium_vf_full.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What is the lowest value?",
                    "image": "/dataset/images/chart_001.png",
                    "answer": "70",
                    "hint": (
                        "Goal: Find the lowest value.\n"
                        "Observation: The chart has values 70, 72, and 77.\n"
                        "Reasoning: Compare the values and select the smallest one.\n"
                        "Conclusion: The lowest value is 70."
                    ),
                    "visual_fact_deplot": "DePlot says 999",
                }
            ]
        ),
        encoding="utf-8",
    )
    cand_dir = tmp_path / "teacher_probe_candidates"
    cand_dir.mkdir()
    cand_path = cand_dir / "rank0.jsonl"
    _write_jsonl(
        cand_path,
        [
            {
                "question": "What is the lowest value?",
                "image": "/run/images/chart_001.png",
                "reference": "Answer: 70",
                "teacher_correct": True,
                "student_correct": False,
                "parse_failed": False,
                "has_answer_flag": True,
                "teacher_output": (
                    "[Verified Hint]\n"
                    "Goal: Use the wrong clipped output.\n"
                    "Reasoning: Trust 999.\n"
                    "Answer: 999..."
                ),
                "route_reason": "all_wrong_teacher_rescue",
                "group_all_wrong": True,
                "global_step": 12,
            }
        ],
    )
    out_dir = tmp_path / "replay"
    module = _load_export_module()

    rc = module.main(
        [
            "--candidate-glob",
            str(cand_path),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--target-style",
            "student_hint_short",
        ]
    )

    assert rc == 0
    replay_rows = [
        json.loads(line)
        for line in (out_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(replay_rows) == 1
    assert replay_rows[0]["target"] == (
        "Reasoning: Compare the values and select the smallest one.\n"
        "Answer: 70"
    )
    assert "999" not in replay_rows[0]["target"]
    assert replay_rows[0]["target_quality"]["student_short_format"] is True
    assert replay_rows[0]["target_quality"]["privileged_tag_present"] is False
    assert replay_rows[0]["match_method"] == "exact"

    summary = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    assert summary[0]["emitted"] == "1"
    assert summary[0]["teacher_correct_candidates"] == "1"
    assert summary[0]["student_short_rate"] == "1.0000"
    assert summary[0]["privileged_tag_rate"] == "0.0000"

    replay_train = json.loads((out_dir / "replay_train.json").read_text(encoding="utf-8"))
    assert replay_train == [
        {
            "question": "What is the lowest value?",
            "image": "/dataset/images/chart_001.png",
            "answer": "70",
            "hint": "Reasoning: Compare the values and select the smallest one.",
            "target": (
                "Reasoning: Compare the values and select the smallest one.\n"
                "Answer: 70"
            ),
            "target_style": "student_hint_short",
            "source": "teacher_correct_positive_replay",
            "dataset_idx": 0,
        }
    ]


def test_positive_replay_export_filters_bad_candidates_and_deduplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "train_medium_vf_full.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "How many blue bars are shown?",
                    "image": "chart_002.png",
                    "answer": "4",
                    "hint": (
                        "Goal: Count the blue bars.\n"
                        "Observation: Four blue bars are visible.\n"
                        "Reasoning: Count each visible blue bar once.\n"
                        "Conclusion: There are 4 blue bars."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    cand_path = tmp_path / "rank0.jsonl"
    base = {
        "question": "How many blue bars are shown?",
        "image": "/other/root/chart_002.png",
        "reference": "Answer: 4",
        "student_correct": False,
        "teacher_output": "Answer: 4",
        "route_reason": "all_wrong_teacher_rescue",
        "group_all_wrong": True,
    }
    _write_jsonl(
        cand_path,
        [
            {**base, "teacher_correct": False, "parse_failed": False, "has_answer_flag": True},
            {**base, "teacher_correct": True, "parse_failed": True, "has_answer_flag": True},
            {**base, "teacher_correct": True, "parse_failed": False, "has_answer_flag": False},
            {**base, "teacher_correct": True, "parse_failed": False, "has_answer_flag": True, "generation_idx": 0},
            {**base, "teacher_correct": True, "parse_failed": False, "has_answer_flag": True, "generation_idx": 1},
        ],
    )
    out_dir = tmp_path / "replay"
    module = _load_export_module()

    rc = module.main(
        [
            "--candidate-glob",
            str(cand_path),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--target-style",
            "student_hint_short",
        ]
    )

    assert rc == 0
    replay_rows = [
        json.loads(line)
        for line in (out_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(replay_rows) == 1
    assert replay_rows[0]["target"] == (
        "Reasoning: Count each visible blue bar once.\n"
        "Answer: 4"
    )

    summary = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    assert summary[0]["candidate_rows"] == "5"
    assert summary[0]["teacher_correct_candidates"] == "4"
    assert summary[0]["filtered_parse_fail"] == "1"
    assert summary[0]["filtered_missing_answer_flag"] == "1"
    assert summary[0]["deduplicated"] == "1"
    assert summary[0]["emitted"] == "1"


def test_positive_replay_export_supports_answer_only_style(tmp_path: Path) -> None:
    dataset = tmp_path / "train_medium_vf_full.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What is the percentage?",
                    "image": "chart_003.png",
                    "answer": "51%",
                    "hint": "Reasoning: Read the marked percentage.",
                }
            ]
        ),
        encoding="utf-8",
    )
    cand_path = tmp_path / "rank0.jsonl"
    _write_jsonl(
        cand_path,
        [
            {
                "question": "What is the percentage?",
                "image": "chart_003.png",
                "reference": "Answer: 51%",
                "teacher_correct": True,
                "parse_failed": False,
                "has_answer_flag": True,
                "teacher_output": "Answer: 51%",
            }
        ],
    )
    out_dir = tmp_path / "replay"
    module = _load_export_module()

    rc = module.main(
        [
            "--candidate-glob",
            str(cand_path),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--target-style",
            "answer_only",
        ]
    )

    assert rc == 0
    row = json.loads((out_dir / "replay.jsonl").read_text(encoding="utf-8").strip())
    assert row["target"] == "Answer: 51%"
    assert row["target_quality"]["answer_only_format"] is True
