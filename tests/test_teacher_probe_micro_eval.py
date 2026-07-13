from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_micro_eval_module():
    spec = importlib.util.spec_from_file_location(
        "teacher_probe_micro_eval",
        ROOT / "scripts" / "analysis" / "teacher_probe_micro_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_dataset_and_candidates(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "train_medium_vf_full.json"
    records = [
        {
            "question": "What's the lowest value of red graph?",
            "question_wo_prompt": "What's the lowest value of red graph?",
            "prompt": "Question: What's the lowest value of red graph?",
            "image": "/chartqa_output/images/train_000001.png",
            "answer": "70",
            "hint": (
                "Goal: Find the lowest value of the red graph.\n"
                "Observation: The red values are 70, 72, and 77.\n"
                "Reasoning: Compare the values.\n"
                "Conclusion: The lowest value is 70."
            ),
            "visual_fact_deplot": {
                "source": "google/deplot",
                "parsed_table": "Year | Value\n2019 | 999\n2020 | 72",
            },
        },
        {
            "question": "For how many years has the line been over 50?",
            "question_wo_prompt": "For how many years has the line been over 50?",
            "prompt": "Question: For how many years has the line been over 50?",
            "image": "/chartqa_output/images/train_000002.png",
            "answer": "4",
            "hint": (
                "Goal: Determine the number of years the line has been over 50.\n"
                "Observation: The values exceed 50 for four years.\n"
                "Reasoning: Count the qualifying years.\n"
                "Conclusion: The line has been over 50 for 4 years."
            ),
            "visual_fact_deplot": {
                "source": "google/deplot",
                "parsed_table": "Year | Values\n2013 | 0\n2014 | 0",
            },
        },
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")

    cand_dir = tmp_path / "teacher_probe_candidates"
    cand_dir.mkdir()
    candidate_log = cand_dir / "rank0.jsonl"
    candidates = [
        {
            "question": records[0]["question"],
            "image": "/dev/shm/data/images/train_000001.png",
            "reference": "Answer: 70",
            "is_all_wrong_probe_candidate": True,
            "is_mixed_wrong_probe_candidate": False,
        },
        {
            "question": records[1]["question"],
            "image": "/dev/shm/data/images/train_000002.png",
            "reference": "Answer: 4",
            "is_all_wrong_probe_candidate": False,
            "is_mixed_wrong_probe_candidate": True,
        },
    ]
    candidate_log.write_text(
        "\n".join(json.dumps(row) for row in candidates) + "\n",
        encoding="utf-8",
    )
    return dataset, candidate_log


def test_micro_eval_dry_run_builds_baseline_and_oracle_prompts(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "micro"

    rc = module.main(
        [
            "--candidate-glob",
            str(candidate_log),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "2",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    controls = {row["control"] for row in preview_rows}
    assert controls == {"baseline_deplot_only", "oracle_hint_deplot"}
    baseline_prompt = next(row["prompt"] for row in preview_rows if row["control"] == "baseline_deplot_only")
    oracle_row = next(row for row in preview_rows if row["control"] == "oracle_hint_deplot")
    oracle_prompt = oracle_row["prompt"]
    oracle_prefix = oracle_row["response_prefix"]
    assert "[Verified Hint]" not in baseline_prompt
    assert "[Reference Answer]" not in baseline_prompt
    assert "999" in oracle_prompt
    assert oracle_prompt.index("[Visual Facts - DePlot]") < oracle_prompt.index("[Verified Hint]")
    assert "Do not output a short answer only." in oracle_prompt
    assert "Do not transcribe the DePlot table" in oracle_prompt
    assert "[Teacher Response Prefix]" not in oracle_prompt
    assert oracle_prefix.startswith("Goal: Find the lowest value")
    assert "\nObservation: The red values are 70, 72, and 77." in oracle_prefix
    assert oracle_prefix.rstrip().endswith("Answer:")
    assert "Answer: 70" not in oracle_prefix


def test_micro_eval_fake_teacher_outputs_summary_metrics(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "micro"

    rc = module.main(
        [
            "--candidate-glob",
            str(candidate_log),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "2",
            "--fake-teacher",
        ]
    )

    assert rc == 0
    rows = {row["control"]: row for row in csv.DictReader((out_dir / "summary.csv").open())}
    assert rows["baseline_deplot_only"]["n"] == "2"
    assert rows["baseline_deplot_only"]["teacher_correct_rate"] == "0.0000"
    assert rows["oracle_hint_deplot"]["teacher_correct_rate"] == "1.0000"
    assert rows["oracle_hint_deplot"]["parse_fail_rate"] == "0.0000"
    assert rows["oracle_hint_deplot"]["answer_flag_rate"] == "1.0000"
    assert rows["oracle_hint_deplot"]["full_hint_format_rate"] == "1.0000"
    assert rows["oracle_hint_deplot"]["answer_last_line_rate"] == "1.0000"
    assert rows["oracle_hint_deplot"]["exact_reference_answer_line_rate"] == "1.0000"
    assert rows["baseline_deplot_only"]["full_hint_format_rate"] == "0.0000"

    by_scope = list(csv.DictReader((out_dir / "by_scope.csv").open()))
    assert {row["scope"] for row in by_scope} == {"all_wrong", "mixed_wrong"}
    assert "full_hint_format_rate" in by_scope[0]
    by_qtype = list(csv.DictReader((out_dir / "by_qtype.csv").open()))
    assert "exact_reference_answer_line_rate" in by_qtype[0]
    records = [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    oracle_outputs = [r["teacher_output"] for r in records if r["control"] == "oracle_hint_deplot"]
    assert all("Answer:" in output for output in oracle_outputs)
    oracle_records = [r for r in records if r["control"] == "oracle_hint_deplot"]
    assert all(r["full_hint_format"] is True for r in oracle_records)
    assert all(r["answer_last_line"] is True for r in oracle_records)
    assert all(r["exact_reference_answer_line"] is True for r in oracle_records)


def test_micro_eval_controls_option_selects_official_oracle_only(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "micro"

    rc = module.main(
        [
            "--candidate-glob",
            str(candidate_log),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "2",
            "--controls",
            "baseline_deplot_only,oracle_hint_deplot",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    controls = {row["control"] for row in preview_rows}
    assert controls == {"baseline_deplot_only", "oracle_hint_deplot"}


def test_micro_eval_rejects_legacy_oracle_controls() -> None:
    module = _load_micro_eval_module()

    with pytest.raises(ValueError, match="oracle_hint_v4_deplot"):
        module._selected_controls("oracle_hint_v4_deplot")


def test_micro_eval_defaults_to_long_oracle_teacher_budget() -> None:
    module = _load_micro_eval_module()
    args = module.parse_args([])

    assert args.max_new_tokens == 500


def test_micro_eval_progress_iter_uses_tqdm_for_real_teacher(monkeypatch) -> None:
    module = _load_micro_eval_module()
    calls = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(module, "tqdm", fake_tqdm)
    chunks = list(module._iter_progress_chunks([1, 2, 3, 4, 5], chunk_size=2, desc="teacher"))

    assert chunks == [[1, 2], [3, 4], [5]]
    assert calls
    assert calls[0]["total"] == 3
    assert calls[0]["desc"] == "teacher"
