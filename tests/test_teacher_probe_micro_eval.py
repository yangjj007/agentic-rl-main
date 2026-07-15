from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


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


def test_micro_eval_reasoned_deplot_control_is_structured_and_gold_hidden(tmp_path: Path) -> None:
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
            "reasoned_deplot_only",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"reasoned_deplot_only"}
    assert all(row["response_prefix"].startswith("Goal:") for row in preview_rows)
    assert all(row["response_prefix"].rstrip().endswith("Observation:") for row in preview_rows)
    for row in preview_rows:
        prompt = row["prompt"]
        assert "Goal:" in prompt
        assert "Observation:" in prompt
        assert "Reasoning:" in prompt
        assert "Conclusion:" in prompt
        assert "Answer:" in prompt
        assert "[Visual Facts - DePlot]" in prompt
        assert "[Verified Hint]" not in prompt
        assert "[Reference Answer]" not in prompt
        assert "secret dataset hint" not in prompt


def test_micro_eval_visual_reasoned_control_uses_image_native_gold_hidden_prompt(tmp_path: Path) -> None:
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
            "visual_reasoned_answer",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_reasoned_answer"}
    for row in preview_rows:
        prompt = row["prompt"]
        assert "full chart image" in prompt.lower()
        assert "question type" in prompt.lower()
        assert "[Visual Facts - DePlot]" not in prompt
        assert "[Verified Hint]" not in prompt
        assert "[Reference Answer]" not in prompt


def test_micro_eval_visual_chain_control_uses_response_prefix_and_no_deplot(tmp_path: Path) -> None:
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
            "visual_chain_of_charts",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_chain_of_charts"}
    for row in preview_rows:
        assert row["response_prefix"] == "Task:"
        prompt = row["prompt"]
        assert "Visual Evidence:" in prompt
        assert "Computation:" in prompt
        assert "[Visual Facts - DePlot]" not in prompt
        assert "[Verified Hint]" not in prompt
        assert "[Reference Answer]" not in prompt


def test_micro_eval_visual_zoom_control_builds_dual_image_jobs(monkeypatch, tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "micro"
    image = Image.new("RGB", (80, 80), "white")
    monkeypatch.setattr(module, "load_rgb", lambda value: image)

    samples, _ = module._prepare_samples(
        module.parse_args(
            [
                "--candidate-glob",
                str(candidate_log),
                "--dataset",
                str(dataset),
                "--out-dir",
                str(out_dir),
                "--max-samples",
                "2",
            ]
        )
    )
    controls = module._selected_controls("visual_zoom_short_answer")
    for sample in samples:
        sample["image"] = image

    jobs, previews = module._build_jobs(samples, controls, load_images=True)

    assert len(jobs) == 2
    assert all(len(job["images"]) == 2 for job in jobs)
    assert all("two chart images" in row["prompt"].lower() for row in previews)
    assert all("[Visual Facts - DePlot]" not in row["prompt"] for row in previews)


def test_micro_eval_visual_answer_prefix_control_exports_answer_prefix(tmp_path: Path) -> None:
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
            "visual_answer_prefix",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_answer_prefix"}
    for row in preview_rows:
        assert row["response_prefix"] == "Answer:"
        assert "Return only the final answer text" in row["prompt"]
        assert "[Visual Facts - DePlot]" not in row["prompt"]
        assert "[Reference Answer]" not in row["prompt"]


def test_micro_eval_visual_answer_prefix_numeric_control_exports_numeric_rules(tmp_path: Path) -> None:
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
            "visual_answer_prefix_numeric",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_answer_prefix_numeric"}
    for row in preview_rows:
        assert row["response_prefix"] == "Answer:"
        assert "Use Arabic numerals for counts and numeric answers" in row["prompt"]
        assert "include a percent sign" in row["prompt"]
        assert "[Visual Facts - DePlot]" not in row["prompt"]
        assert "[Reference Answer]" not in row["prompt"]


def test_micro_eval_deplot_answer_prefix_control_exports_deplot_and_answer_prefix(tmp_path: Path) -> None:
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
            "visual_deplot_answer_prefix",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_deplot_answer_prefix"}
    for row in preview_rows:
        assert row["response_prefix"] == "Answer:"
        assert "[Visual Facts - DePlot]" in row["prompt"]
        assert "fallible OCR" in row["prompt"]
        assert "Return only the final answer text" in row["prompt"]
        assert "[Reference Answer]" not in row["prompt"]


def test_micro_eval_visual_operation_answer_prefix_control_exports_answer_prefix(tmp_path: Path) -> None:
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
            "visual_operation_answer_prefix",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"visual_operation_answer_prefix"}
    for row in preview_rows:
        assert row["response_prefix"] == "Answer:"
        assert "silently classify the required operation" in row["prompt"]
        assert "Select only the operands named by the question" in row["prompt"]
        assert "count only data entries" in row["prompt"]
        assert "[Visual Facts - DePlot]" not in row["prompt"]
        assert "[Reference Answer]" not in row["prompt"]


def test_micro_eval_deplot_operation_answer_prefix_control_exports_deplot_and_answer_prefix(
    tmp_path: Path,
) -> None:
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
            "deplot_operation_answer_prefix",
            "--dry-run",
        ]
    )

    assert rc == 0
    preview_rows = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control"] for row in preview_rows} == {"deplot_operation_answer_prefix"}
    for row in preview_rows:
        assert row["response_prefix"] == "Answer:"
        assert "[Visual Facts - DePlot]" in row["prompt"]
        assert "fallible OCR" in row["prompt"]
        assert "Check row/column orientation" in row["prompt"]
        assert "perform the requested arithmetic" in row["prompt"]
        assert "[Reference Answer]" not in row["prompt"]


def test_verifier_first_correct_selector_accepts_later_correct_control() -> None:
    module = _load_micro_eval_module()
    records = [
        {
            "control": "visual_answer_prefix",
            "sample_idx": 0,
            "scope": "all_wrong",
            "qtype": "sum",
            "question": "What is the sum?",
            "image_basename": "chart.png",
            "reference": "12",
            "teacher_output": "Answer: 10",
            "parsed_answer": "10",
            "teacher_correct": False,
            "parse_failed": False,
        },
        {
            "control": "deplot_operation_answer_prefix",
            "sample_idx": 0,
            "scope": "all_wrong",
            "qtype": "sum",
            "question": "What is the sum?",
            "image_basename": "chart.png",
            "reference": "12",
            "teacher_output": "Answer: 12",
            "parsed_answer": "12",
            "teacher_correct": True,
            "parse_failed": False,
        },
    ]

    selected = module._select_verifier_first_correct(
        records,
        ["visual_answer_prefix", "deplot_operation_answer_prefix"],
    )

    assert len(selected) == 1
    assert selected[0]["status"] == "accepted"
    assert selected[0]["selected_control"] == "deplot_operation_answer_prefix"
    assert selected[0]["selected_output"] == "Answer: 12"
    assert selected[0]["parsed_answer"] == "12"
    assert selected[0]["attempt_count"] == 2
    assert selected[0]["oracle_any_attempt_correct"] is True


def test_verifier_first_correct_selector_abstains_when_all_controls_wrong() -> None:
    module = _load_micro_eval_module()
    records = [
        {
            "control": "visual_answer_prefix",
            "sample_idx": 0,
            "scope": "all_wrong",
            "qtype": "sum",
            "question": "What is the sum?",
            "image_basename": "chart.png",
            "reference": "12",
            "teacher_output": "Answer: 10",
            "parsed_answer": "10",
            "teacher_correct": False,
            "parse_failed": False,
        },
        {
            "control": "deplot_operation_answer_prefix",
            "sample_idx": 0,
            "scope": "all_wrong",
            "qtype": "sum",
            "question": "What is the sum?",
            "image_basename": "chart.png",
            "reference": "12",
            "teacher_output": "Answer: 11",
            "parsed_answer": "11",
            "teacher_correct": False,
            "parse_failed": False,
        },
    ]

    selected = module._select_verifier_first_correct(
        records,
        ["visual_answer_prefix", "deplot_operation_answer_prefix"],
    )

    assert len(selected) == 1
    assert selected[0]["status"] == "abstained"
    assert selected[0]["selected_control"] == "visual_answer_prefix"
    assert selected[0]["selected_output"] == "Answer: 10"
    assert selected[0]["teacher_correct"] is False
    assert selected[0]["attempt_count"] == 2
    assert selected[0]["oracle_any_attempt_correct"] is False


def test_micro_eval_selection_policy_writes_selected_artifacts(tmp_path: Path) -> None:
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
            "visual_answer_prefix,deplot_operation_answer_prefix",
            "--selection-policy",
            "verifier_first_correct",
            "--fake-teacher",
        ]
    )

    assert rc == 0
    selected_rows = [
        json.loads(line)
        for line in (out_dir / "selected_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(selected_rows) == 2
    assert {row["status"] for row in selected_rows} == {"accepted"}
    assert {row["selected_control"] for row in selected_rows} == {
        "deplot_operation_answer_prefix"
    }
    assert all(row["attempt_count"] == 2 for row in selected_rows)

    summary_rows = list(csv.DictReader((out_dir / "selected_summary.csv").open()))
    assert summary_rows[0]["selected_coverage_rate"] == "1.0000"
    assert summary_rows[0]["selected_precision"] == "1.0000"
    assert summary_rows[0]["oracle_union_accuracy"] == "1.0000"
    assert "deplot_operation_answer_prefix" in summary_rows[0]["accepted_by_control"]


def test_micro_eval_verifier_early_stop_skips_late_controls_after_acceptance(tmp_path: Path) -> None:
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
            "visual_answer_prefix,deplot_operation_answer_prefix,reasoned_deplot_only",
            "--selection-policy",
            "verifier_first_correct",
            "--execution-policy",
            "verifier_early_stop",
            "--fake-teacher",
        ]
    )

    assert rc == 0
    records = [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["control"] for row in records] == [
        "visual_answer_prefix",
        "visual_answer_prefix",
        "deplot_operation_answer_prefix",
        "deplot_operation_answer_prefix",
    ]

    selected_rows = [
        json.loads(line)
        for line in (out_dir / "selected_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["selected_control"] for row in selected_rows} == {
        "deplot_operation_answer_prefix"
    }
    assert all(row["attempt_count"] == 2 for row in selected_rows)

    summary_rows = {
        row["control"]: row
        for row in csv.DictReader((out_dir / "summary.csv").open())
    }
    assert summary_rows["reasoned_deplot_only"]["n"] == "0"
    selected_summary = list(csv.DictReader((out_dir / "selected_summary.csv").open()))[0]
    assert selected_summary["mean_attempts"] == "2.00"


def test_micro_eval_verifier_early_stop_requires_verifier_selector(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)

    with pytest.raises(ValueError, match="verifier_early_stop requires"):
        module.main(
            [
                "--candidate-glob",
                str(candidate_log),
                "--dataset",
                str(dataset),
                "--out-dir",
                str(tmp_path / "micro"),
                "--max-samples",
                "2",
                "--controls",
                "visual_answer_prefix,deplot_operation_answer_prefix",
                "--execution-policy",
                "verifier_early_stop",
                "--fake-teacher",
            ]
        )


def test_reasoned_canonicalization_jobs_use_draft_but_not_reference() -> None:
    module = _load_micro_eval_module()
    jobs = [
        {
            "control": "reasoned_deplot_only",
            "sample_idx": 0,
            "sample": {"answer": "SECRET_GOLD", "question": "What is the value?"},
            "prompt": "What is the value?\n\n[Visual Facts - DePlot]\nA | 12",
            "images": [],
            "response_prefix": "Goal: generic\nObservation:",
        }
    ]
    controls = {"reasoned_deplot_only": module.CONTROL_SPECS["reasoned_deplot_only"]}

    canonical_jobs, canonical_indices = module._build_canonicalization_jobs(
        jobs,
        ["Goal: inspect.\nConclusion: The value is 12."],
        controls,
    )

    assert canonical_indices == [0]
    assert len(canonical_jobs) == 1
    canonical = canonical_jobs[0]
    assert canonical["response_prefix"] == "Answer:"
    assert "[Teacher Draft Reasoning]" in canonical["prompt"]
    assert "The value is 12" in canonical["prompt"]
    assert "[Visual Facts - DePlot]" in canonical["prompt"]
    assert "SECRET_GOLD" not in canonical["prompt"]
    assert "Reference Answer" not in canonical["prompt"]


def test_micro_eval_rejects_legacy_oracle_controls() -> None:
    module = _load_micro_eval_module()

    with pytest.raises(ValueError, match="oracle_hint_v4_deplot"):
        module._selected_controls("oracle_hint_v4_deplot")


def test_micro_eval_defaults_to_long_oracle_teacher_budget() -> None:
    module = _load_micro_eval_module()
    args = module.parse_args([])

    assert args.max_new_tokens == 500
    assert args.teacher_backend == "llava_onevision"


def test_micro_eval_accepts_qwen25vl_teacher_backend() -> None:
    module = _load_micro_eval_module()
    args = module.parse_args(["--teacher-backend", "qwen25vl"])

    assert args.teacher_backend == "qwen25vl"


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


def test_chartqa_harness_initial_jobs_keep_full_image_and_hide_gold() -> None:
    module = _load_micro_eval_module()
    image = Image.new("RGB", (32, 32), "white")
    samples = [
        {
            "question": "What is the value?",
            "prompt": "Question: What is the value?",
            "image": image,
            "answer": "SECRET_GOLD",
            "hint": "SECRET_HINT",
            "visual_fact_deplot": {
                "source": "google/deplot",
                "parsed_table": "Year | Value\n2020 | 42",
            },
        }
    ]

    jobs, previews = module._build_chartqa_harness_initial_jobs(samples, load_images=True)

    assert len(jobs) == 2
    assert {job["configuration"] for job in jobs} == {"visual_base", "visual_deplot"}
    assert all(len(job["images"]) == 1 for job in jobs)
    assert all(job["images"][0] is image for job in jobs)
    assert len(previews) == 2
    for row in previews:
        assert "SECRET_GOLD" not in row["prompt"]
        assert "SECRET_HINT" not in row["prompt"]
        assert row["native_image_available"] is True
    base = next(row for row in previews if row["configuration"] == "visual_base")
    deplot = next(row for row in previews if row["configuration"] == "visual_deplot")
    assert "[Visual Facts - DePlot]" not in base["prompt"]
    assert "[Visual Facts - DePlot]" in deplot["prompt"]


def test_chartqa_harness_fake_run_writes_reference_free_traces(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "harness"

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
            "--harness",
            "chartqa_recoverable",
            "--fake-teacher",
        ]
    )

    assert rc == 0
    assert (out_dir / "harness_attempts.jsonl").exists()
    assert (out_dir / "harness_records.jsonl").exists()
    assert (out_dir / "harness_summary.csv").exists()

    previews = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["configuration"] for row in previews} == {
        "visual_base",
        "visual_deplot",
        "visual_recovery",
    }
    for row in previews:
        assert "[Reference Answer]" not in row["prompt"]
        assert "[Verified Hint]" not in row["prompt"]
        assert "SECRET_GOLD" not in row["prompt"]

    records = [
        json.loads(line)
        for line in (out_dir / "harness_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert sum(row["recovery_triggered"] for row in records) == 1
    assert all(row["status"] == "accepted" for row in records)
    runtime_payload = json.dumps(
        [{"attempts": row["attempts"], "decision": row["decision"]} for row in records]
    ).lower()
    assert "reference" not in runtime_payload
    assert "teacher_correct" not in runtime_payload

    with (out_dir / "harness_summary.csv").open(encoding="utf-8", newline="") as f:
        summary = next(csv.DictReader(f))
    assert summary["n"] == "2"
    assert summary["base_accuracy"] == "1.0000"
    assert summary["deplot_accuracy"] == "0.5000"
    assert summary["selected_accuracy"] == "1.0000"
    assert summary["recovery_trigger_rate"] == "0.5000"
    assert summary["recovered_accuracy"] == "1.0000"
    assert summary["abstain_rate"] == "0.0000"
    assert summary["mean_attempts"] == "2.50"


def test_chartqa_closed_loop_fake_run_writes_state_trace(tmp_path: Path) -> None:
    module = _load_micro_eval_module()
    dataset, candidate_log = _write_dataset_and_candidates(tmp_path)
    out_dir = tmp_path / "closed_loop"

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
            "--harness",
            "chartqa_closed_loop_recovery",
            "--fake-teacher",
        ]
    )

    assert rc == 0
    assert (out_dir / "closed_loop_records.jsonl").exists()
    assert (out_dir / "closed_loop_attempts.jsonl").exists()
    assert (out_dir / "closed_loop_summary.csv").exists()
    assert (out_dir / "prompt_previews.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    previews = [
        json.loads(line)
        for line in (out_dir / "prompt_previews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "visual_answer" in {row["action"] for row in previews}
    assert "visual_operation_recovery" in {row["action"] for row in previews}
    assert "deplot_operation_recovery" in {row["action"] for row in previews}
    for row in previews:
        assert "[Reference Answer]" not in row["prompt"]
        assert "[Verified Hint]" not in row["prompt"]
        assert "The lowest value is 70" not in row["prompt"]
        assert "four years" not in row["prompt"]

    records = [
        json.loads(line)
        for line in (out_dir / "closed_loop_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(row["status"] == "accepted" for row in records)
    assert any(row["attempt_count"] > 1 for row in records)
    for row in records:
        assert isinstance(row["events"], list)
        assert isinstance(row["actions"], list)
        assert row["selected_action"] in row["actions"]
        assert row["oracle_any_attempt_correct"] is True

    attempts = [
        json.loads(line)
        for line in (out_dir / "closed_loop_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["action"] for row in attempts} >= {
        "visual_answer",
        "visual_operation_recovery",
        "deplot_operation_recovery",
    }
    assert any(row["event"] == "operation_recovery_required" for row in attempts)

    summary = next(csv.DictReader((out_dir / "closed_loop_summary.csv").open()))
    assert summary["n"] == "2"
    assert summary["selected_accuracy"] == "1.0000"
    assert summary["accepted_coverage_rate"] == "1.0000"
    assert "deplot_operation_recovery" in summary["accepted_by_action"]

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["controller"] == "integrated_closed_loop_recovery_controller"
    assert manifest["max_teacher_attempts"] == len(manifest["actions"])


def test_closed_loop_recovery_controller_executes_observe_act_loop() -> None:
    module = _load_micro_eval_module()
    sample = {
        "question": "For how many years has the line been over 50?",
        "prompt": "Question: For how many years has the line been over 50?",
        "image": None,
        "answer": "4",
        "_answer_flag": "Answer:",
        "visual_fact_deplot": {
            "source": "google/deplot",
            "parsed_table": "Year | Value\n2018 | 60\n2019 | 70\n2020 | 80\n2021 | 90",
        },
    }

    controller = module.ClosedLoopRecoveryController(
        sample_idx=7,
        sample=sample,
        load_images=False,
    )

    first_job, first_preview = controller.build_next_job()
    assert first_job["action"] == "visual_answer"
    assert first_job["max_new_tokens"] == 32
    assert first_preview["controller"] == "integrated_closed_loop_recovery_controller"

    first_event = controller.observe("Answer: 3")
    assert first_event["event"] == "operation_recovery_required"
    assert controller.status == "active"
    assert controller.next_action == "visual_operation_recovery"

    recovery_job, recovery_preview = controller.build_next_job()
    assert recovery_job["action"] == "visual_operation_recovery"
    assert recovery_job["max_new_tokens"] == 96
    assert recovery_job["event"] == "operation_recovery_required"
    assert recovery_preview["controller_step"] == 2

    accepted_event = controller.observe("Answer: 4")
    assert accepted_event["event"] == "accepted"

    record = controller.to_record()
    assert record["controller"] == "integrated_closed_loop_recovery_controller"
    assert record["status"] == "accepted"
    assert record["attempt_count"] == 2
    assert record["selected_action"] == "visual_operation_recovery"
    assert record["oracle_any_attempt_correct"] is True


def test_closed_loop_verifier_events_drive_recovery_actions() -> None:
    module = _load_micro_eval_module()
    count_sample = {
        "question": "For how many years has the line been over 50?",
        "answer": "4",
        "_answer_flag": "Answer:",
    }
    parse_sample = {
        "question": "What is the lowest value?",
        "answer": "70",
        "_answer_flag": "Answer:",
    }
    lookup_sample = {
        "question": "What is the lowest value?",
        "answer": "70",
        "_answer_flag": "Answer:",
    }

    parse_event = module._closed_loop_verifier_event("", parse_sample)
    count_event = module._closed_loop_verifier_event("Answer: 3", count_sample)
    lookup_event = module._closed_loop_verifier_event("Answer: 100", lookup_sample)

    assert parse_event["event"] == "canonical_repair_required"
    assert count_event["event"] == "operation_recovery_required"
    assert lookup_event["event"] == "evidence_recovery_required"
    assert (
        module._closed_loop_next_action("canonical_repair_required", "other", ["visual_answer"])
        == "reasoned_recovery"
    )
    assert (
        module._closed_loop_next_action("operation_recovery_required", "count", ["visual_answer"])
        == "visual_operation_recovery"
    )
    assert (
        module._closed_loop_next_action("evidence_recovery_required", "other", ["visual_answer"])
        == "visual_operation_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "count",
            ["visual_answer", "visual_operation_recovery"],
        )
        == "deplot_operation_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "count",
            ["visual_answer", "visual_operation_recovery", "deplot_operation_recovery"],
        )
        == "executable_deplot_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "count",
            [
                "visual_answer",
                "visual_operation_recovery",
                "deplot_operation_recovery",
                "executable_deplot_recovery",
            ],
        )
        == "reasoned_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "count",
            [
                "visual_answer",
                "visual_operation_recovery",
                "deplot_operation_recovery",
                "executable_deplot_recovery",
                "reasoned_recovery",
            ],
        )
        == "target_phrase_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "count",
            [
                "visual_answer",
                "visual_operation_recovery",
                "deplot_operation_recovery",
                "executable_deplot_recovery",
                "reasoned_recovery",
                "target_phrase_recovery",
            ],
        )
        == "arithmetic_recovery"
    )


def test_closed_loop_schedules_target_phrase_recovery_before_arithmetic() -> None:
    module = _load_micro_eval_module()
    attempted = [
        "visual_answer",
        "visual_operation_recovery",
        "deplot_operation_recovery",
        "reasoned_recovery",
    ]

    assert (
        module._closed_loop_next_action(
            "evidence_recovery_required",
            "other",
            attempted[:3],
        )
        == "executable_deplot_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "evidence_recovery_required",
            "other",
            attempted[:3] + ["executable_deplot_recovery"],
        )
        == "reasoned_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "evidence_recovery_required",
            "other",
            attempted + ["executable_deplot_recovery"],
        )
        == "target_phrase_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "percent",
            attempted + ["target_phrase_recovery", "executable_deplot_recovery"],
        )
        == "arithmetic_recovery"
    )
    assert (
        module._closed_loop_next_action(
            "operation_recovery_required",
            "percent",
            attempted + ["target_phrase_recovery", "executable_deplot_recovery", "arithmetic_recovery"],
        )
        == "scale_unit_recovery"
    )


def test_closed_loop_operation_recovery_prompt_avoids_wrong_answer_anchor() -> None:
    module = _load_micro_eval_module()
    sample = {
        "question": "What is the difference between A and B?",
        "prompt": "Question: What is the difference between A and B?",
        "image": None,
        "answer": "42",
        "visual_fact_deplot": {
            "source": "google/deplot",
            "parsed_table": "Year | A | B\n2020 | 72 | 30",
        },
    }

    job, preview = module._build_closed_loop_job(
        sample_idx=0,
        sample=sample,
        action="deplot_operation_recovery",
        attempts=[{"teacher_output": "Answer: WRONG_ANCHOR"}],
        event="operation_recovery_required",
        load_images=False,
    )

    assert job["response_prefix"] == "Answer:"
    assert "Closed-Loop Recovery State" not in preview["prompt"]
    assert "WRONG_ANCHOR" not in preview["prompt"]


def test_closed_loop_target_phrase_recovery_prompt_focuses_requested_label_without_gold() -> None:
    module = _load_micro_eval_module()
    sample = {
        "question": "What's the percentage value of U.S. adults who have heard a lot about facial recognition technology?",
        "prompt": "Question: What's the percentage value of U.S. adults who have heard a lot about facial recognition technology?",
        "image": None,
        "answer": "25",
        "visual_fact_deplot": {
            "source": "google/deplot",
            "parsed_table": "Response | Percent\nA little | 61\nA lot | 25\nNothing at all | 14",
        },
    }

    job, preview = module._build_closed_loop_job(
        sample_idx=0,
        sample=sample,
        action="target_phrase_recovery",
        attempts=[{"teacher_output": "Answer: A little 61"}],
        event="evidence_recovery_required",
        load_images=False,
    )

    assert job["control"] == "target_phrase_recovery"
    assert job["response_prefix"] == "Target phrase: A lot\nEvidence:"
    assert "Target phrase" in preview["prompt"]
    assert "legend/color" in preview["prompt"]
    assert "A little 61" not in preview["prompt"]
    assert "Answer: 25" not in preview["prompt"]
    assert "[Reference Answer]" not in preview["prompt"]


def test_closed_loop_scale_unit_recovery_uses_prior_attempts_without_gold() -> None:
    module = _load_micro_eval_module()
    sample = {
        "question": "What's the value of largest bar?",
        "prompt": "Question: What's the value of largest bar?",
        "image": None,
        "answer": "0.0886",
        "visual_fact_deplot": {
            "source": "google/deplot",
            "parsed_table": "Measure | Value\n9-year average | 8.86\nExtreme one-day precipitation | 4.0",
        },
    }

    job, preview = module._build_closed_loop_job(
        sample_idx=0,
        sample=sample,
        action="scale_unit_recovery",
        attempts=[
            {"action": "visual_answer", "teacher_output": "Answer: 8.86"},
            {
                "action": "reasoned_recovery",
                "teacher_output": "Observation: the chart is in percent.\nAnswer: 8.86",
            },
        ],
        event="operation_recovery_required",
        load_images=False,
    )

    assert job["control"] == "scale_unit_recovery"
    assert job["response_prefix"] == "Answer:"
    assert job["max_new_tokens"] == 48
    assert "percent sign" in preview["prompt"]
    assert "[Prior Teacher Attempts]" in preview["prompt"]
    assert "8.86" in preview["prompt"]
    assert "0.0886" not in preview["prompt"]
    assert "[Reference Answer]" not in preview["prompt"]


def test_closed_loop_executable_deplot_recovery_embeds_computed_evidence_without_gold() -> None:
    module = _load_micro_eval_module()
    sample = {
        "question": "What is the sum of the bars which is above 200 ?",
        "prompt": "Question: What is the sum of the bars which is above 200 ?",
        "image": Image.new("RGB", (32, 32), "white"),
        "answer": "923",
        "visual_fact_deplot": {
            "source": "google/deplot",
            "parsed_table": (
                "Characteristic | Number of drugs and vaccines\n"
                "Preclinical | 707\n"
                "Public Clinical | 98\n"
                "Phase II Clinical | 216"
            ),
        },
    }

    job, preview = module._build_closed_loop_job(
        sample_idx=0,
        sample=sample,
        action="executable_deplot_recovery",
        attempts=[{"teacher_output": "Answer: 707"}],
        event="operation_recovery_required",
        load_images=True,
    )

    assert job["control"] == "executable_deplot_recovery"
    assert job["images"] == []
    assert preview["loaded_image_count"] == 0
    assert job["response_prefix"] == "Answer: 923"
    assert job["max_new_tokens"] == 1
    assert "[Executable DePlot Recovery]" in preview["prompt"]
    assert "Operation: threshold_sum" in preview["prompt"]
    assert "Candidate answer: 923" in preview["prompt"]
    assert "[Reference Answer]" not in preview["prompt"]
    assert "[Verified Hint]" not in preview["prompt"]


def test_closed_loop_arithmetic_recovery_uses_answer_canonicalization() -> None:
    module = _load_micro_eval_module()
    controls = module._closed_loop_generation_controls()
    assert controls["executable_deplot_recovery"].get("canonicalize_draft") is not True
    jobs = [
        {
            "control": "arithmetic_recovery",
            "action": "arithmetic_recovery",
            "sample_idx": 0,
            "sample": {"answer": "SECRET_GOLD", "question": "Which year changed most?"},
            "prompt": "Which year changed most?\n\n[Visual Facts - DePlot]\n2014/15 | 100",
            "images": [],
            "response_prefix": "",
        }
    ]

    canonical_jobs, canonical_indices = module._build_canonicalization_jobs(
        jobs,
        ["The greatest change occurs between 2013/14 and 2014/15."],
        controls,
    )

    assert canonical_indices == [0]
    assert len(canonical_jobs) == 1
    assert canonical_jobs[0]["response_prefix"] == "Answer:"
    assert "2014/15" in canonical_jobs[0]["prompt"]
    assert "SECRET_GOLD" not in canonical_jobs[0]["prompt"]
