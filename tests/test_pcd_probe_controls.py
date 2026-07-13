from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "pcd_probe_controls",
        ROOT / "scripts" / "analysis" / "pcd_probe_controls.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest_with_candidates(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    cand_path = run_dir / "teacher_probe_candidates" / "rank0.jsonl"
    cand_path.parent.mkdir(parents=True)
    records = [
        {
            "question": "What is the highest bar?",
            "reference": "42",
            "student_output": "Answer: 41",
            "teacher_output": "Reasoning. Answer: 42",
            "teacher_correct": True,
            "parse_failed": False,
            "teacher_output_is_placeholder": False,
            "teacher_output_word_count": 5,
            "privileged": {"visual_fact_deplot_status": "real"},
        },
        {
            "question": "What is the lowest bar?",
            "reference": "7",
            "student_output": "Answer: 9",
            "teacher_output": "",
            "teacher_correct": False,
            "parse_failed": True,
            "teacher_output_is_placeholder": True,
            "teacher_output_word_count": 0,
            "privileged": {"visual_fact_deplot_status": "real"},
        },
    ]
    cand_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    manifest = tmp_path / "run_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "variant,role,run_dir,train_log,eval_log,candidate_log_glob,config_path,enabled",
                f"deplot_no_vs_opd_pcd,pcd,{run_dir},,,{cand_path},,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_pcd_probe_controls_dry_run_builds_prompts_without_gpu(tmp_path: Path) -> None:
    module = _load_probe_module()
    manifest = _write_manifest_with_candidates(tmp_path)
    out = tmp_path / "table3_recoverability_controls.csv"

    rc = module.main(
        [
            "--manifest",
            str(manifest),
            "--variant",
            "deplot_no_vs_opd_pcd",
            "--modes",
            "teacher_only,completion_conditioned,shuffled_completion",
            "--max-samples",
            "2",
            "--out",
            str(out),
            "--dry-run",
        ]
    )

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["control"] for row in rows] == [
        "teacher_only",
        "completion_conditioned",
        "shuffled_completion",
    ]
    assert all(row["status"] == "dry_run_prompt_only" for row in rows)

    preview = out.with_name(out.stem + "_prompts.jsonl")
    prompt_rows = [json.loads(line) for line in preview.read_text(encoding="utf-8").splitlines()]
    assert len(prompt_rows) == 6
    teacher_only_prompt = next(row["prompt"] for row in prompt_rows if row["control"] == "teacher_only")
    conditioned_prompt = next(row["prompt"] for row in prompt_rows if row["control"] == "completion_conditioned")
    shuffled_prompt = next(row["prompt"] for row in prompt_rows if row["control"] == "shuffled_completion")
    assert "Student completion" not in teacher_only_prompt
    assert "Student completion" in conditioned_prompt
    assert "Answer: 41" in conditioned_prompt
    assert "Student completion" in shuffled_prompt
    assert "Answer: 41" not in shuffled_prompt


def test_pcd_probe_controls_completion_conditioned_uses_candidate_log_stats(tmp_path: Path) -> None:
    module = _load_probe_module()
    manifest = _write_manifest_with_candidates(tmp_path)
    out = tmp_path / "table3_recoverability_controls.csv"

    rc = module.main(
        [
            "--manifest",
            str(manifest),
            "--variant",
            "deplot_no_vs_opd_pcd",
            "--modes",
            "teacher_only,completion_conditioned",
            "--max-samples",
            "2",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    rows = {row["control"]: row for row in csv.DictReader(out.open(encoding="utf-8"))}
    assert rows["completion_conditioned"]["teacher_correct_rate"] == "0.5000"
    assert rows["completion_conditioned"]["parse_fail_rate"] == "0.5000"
    assert rows["completion_conditioned"]["placeholder_rate"] == "0.5000"
    assert rows["completion_conditioned"]["deplot_real_rate"] == "1.0000"
    assert rows["completion_conditioned"]["status"] == "from_candidate_log"
    assert rows["teacher_only"]["teacher_correct_rate"] == ""
    assert rows["teacher_only"]["status"] == "missing_offline_teacher_run"

    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "\\toprule" in md
    assert "\\rowcolor{gray!10} completion_conditioned" in md
