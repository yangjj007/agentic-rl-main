from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_artifact_module():
    spec = importlib.util.spec_from_file_location(
        "pcd_paper_artifacts",
        ROOT / "scripts" / "analysis" / "pcd_paper_artifacts.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_train_log(path: Path, *, pcd: bool, va: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, 4):
        rows.append(
            {
                "loss": 1.0 - step * 0.1,
                "reward": 0.1 * step,
                "rewards/accuracy/mean": 0.2 + step * 0.01,
                "rewards/format/mean": 0.8,
                "signal/reward_std_mean": 0.03 * step,
                "signal/group_all_wrong_rate": 0.5 if pcd else 0.4,
                "signal/group_mixed_rate": 0.25,
                "signal/reward_std_lt_0_01_rate": 0.1,
                "signal/reward_std_lt_0_05_rate": 0.6,
                "signal/reward_std_lt_0_10_rate": 0.8,
                "routing/grpo_route_rate": 0.2,
                "routing/opd_route_rate": 0.3 if pcd else 0.15,
                "routing/sft_route_rate": 0.5,
                "routing/teacher_probe_candidate_rate": 0.4 if pcd else 0.2,
                "routing/teacher_probe_correct_rate": 0.25 if pcd else 0.1,
                "routing/teacher_probe_gold_suffix_rate": 0.0,
                "routing/teacher_probe_deplot_real_rate": 1.0,
                "routing/teacher_probe_skipped_no_evidence_rate": 0.0,
                "teacher_probe/generated_tokens_mean": 42.0,
                "teacher_probe/generated_tokens_p95": 80.0,
                "loss/opsd_effective_weight": 1.7 if va else 1.0,
                "loss/opsd_adaptive_multiplier": 1.7 if va else 1.0,
                "routing/total_completion_count": 8,
                "routing/wrong_completion_count": 5,
                "routing/probe_candidate_count": 3 if pcd else 2,
                "routing/teacher_correct_count": 2 if pcd else 1,
                "routing/opd_route_count": 2 if pcd else 1,
                "routing/sft_route_count": 4,
                "routing/grpo_route_count": 2,
                "epoch": step / 10,
            }
        )
    path.write_text("\n".join(str(row) for row in rows) + "\n", encoding="utf-8")


def _write_eval_log(path: Path, accuracy: float) -> None:
    path.write_text(
        "\n".join(
            [
                "Global samples processed: 2500 / 2500",
                f"Current Global Mean Accuracy: {accuracy:.4f}",
                "Output type counts: {'full_cot': 2300, 'answer_flag': 100, 'other': 100}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_candidate_log(path: Path, *, all_wrong: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "global_step": 1,
            "final_route": "opd",
            "teacher_correct": True,
            "student_correct": False,
            "parse_failed": False,
            "teacher_output_is_placeholder": False,
            "teacher_output_word_count": 7,
            "group_has_correct": not all_wrong,
            "group_all_wrong": all_wrong,
            "group_reward_std": 0.0 if all_wrong else 0.2,
            "answer_type": "numeric",
            "is_all_wrong_probe_candidate": all_wrong,
            "is_mixed_wrong_probe_candidate": not all_wrong,
            "route_reason": "all_wrong_teacher_rescue" if all_wrong else "mixed_wrong_teacher_probe",
            "question": "How many units?",
            "reference": "42",
            "student_output": "Answer: 41",
            "teacher_output": "Reasoning. Answer: 42",
            "image": "/tmp/chart.png",
            "privileged": {"visual_fact_deplot_status": "real", "evidence_status": {"evidence_present": True}},
        },
        {
            "global_step": 2,
            "final_route": "sft",
            "teacher_correct": False,
            "student_correct": False,
            "parse_failed": True,
            "teacher_output_is_placeholder": True,
            "teacher_output_word_count": 0,
            "group_has_correct": not all_wrong,
            "group_all_wrong": all_wrong,
            "group_reward_std": 0.0 if all_wrong else 0.2,
            "answer_type": "numeric",
            "is_all_wrong_probe_candidate": all_wrong,
            "is_mixed_wrong_probe_candidate": not all_wrong,
            "route_reason": "all_wrong_teacher_rescue" if all_wrong else "mixed_wrong_teacher_probe",
            "question": "How many units?",
            "reference": "42",
            "student_output": "Answer: 10",
            "teacher_output": "",
            "image": "/tmp/chart.png",
            "privileged": {"visual_fact_deplot_status": "real", "evidence_status": {"evidence_present": True}},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")


def test_pcd_paper_artifacts_make_all_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_artifact_module()
    out_dir = tmp_path / "paper_artifacts"
    manifest = tmp_path / "run_manifest.csv"
    variants = [
        ("deplot_no_vs_opd", "anchor", False, False, 0.60),
        ("deplot_no_vs_opd_va", "va", False, True, 0.61),
        ("deplot_no_vs_opd_pcd", "pcd", True, False, 0.63),
        ("deplot_no_vs_opd_va_pcd", "va_pcd", True, True, 0.62),
    ]

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "role",
                "run_dir",
                "train_log",
                "eval_log",
                "candidate_log_glob",
                "config_path",
                "enabled",
            ],
        )
        writer.writeheader()
        for variant, role, pcd, va, accuracy in variants:
            run_dir = tmp_path / variant
            train_log = run_dir / "train.log"
            eval_log = run_dir / "eval.log"
            candidate_log = run_dir / "teacher_probe_candidates" / "rank0.jsonl"
            config_path = run_dir / "config.py"
            _write_train_log(train_log, pcd=pcd, va=va)
            _write_eval_log(eval_log, accuracy)
            _write_candidate_log(candidate_log, all_wrong=pcd)
            config_path.write_text("visual_supervision=False\nuse_deplot=True\n", encoding="utf-8")
            writer.writerow(
                {
                    "variant": variant,
                    "role": role,
                    "run_dir": run_dir,
                    "train_log": train_log,
                    "eval_log": eval_log,
                    "candidate_log_glob": str(candidate_log),
                    "config_path": config_path,
                    "enabled": "1",
                }
            )

    rc = module.main(["--manifest", str(manifest), "--out-dir", str(out_dir), "--make", "all"])

    assert rc == 0
    for name in [
        "fig0_training_basics.csv",
        "fig1_motivation.csv",
        "fig4_training_dynamics.csv",
        "fig5_teacher_rescue_funnel.csv",
        "fig6_va_vs_pcd_diagnosis.csv",
        "table1_method_positioning.md",
        "table3_recoverability_controls.csv",
        "table3_recoverability_controls.md",
        "table4_routing_antileakage.csv",
        "table4_routing_antileakage.md",
        "paper_argument_report.md",
        "chart_review_report.md",
        "artifacts_manifest.json",
        "index.html",
        "data_quality_report.md",
    ]:
        assert (out_dir / name).exists(), name
    for name in [
        "fig0_training_basics.png",
        "fig1_motivation.png",
        "fig4_training_dynamics.png",
        "fig5_teacher_rescue_funnel.png",
        "fig6_va_vs_pcd_diagnosis.png",
    ]:
        assert (out_dir / name).stat().st_size > 0, name
    for name in [
        "fig2_pcd_opd_flow.png",
        "fig2_pcd_opd_flow.pdf",
        "fig2_pcd_opd_flow.svg",
    ]:
        assert not (out_dir / name).exists(), name

    fig5_rows = list(csv.DictReader((out_dir / "fig5_teacher_rescue_funnel.csv").open(encoding="utf-8")))
    pcd_row = next(row for row in fig5_rows if row["variant"] == "deplot_no_vs_opd_pcd")
    assert pcd_row["total_completion_count"] == "24"
    assert pcd_row["probe_candidate_count"] == "9"
    assert float(pcd_row["teacher_correct_given_probe_rate"]) > 0

    table4 = (out_dir / "table4_routing_antileakage.csv").read_text(encoding="utf-8")
    assert "teacher_probe_gold_suffix_rate" in table4
    assert "deplot_no_vs_opd_pcd" in table4
    table3_md = (out_dir / "table3_recoverability_controls.md").read_text(encoding="utf-8")
    table4_md = (out_dir / "table4_routing_antileakage.md").read_text(encoding="utf-8")
    assert "\\toprule" in table3_md
    assert "\\toprule" in table4_md
    assert "Acc ↑" in table4_md
    assert "\\rowcolor{gray!10} PCD" in table3_md
    assert "exact" in (out_dir / "data_quality_report.md").read_text(encoding="utf-8")

    fig1_csv = (out_dir / "fig1_motivation.csv").read_text(encoding="utf-8")
    assert "candidate_volume" in fig1_csv

    manifest_payload = json.loads((out_dir / "artifacts_manifest.json").read_text(encoding="utf-8"))
    artifact_ids = {row["id"] for row in manifest_payload["artifacts"]}
    assert {
        "fig0_training_basics",
        "fig1_motivation",
        "fig4_training_dynamics",
        "fig5_teacher_rescue_funnel",
        "fig6_va_vs_pcd_diagnosis",
        "table1_method_positioning",
        "table3_recoverability_controls",
        "table4_routing_antileakage",
        "paper_argument_report",
        "chart_review_report",
    }.issubset(artifact_ids)
    assert "fig2_pcd_opd_flow" not in artifact_ids
    report = (out_dir / "paper_argument_report.md").read_text(encoding="utf-8")
    assert "PCD expands recoverable wrong-completion supervision" in report
    assert "PCD is the main method" in report
    assert "Recommended paper claim" in report
    chart_review = (out_dir / "chart_review_report.md").read_text(encoding="utf-8")
    for figure_id in [
        "fig0_training_basics",
        "fig1_motivation",
        "fig4_training_dynamics",
        "fig5_teacher_rescue_funnel",
        "fig6_va_vs_pcd_diagnosis",
    ]:
        assert figure_id in chart_review
    assert "Use in paper" in chart_review
    assert "paper-style review" in chart_review
    assert "fig2_pcd_opd_flow" not in chart_review
    assert "anti-leakage audit" in chart_review
    dashboard = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "PCD-OPD Artifact Dashboard" in dashboard
    assert "artifacts_manifest.json" in dashboard
    assert "window.__PCD_ARTIFACTS_MANIFEST__" in dashboard
    assert "loadManifestWithFallback" in dashboard


def test_manifest_paths_prefer_current_working_directory_for_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_artifact_module()
    repo = tmp_path / "repo"
    manifest_dir = repo / "docs" / "figures" / "pcd_paper"
    manifest_dir.mkdir(parents=True)
    train_log = repo / "outputs" / "run" / "train.log"
    _write_train_log(train_log, pcd=True, va=False)
    manifest = manifest_dir / "run_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "variant,role,run_dir,train_log,eval_log,candidate_log_glob,config_path,enabled",
                "v,pcd,outputs/run,outputs/run/train.log,,,,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    rows = module.read_manifest(manifest)

    assert rows[0]["train_log"] == Path("outputs/run/train.log")


def test_artifact_registry_resolves_aliases_and_rejects_unknown_targets() -> None:
    module = _load_artifact_module()
    registry = module.build_registry()

    resolved = registry.resolve({"fig0", "table4"})

    assert [spec.id for spec in resolved] == ["fig0_training_basics", "table4_routing_antileakage"]
    try:
        registry.resolve({"not_a_real_artifact"})
    except ValueError as exc:
        assert "unknown artifact target" in str(exc)
    else:
        raise AssertionError("unknown target did not raise")


def test_fig5_uses_candidate_proxy_when_total_counts_are_missing(tmp_path: Path) -> None:
    module = _load_artifact_module()
    data = {
        "deplot_no_vs_opd_pcd": {
            "train": [],
            "candidates": [
                {"teacher_correct": True, "final_route": "opd"},
                {"teacher_correct": False, "final_route": "sft"},
            ],
        }
    }

    module.make_fig5(data, tmp_path)

    rows = list(csv.DictReader((tmp_path / "fig5_teacher_rescue_funnel.csv").open(encoding="utf-8")))
    assert rows[0]["funnel_scope"] == "candidate_proxy"
    assert rows[0]["total_completion_count"] == ""
    assert rows[0]["wrong_completion_count"] == ""
    assert rows[0]["probe_candidate_count"] == "2"


def test_make_quality_does_not_regenerate_all_artifacts(tmp_path: Path) -> None:
    module = _load_artifact_module()
    manifest = tmp_path / "run_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "variant,role,run_dir,train_log,eval_log,candidate_log_glob,config_path,enabled",
                "deplot_no_vs_opd,anchor,,,,,,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    table3 = out_dir / "table3_recoverability_controls.csv"
    table3.write_text("sentinel\n", encoding="utf-8")

    rc = module.main(["--manifest", str(manifest), "--out-dir", str(out_dir), "--make", "quality"])

    assert rc == 0
    assert table3.read_text(encoding="utf-8") == "sentinel\n"
    assert (out_dir / "data_quality_report.md").exists()
    assert (out_dir / "artifacts_manifest.json").exists()
    assert (out_dir / "index.html").exists()


def test_fig6_keeps_final_eval_when_training_log_is_missing(tmp_path: Path) -> None:
    module = _load_artifact_module()
    data = {
        "deplot_no_vs_opd_pcd": {
            "train": [],
            "eval": {"accuracy": 0.64, "full_cot_rate": 0.9, "other_rate": 0.02},
            "candidates": [],
        }
    }

    module.make_fig6(data, tmp_path)

    rows = list(csv.DictReader((tmp_path / "fig6_va_vs_pcd_diagnosis.csv").open(encoding="utf-8")))
    pcd_row = next(row for row in rows if row["variant"] == "deplot_no_vs_opd_pcd")
    assert pcd_row["final_accuracy"] == "0.6400"
    assert pcd_row["final_other_rate"] == "0.0200"
