from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chartqa_10epoch_matrix_dry_run_lists_main5_default_matrix(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
            "--dry-run",
            "--run-id",
            "pytest_chartqa10",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    variants = re.findall(r"^Variant: ([^\n]+)$", out, flags=re.MULTILINE)
    assert variants == [
        "dyme_full_original",
        "clrc_full",
        "answer_anchor_clrc",
        "confidence_weighted_clrc",
        "evidence_adaptive_clrc",
    ]

    assert "epochs: 10" in out
    assert "diagnostic epochs: 1" in out
    assert "preset: main5" in out
    assert "DYME_DYME_EPOCHS=10" in out
    assert "scripts/test/run_dyme_matched_4epoch.sh --variant full --epochs 10" in out
    assert "bash scripts/test/train_dyme.sh" not in out
    assert "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision" in out
    assert "deplot_no_vs_opd_pcd_gold_hidden_answer_anchor_clrc" in out
    assert "deplot_no_vs_opd_pcd_gold_hidden_confidence_weighted_clrc" in out
    assert "deplot_no_vs_opd_pcd_gold_hidden_evidence_adaptive_clrc" in out
    assert "deplot_no_vs_opd_pcd_gold_hidden_grpo_only" not in out
    assert "grpo_recovery_boost_clrc" not in out
    assert "vold_cold_start" not in out
    assert "ssopd_mixed_group" not in out
    assert "python" not in out.lower() or "-m accelerate.commands.launch" in out


def test_chartqa_10epoch_matrix_all_preset_lists_appendix_matrix(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
            "--dry-run",
            "--run-id",
            "pytest_chartqa10_all",
            "--preset",
            "all",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    variants = re.findall(r"^Variant: ([^\n]+)$", out, flags=re.MULTILINE)
    assert variants == [
        "dyme_pure_original",
        "dyme_full_original",
        "oracle_official_best_4e",
        "gold_hidden_no_opd",
        "gold_hidden_uncond_opd",
        "gold_hidden_routed_opd_fixed",
        "clrc_full",
        "clrc_target020",
        "grpo_only_matched",
        "opd_only_matched",
        "fallback_only_matched",
        "oracle_clean_no_full_hint",
        "token_reliability_clrc",
        "answer_anchor_clrc",
        "confidence_weighted_clrc",
        "grpo_recovery_boost_clrc",
        "evidence_adaptive_clrc",
        "mixed_group_shortest_correct_hard_replay",
    ]

    assert "preset: all" in out
    assert "bash scripts/test/run_pcd_no_visual.sh 4 --resume none --variant deplot_no_vs_opd_pcd_oracle_hint" in out
    assert "bash scripts/test/run_pcd_no_visual.sh 1 --resume none --variant deplot_no_vs_opd_pcd_gold_hidden_fallback_only" in out
    assert "bash scripts/test/run_pcd_no_visual.sh 1 --resume none --variant deplot_no_vs_opd_pcd_gold_hidden_mixed_group_shortest_correct_hard_replay" in out


def test_chartqa_10epoch_matrix_supports_shards_and_smoke(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
            "--dry-run",
            "--run-id",
            "pytest_smoke",
            "--smoke",
            "--shard-index",
            "1",
            "--shard-count",
            "4",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout
    variants = re.findall(r"^Variant: ([^\n]+)$", out, flags=re.MULTILINE)

    assert variants
    assert len(variants) < 19
    assert "smoke: 1" in out
    assert "DYME_PCD_MAX_STEPS=2" in out
    assert "DYME_SKIP_FINAL_SAVE=1" in out
    assert "DYME_SAVE_STRATEGY=no" in out
    assert "shard: 1/4" in out


def test_chartqa_10epoch_matrix_supports_variant_subset(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
            "--dry-run",
            "--run-id",
            "pytest_subset",
            "--variants",
            "oracle_official_best_4e,token_reliability_clrc",
            "--stages",
            "train",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    variants = re.findall(r"^Variant: ([^\n]+)$", result.stdout, flags=re.MULTILINE)

    assert variants == ["oracle_official_best_4e", "token_reliability_clrc"]


def test_chartqa_10epoch_matrix_supports_diagnostic_epoch_override(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_CHARTQA_ABLATION_DIAGNOSTIC_EPOCHS": "2",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
            "--dry-run",
            "--run-id",
            "pytest_diag_short",
            "--variants",
            "fallback_only_matched,mixed_group_shortest_correct_hard_replay",
            "--stages",
            "train",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "diagnostic epochs: 2" in out
    assert "bash scripts/test/run_pcd_no_visual.sh 2 --resume none --variant deplot_no_vs_opd_pcd_gold_hidden_fallback_only" in out
    assert "bash scripts/test/run_pcd_no_visual.sh 2 --resume none --variant deplot_no_vs_opd_pcd_gold_hidden_mixed_group_shortest_correct_hard_replay" in out


def test_chartqa_10epoch_matrix_rejects_retired_near_neighbor_labels(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_CHARTQA_ABLATION_OUTPUT_ROOT": str(tmp_path / "matrix"),
        "DYME_CHARTQA_ABLATION_LOG_ROOT": str(tmp_path / "logs"),
    }

    for retired_label in ("vold_cold_start", "ssopd_mixed_group"):
        result = subprocess.run(
            [
                "bash",
                "scripts/test/run_chartqa_10epoch_ablation_matrix.sh",
                "--dry-run",
                "--variants",
                retired_label,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 2
        assert "retired near-neighbor label" in result.stderr


def test_dyme_matched_runner_accepts_10epoch_override() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_dyme_matched_4epoch.sh",
            "--dry-run",
            "--variant",
            "pure",
            "--epochs",
            "10",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "Matched Pure DyME 10epoch run" in out
    assert "DYME_NUM_TRAIN_EPOCHS=10" in out
    assert "-m accelerate.commands.launch" in out
