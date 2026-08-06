from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deplot_ablation_runner_accepts_epoch_override_for_10epoch_anchor() -> None:
    env = {
        **os.environ,
        "DYME_DEPLOT_ABLATION_EPOCHS": "10",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_opd_deplot_ablation.sh",
            "--dry-run",
            "--run-id",
            "pytest_10epoch",
            "--variants",
            "deplot_no_vs_opd",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DYME_NUM_TRAIN_EPOCHS=10" in result.stdout
    assert "DYME_FAST_NUM_TRAIN_EPOCHS=10" in result.stdout


def test_paper_required_10epoch_dry_run_lists_required_non_main_experiments() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_paper_required_10epoch.sh",
            "--dry-run",
            "--run-id",
            "pytest_required",
            "--main-run-id",
            "pytest_main",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    stages = re.findall(r"^Stage: ([^\n]+)$", out, flags=re.MULTILINE)
    assert stages == [
        "base_eval",
        "sft_train",
        "dyme_train",
        "no_pcd_anchor",
        "sanity",
        "eval_required",
    ]

    assert "budget epochs: 10" in out
    assert "DYME_FAST_SFT_EPOCHS=10" in out
    assert "DYME_FAST_NUM_TRAIN_EPOCHS=10" in out
    assert "DYME_FAST_OUTPUT_ROOT=outputs/test-fast/paper-required/pytest_required/baselines" in out
    assert "bash scripts/test/train_sft.sh" in out
    assert "bash scripts/test/train_dyme.sh" in out
    assert "DYME_DEPLOT_ABLATION_EPOCHS=10" in out
    assert "--variants deplot_no_vs_opd" in out
    assert "scripts/check_chartqa_teacher_evidence.py --input data/chartqa/train_medium_vf_full.json --json" in out
    assert "cat > outputs/test-fast/paper-required/pytest_required/sanity/pcd_manifest.csv" in out
    assert "scripts/analysis/pcd_probe_controls.py" in out
    assert "--manifest outputs/test-fast/paper-required/pytest_required/sanity/pcd_manifest.csv" in out
    assert "--variant deplot_no_vs_opd_pcd" in out
    assert "--out outputs/test-fast/paper-required/pytest_required/sanity/pcd_probe_controls.csv" in out
    assert "--model_path models/llava-0.5b-ov" in out
    assert "--model_path outputs/test-fast/paper-required/pytest_required/baselines/sft/final_checkpoint" in out
    assert "--model_path outputs/test-fast/paper-required/pytest_required/baselines/dyme/final_checkpoint" in out
    assert "--model_path outputs/test-fast/paper-required/pytest_required/opd-deplot-anchor/deplot_no_vs_opd/final_checkpoint" in out
    assert "--model_path outputs/test-fast/pcd-no-visual/pytest_main/deplot_no_vs_opd_pcd/final_checkpoint" in out
    assert "run_pcd_no_visual_10epoch.sh" not in out


def test_paper_required_10epoch_can_select_stages() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_paper_required_10epoch.sh",
            "--dry-run",
            "--run-id",
            "pytest_required",
            "--stages",
            "sft_train,no_pcd_anchor",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stages = re.findall(r"^Stage: ([^\n]+)$", result.stdout, flags=re.MULTILINE)
    assert stages == ["sft_train", "no_pcd_anchor"]
