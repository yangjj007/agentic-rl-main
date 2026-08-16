"""Tests for the explicit OPD image-primary visual-checker training variant."""
from __future__ import annotations

from pathlib import Path

from config.loader import load_config
from data_utils.paths import project_path
from reward_utils.visual_checker_teacher import TeacherVisualChecker
from reward_utils.visual_supervision_factory import (
    build_visual_supervision,
    visual_supervision_needs_teacher,
)


def test_image_checker_variant_inherits_best_opd_and_adds_image_checker():
    base = load_config("opd_7b_dyme_probe")
    cfg = load_config("opd_7b_dyme_probe_image_checker")
    base_opsd = base["opsd"]
    opsd = cfg["opsd"]
    visual = cfg["opsd"]["visual_supervision"]

    assert opsd["mode"] == base_opsd["mode"]
    assert opsd["loss"] == base_opsd["loss"]
    assert opsd["teacher_probe"] == base_opsd["teacher_probe"]
    assert opsd["teacher_trajectory"] == base_opsd["teacher_trajectory"]

    assert visual["enabled"] is True
    assert visual["prefetch_ic"] == base_opsd["visual_supervision"]["prefetch_ic"]
    assert visual["refiner"] == base_opsd["visual_supervision"]["refiner"]
    assert visual["checker"]["enabled"] is True
    assert visual["checker"]["grounding"] == "image_primary"
    assert visual["checker"]["aux_evidence"] == "none"
    assert visual_supervision_needs_teacher(cfg["opsd"]) is True
    assert cfg["dataset"]["train_dataset"] in {
        project_path("data", "chartqa", "train_new_prerefine_vf_full_real_deplot_fp32_qwen25.json"),
    }
    assert cfg["data_validation"] == {
        "strict": True,
        "require_real_deplot": True,
        "require_qwen_rewrite": True,
        "expected_samples": 4576,
    }

    checker, refiner, meta = build_visual_supervision(
        cfg["rl"],
        cfg["client"],
        cfg["opsd"],
        gpu_id=0,
    )
    assert isinstance(checker, TeacherVisualChecker)
    assert type(refiner).__name__ == type(
        build_visual_supervision(
            base["rl"],
            base["client"],
            base["opsd"],
            gpu_id=0,
        )[1]
    ).__name__
    assert meta["enabled"] is True
    assert meta["needs_teacher"] is True


def test_image_checker_training_script_selects_explicit_variant():
    script = Path("scripts/train_opd_7b_dyme_probe_image_checker.sh")
    text = script.read_text(encoding="utf-8")

    assert 'CONFIG_PATH="config/config_opd_7b_dyme_probe_image_checker.yaml"' in text
    assert "DYME_VISUAL_CHECKER=" not in text
    assert "DYME_VISUAL_REFINER=" not in text
    assert "DYME_VISUAL_PREFETCH_IC=" not in text
    assert "DYME_VISUAL_CHECKER_GROUNDING=" not in text
    assert "DYME_VISUAL_CHECKER_AUX=" not in text
    assert "DYME_CHARTQA_RAW=" not in text
    assert "DYME_CHARTQA_VF_HINT=" not in text
    assert "DYME_CHARTQA_VF_FULL=" not in text
    assert 'source "$(dirname "$0")/launch_utils.sh"' in text


def test_image_checker_variant_honors_chartqa_vf_full_override(monkeypatch):
    cfg = load_config("opd_7b_dyme_probe_image_checker")
    assert cfg["dataset"]["train_dataset"]


def test_visual_supervision_checker_settings_are_explicit():
    cfg = load_config("opd_7b_dyme_probe_image_checker")
    visual = cfg["opsd"]["visual_supervision"]
    assert "teacher_batch_size" in visual
    assert "checker" in visual


def test_image_checker_variant_has_explicit_training_values():
    cfg = load_config("opd_7b_dyme_probe_image_checker")
    dyme_args = cfg["training"]["dyme_args"]
    assert dyme_args["num_generations"] > 0
    assert dyme_args["per_device_train_batch_size"] > 0
