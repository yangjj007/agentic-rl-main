"""Tests for the explicit OPD image-primary visual-checker training variant."""
from __future__ import annotations

from pathlib import Path

from config.loader import load_config
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

    assert 'DYME_CONFIG="opd_7b_dyme_probe_image_checker"' in text
    assert 'DYME_VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-1}"' in text
    assert 'DYME_VISUAL_REFINER=' not in text
    assert 'DYME_VISUAL_PREFETCH_IC=' not in text
    assert 'DYME_VISUAL_CHECKER_GROUNDING="${DYME_VISUAL_CHECKER_GROUNDING:-image_primary}"' in text
    assert 'DYME_VISUAL_CHECKER_AUX="${DYME_VISUAL_CHECKER_AUX:-none}"' in text
