"""Best OPD baseline plus explicit image-primary visual checker.

This variant inherits the current main DyME teacher-probe OPD configuration and
only fixes the checker-facing visual supervision knobs. Refiner, teacher probe,
teacher trajectory, prefetch, and OPD loss settings stay inherited from the base
variant unless their existing environment overrides are used.
"""
from __future__ import annotations

import copy
import os

import config.config_opd_7b_dyme_probe as base
from config.env_overrides import env_bool, env_str
from data_utils.paths import OUTPUTS_DIR, project_path

OUTPUT_DIR = env_str(
    "DYME_OUTPUT_DIR",
    os.path.join(OUTPUTS_DIR, "opd-7b-dyme-probe-image-checker"),
)

CONFIG = copy.deepcopy(base.CONFIG)
CONFIG["training"]["dyme_args"]["output_dir"] = OUTPUT_DIR
CONFIG["dataset"]["train_dataset"] = project_path("data", "chartqa", "train_new_prerefine_vf_full.json")

_visual = CONFIG["opsd"].setdefault("visual_supervision", {})
_checker = _visual.setdefault("checker", {})
_checker["enabled"] = env_bool("DYME_VISUAL_CHECKER", True)
_checker["model_source"] = env_str("DYME_VISUAL_CHECKER_MODEL_SOURCE", "loaded_teacher")
_checker["grounding"] = env_str("DYME_VISUAL_CHECKER_GROUNDING", "image_primary")
_checker["aux_evidence"] = env_str("DYME_VISUAL_CHECKER_AUX", "none")

_refiner = _visual.setdefault("refiner", {})
_refiner["enabled"] = env_bool("DYME_VISUAL_REFINER", True)
_refiner["model_source"] = env_str("DYME_VISUAL_REFINER_MODEL_SOURCE", "loaded_teacher")
_refiner["fallback"] = env_str("DYME_VISUAL_REFINER_FALLBACK", "passthrough")

_visual["enabled"] = bool(
    _checker.get("enabled")
    or _refiner.get("enabled", False)
)
