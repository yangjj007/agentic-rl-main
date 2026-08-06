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
_DATASET_CANDIDATES = (
    project_path("data", "chartqa", "train_new_prerefine_vf_full_real_deplot_fp32.json"),
    project_path("data", "chartqa", "train_new_prerefine_vf_full_real.json"),
    project_path("data", "chartqa", "train_new_prerefine_vf_full.json"),
)
_DEFAULT_DATASET = next((p for p in _DATASET_CANDIDATES if os.path.isfile(p)), _DATASET_CANDIDATES[0])
TRAIN_DATASET = env_str("DYME_CHARTQA_VF_FULL", _DEFAULT_DATASET)
if not os.path.isabs(TRAIN_DATASET):
    TRAIN_DATASET = project_path(TRAIN_DATASET)

CONFIG = copy.deepcopy(base.CONFIG)
CONFIG["training"]["dyme_args"]["output_dir"] = OUTPUT_DIR
CONFIG["training"]["dyme_args"]["num_train_epochs"] = int(
    os.environ.get("DYME_NUM_TRAIN_EPOCHS", "16")
)
CONFIG["dataset"]["train_dataset"] = TRAIN_DATASET

# This experiment consumes precomputed visual facts.  The launcher and main
# both run the strict validator before any model is initialized.
CONFIG["data_validation"] = {
    "strict": True,
    "require_real_deplot": True,
    "require_qwen_rewrite": True,
    "expected_samples": 4576,
}

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

CONFIG["deplot"] = dict(CONFIG.get("deplot", {}), enabled=True)
