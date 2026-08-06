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

# Prefer the separately named real-DePlot artifact when it exists.  Some
# servers generated the same real rows under the historical ``*_vf_full.json``
# name, so retain that path as a compatibility candidate.  The selected path
# is still subjected to the same strict validator; a stale/placeholder file is
# never accepted silently.
_REAL_DATASET = project_path(
    "data", "chartqa", "train_new_prerefine_vf_full_real.json"
)
_LEGACY_DATASET = project_path(
    "data", "chartqa", "train_new_prerefine_vf_full.json"
)
CONFIG["dataset"]["train_dataset"] = (
    _REAL_DATASET if os.path.isfile(_REAL_DATASET) else _LEGACY_DATASET
)

# This is a precomputed-data recipe.  The launcher must inspect the exact file
# above before initializing a model; it must never silently create a
# placeholder DePlot dataset for this experiment.
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

# Keep the resolved configuration honest: this experiment consumes real,
# precomputed DePlot rows (the flag is not used to regenerate data at launch).
CONFIG["deplot"] = dict(CONFIG.get("deplot", {}), enabled=True)
