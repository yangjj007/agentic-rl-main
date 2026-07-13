"""Pure DyME comparator matched to the current CLRC optimization budget."""

from __future__ import annotations

import copy

import config.config_rlsd_chartqa as base
from config.env_overrides import env_int, env_optional_int, env_str


OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", "outputs/test-fast/dyme-matched")
MODEL_CONFIG = dict(base.CONFIG["model"])
MODEL_CONFIG["pretrained_model_path"] = env_str(
    "DYME_STUDENT_MODEL",
    MODEL_CONFIG["pretrained_model_path"],
)

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": env_int("DYME_NUM_TRAIN_EPOCHS", 4),
    "save_strategy": env_str("DYME_SAVE_STRATEGY", "steps"),
}
_dyme_args.pop("max_steps", None)
_max_steps = env_optional_int("DYME_MAX_STEPS")
if _max_steps is not None:
    _dyme_args["max_steps"] = _max_steps
_save_steps = env_optional_int("DYME_SAVE_STEPS")
if _save_steps is not None:
    _dyme_args["save_steps"] = _save_steps
_save_total_limit = env_optional_int("DYME_SAVE_TOTAL_LIMIT")
if _save_total_limit is not None:
    _dyme_args["save_total_limit"] = _save_total_limit

_opsd = copy.deepcopy(base.CONFIG["opsd"])
_opsd["enabled"] = False
_opsd["mode"] = "dyme"
_opsd["visual_supervision"] = {"enabled": False}
_opsd["global_signal_logging"] = {"enabled": True}
_opsd.setdefault("gate", {})["online_sft_on_all_wrong"] = True
_opsd["gate"]["disable_online_sft_slots"] = False
_opsd["gate"]["sft_cold_start_frac"] = 0.0
_opsd["gate"].pop("sft_cold_start_steps", None)

CONFIG = {
    **base.CONFIG,
    "model": MODEL_CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "opsd": _opsd,
    "deplot": {
        **base.CONFIG.get("deplot", {}),
        "enabled": False,
    },
}
