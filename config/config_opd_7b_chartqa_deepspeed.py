"""
Cross-model OPD (7B teacher + 0.5B student) with DeepSpeed launch defaults.

Inherits training hyperparameters from config_opd_7b_chartqa; only output path,
teacher placement, and launch/runtime knobs differ from the DDP script.
"""
import os

import config.config_opd_7b_chartqa as base
from config.config import DEPLOT_CONFIG as _BASE_DEPLOT_CONFIG
from config.env_overrides import env_bool, env_float, env_int, env_str
from data_utils.paths import OUTPUTS_DIR

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "opd-7b-chartqa-ds")
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", OUTPUT_DIR)

MODEL_CONFIG = dict(base.MODEL_CONFIG)
_teacher_map = env_str("DYME_TEACHER_DEVICE_MAP", "auto")
if _teacher_map and _teacher_map.lower() not in ("none", "null"):
    MODEL_CONFIG["teacher_device_map"] = _teacher_map

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
}

DYME_OPSD_CONFIG = {
    **base.DYME_OPSD_CONFIG,
    "debug": {
        **base.DYME_OPSD_CONFIG.get("debug", {}),
        "detail_every": env_int("DYME_OPSD_DETAIL_EVERY", 0),
    },
}

LAUNCH_CONFIG = {
    "gradient_checkpointing_enable": env_bool("DYME_GRADIENT_CHECKPOINTING", False),
    "opsd_detail_every": DYME_OPSD_CONFIG["debug"]["detail_every"],
    "opsd_detail_min_free_gb": env_float("DYME_OPSD_DETAIL_MIN_FREE_GB", 4.0),
    "teacher_device_map": _teacher_map,
    "pytorch_cuda_alloc_conf": env_str("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
}

DEPLOT_CONFIG = {
    **_BASE_DEPLOT_CONFIG,
    "enabled": env_bool("DYME_DEPLOT_ENABLED", False),
}

CONFIG = {
    **base.CONFIG,
    "model": MODEL_CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "opsd": DYME_OPSD_CONFIG,
    "checkpoint_eval": dict(base.CONFIG.get("checkpoint_eval", {})),
    "launch": LAUNCH_CONFIG,
    "deplot": DEPLOT_CONFIG,
}
