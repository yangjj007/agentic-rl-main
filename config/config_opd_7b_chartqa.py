"""
COPSD-style cross-model OPD on ChartQA (Method 2).

Frozen LLaVA-OneVision 7B teacher; student default 0.5B.
Inherits RLSD routing + embedded SFT cold-start gates from config_rlsd_chartqa.
"""
import os

import config.config_rlsd_chartqa as rlsd
from config.config import DEPLOT_CONFIG as _BASE_DEPLOT_CONFIG
from config.env_overrides import env_bool, env_float, env_int, env_str
from data_utils.paths import OUTPUTS_DIR, discover_local_model

_student_default = rlsd.MODEL_CONFIG["pretrained_model_path"]
_teacher_default = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
_teacher_override = os.environ.get("DYME_TEACHER_MODEL")
_teacher_path = (
    _teacher_override.strip()
    if _teacher_override is not None
    else discover_local_model("teacher", _teacher_default)
)

OPSD_MODE = env_str("DYME_OPSD_MODE", "rlsd")
PRIVILEGE_PROFILE = env_str("DYME_OPSD_PRIVILEGE_PROFILE", "text")
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", os.path.join(OUTPUTS_DIR, "opd-7b-chartqa"))
OPSD_WEIGHT = env_float("DYME_OPSD_WEIGHT", 1.5)

MODEL_CONFIG = {
    **rlsd.MODEL_CONFIG,
    "pretrained_model_path": discover_local_model("student", _student_default),
    "teacher_model_path": _teacher_path,
    "teacher_dtype": env_str("DYME_TEACHER_DTYPE", "bfloat16"),
    "teacher_device_map": env_str("DYME_TEACHER_DEVICE_MAP", "") or None,
}

DYME_OPSD_CONFIG = {
    **rlsd.DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": OPSD_MODE,
    "privileged_profile": PRIVILEGE_PROFILE,
    "privileged_providers": [],
    "loss": {
        **rlsd.DYME_OPSD_CONFIG.get("loss", {}),
        "opsd_weight": OPSD_WEIGHT,
    },
    "debug": {
        **rlsd.DYME_OPSD_CONFIG.get("debug", {}),
        "detail_every": env_int("DYME_OPSD_DETAIL_EVERY", 10),
    },
}

LAUNCH_CONFIG = {
    "gradient_checkpointing_enable": env_bool("DYME_GRADIENT_CHECKPOINTING", False),
    "opsd_detail_every": DYME_OPSD_CONFIG["debug"]["detail_every"],
    "opsd_detail_min_free_gb": env_float("DYME_OPSD_DETAIL_MIN_FREE_GB", 4.0),
    "teacher_device_map": env_str("DYME_TEACHER_DEVICE_MAP", "auto"),
}

DEPLOT_CONFIG = {
    **_BASE_DEPLOT_CONFIG,
    "enabled": env_bool("DYME_DEPLOT_ENABLED", False),
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **rlsd.CONFIG["training"],
        "dyme_args": {
            **rlsd.CONFIG["training"]["dyme_args"],
            "output_dir": OUTPUT_DIR,
        },
    },
    "rl": rlsd.CONFIG["rl"],
    "opsd": DYME_OPSD_CONFIG,
    "client": rlsd.CONFIG["client"],
    "dataset": rlsd.CONFIG["dataset"],
    "launch": LAUNCH_CONFIG,
    "deplot": DEPLOT_CONFIG,
}
