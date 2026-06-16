"""
COPSD-style cross-model OPD on ChartQA (Method 2).

Frozen LLaVA-OneVision 7B teacher; student default 0.5B.
Inherits RLSD routing + embedded SFT cold-start gates from config_rlsd_chartqa.
"""
import os

import config.config_rlsd_chartqa as rlsd
from data_utils.paths import OUTPUTS_DIR, discover_local_model

_student_default = rlsd.MODEL_CONFIG["pretrained_model_path"]
_teacher_default = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

MODEL_CONFIG = {
    **rlsd.MODEL_CONFIG,
    "pretrained_model_path": discover_local_model("student", _student_default),
    "teacher_model_path": discover_local_model("teacher", _teacher_default),
    "teacher_dtype": os.environ.get("DYME_TEACHER_DTYPE", "bfloat16"),
    "teacher_device_map": os.environ.get("DYME_TEACHER_DEVICE_MAP") or None,
}

DYME_OPSD_CONFIG = {
    **rlsd.DYME_OPSD_CONFIG,
    "mode": os.environ.get("DYME_OPSD_MODE", "rlsd"),
    "privileged_providers": [],
    "loss": {
        **rlsd.DYME_OPSD_CONFIG.get("loss", {}),
        "opsd_weight": float(os.environ.get("DYME_OPSD_WEIGHT", "1.5")),
    },
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **rlsd.CONFIG["training"],
        "dyme_args": {
            **rlsd.CONFIG["training"]["dyme_args"],
            "output_dir": os.environ.get(
                "DYME_OUTPUT_DIR",
                os.path.join(OUTPUTS_DIR, "opd-7b-chartqa"),
            ),
        },
    },
    "rl": rlsd.CONFIG["rl"],
    "opsd": DYME_OPSD_CONFIG,
    "client": rlsd.CONFIG["client"],
    "dataset": rlsd.CONFIG["dataset"],
}
