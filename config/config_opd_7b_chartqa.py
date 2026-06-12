"""
COPSD-style cross-model OPD on ChartQA (Method 2).

Frozen LLaVA-OneVision 7B teacher; student default 0.5B.
Uses rlsd routing from config_rlsd_chartqa (no offline SFT phase).
"""
import os

import config.config_rlsd_chartqa as rlsd
from data_utils.paths import OUTPUTS_DIR

MODEL_CONFIG = {
    **rlsd.MODEL_CONFIG,
    "teacher_model_path": os.environ.get(
        "DYME_TEACHER_MODEL",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    ),
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
        **rlsd.TRAINING_CONFIG,
        "dyme_args": {
            **rlsd.TRAINING_CONFIG["dyme_args"],
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
