"""
Short smoke run for dyme_teacher_probe_opd (200 steps, no HF eval download).
"""
import os

import config.config_opd_7b_dyme_probe as base
from config.env_overrides import env_int, env_str
from data_utils.paths import OUTPUTS_DIR

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "opd-7b-dyme-probe-smoke")
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", OUTPUT_DIR)

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
    "max_steps": env_int("DYME_MAX_STEPS", 200),
    "max_completion_length": env_int("DYME_MAX_COMPLETION_LENGTH", 96),
}

_dataset = {
    **base.CONFIG["dataset"],
    "eval_dataset": None,
}

CONFIG = {
    **base.CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "dataset": _dataset,
}
