"""
200-step smoke run for OPD 7B + RLSD anti-collapse validation.
"""
import os

import config.config_opd_7b_chartqa as base
from config.env_overrides import env_int, env_str
from data_utils.paths import OUTPUTS_DIR

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "opd-7b-chartqa-smoke")
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", OUTPUT_DIR)

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
    "max_steps": env_int("DYME_MAX_STEPS", 200),
    "max_completion_length": env_int("DYME_MAX_COMPLETION_LENGTH", 96),
    "temperature": 0.5,
    "repetition_penalty": 1.5,
}

DYME_OPSD_CONFIG = {
    **base.DYME_OPSD_CONFIG,
    "gate": {
        **base.DYME_OPSD_CONFIG.get("gate", {}),
        "degen_skip_warmup_steps": env_int("DYME_OPSD_DEGEN_WARMUP_STEPS", 200),
        "sft_warmup_steps": env_int("DYME_SFT_WARMUP_STEPS", 500),
        "sft_warmup_slots_per_group": env_int("DYME_SFT_WARMUP_SLOTS", 4),
        "sft_cold_start_frac": 0.08,
    },
}

CONFIG = {
    **base.CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "opsd": DYME_OPSD_CONFIG,
    # Smoke runs intentionally do not download/evaluate an HF validation set.
    "checkpoint_eval": {
        **base.CONFIG.get("checkpoint_eval", {}),
        "enabled": False,
    },
}
