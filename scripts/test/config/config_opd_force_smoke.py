"""
Smoke OPD: no cold-start, no SFT routing — every completion uses OPSD/OPD from step 0.

Gate flags (see trainer/DyMETrainer.py):
  disable_force_sft_replace  — keep model completions even without Answer:
  disable_online_sft_slots   — no online-SFT slot on all-wrong groups

Usage:
  bash scripts/test/train_opd_force_smoke.sh
"""
from __future__ import annotations

import os
import sys

_test_cfg_dir = os.path.dirname(os.path.abspath(__file__))
if _test_cfg_dir not in sys.path:
    sys.path.insert(0, _test_cfg_dir)

import config.config_opd_7b_chartqa_deepspeed as base
from config.env_overrides import env_int, env_str
from config.visual_supervision_defaults import build_visual_supervision_config
from data_utils.paths import OUTPUTS_DIR

OUTPUT_DIR = env_str(
    "DYME_OUTPUT_DIR",
    os.path.join(OUTPUTS_DIR, "test-fast", "opd-force-smoke"),
)

_FORCE_GATE = {
    # No embedded SFT cold-start
    "sft_cold_start_frac": 0.0,
    "sft_cold_start_steps": 0,
    # No RLSD warmup SFT slots
    "sft_warmup_steps": 0,
    "sft_warmup_slots_per_group": 0,
    "degen_skip_warmup_steps": 0,
    # Always run OPSD on degenerate / malformed completions
    "skip_degenerate_for_opsd": False,
    "opsd_degenerate_require_answer_flag": False,
    "require_format_for_opsd": False,
    "online_sft_on_all_wrong": False,
    # Test-only: bypass SFT replacement paths in DyMETrainer routing
    "disable_force_sft_replace": True,
    "disable_online_sft_slots": True,
    "per_completion_opsd": True,
}

_opsd_mode = env_str("DYME_OPSD_MODE", "opsd_only")

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
    "max_steps": env_int("DYME_MAX_STEPS", 40),
    "num_train_epochs": env_int("DYME_NUM_TRAIN_EPOCHS", 1),
    "max_completion_length": env_int("DYME_MAX_COMPLETION_LENGTH", 96),
    "per_device_train_batch_size": env_int("DYME_PER_DEVICE_TRAIN_BATCH_SIZE", 1),
}

_dataset = {
    **base.CONFIG["dataset"],
    "max_train_samples": env_int("DYME_MAX_TRAIN_SAMPLES", 64),
    "eval_dataset": None,
}

DYME_OPSD_CONFIG = {
    **base.DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": _opsd_mode,
    "visual_supervision": build_visual_supervision_config(),
    "gate": {
        **base.DYME_OPSD_CONFIG.get("gate", {}),
        **_FORCE_GATE,
    },
    "debug": {
        **base.DYME_OPSD_CONFIG.get("debug", {}),
        "detail_every": env_int("DYME_OPSD_DETAIL_EVERY", 1),
    },
}

CONFIG = {
    **base.CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "dataset": _dataset,
    "opsd": DYME_OPSD_CONFIG,
    "checkpoint_eval": {
        **base.CONFIG.get("checkpoint_eval", {}),
        "enabled": False,
    },
}
