"""
Fast DyME baseline (pure GRPO) with DeepSpeed ZeRO-2 + memory-friendly RL defaults.

Tuned for 8× GPU GRPO: shorter completions, fewer generations per prompt, smaller
per-device batch — avoids GEN_CLIP_COLLAPSE → full-length SFT GT replace → logps OOM.

Usage:
  bash scripts/test/train_dyme.sh
  accelerate launch main.py --config scripts/test/config/config_dyme_deepspeed.py --mode rl
"""
import os
import sys

_test_cfg_dir = os.path.dirname(os.path.abspath(__file__))
if _test_cfg_dir not in sys.path:
    sys.path.insert(0, _test_cfg_dir)

import config.config as base
from config.env_overrides import env_bool, env_float, env_int, env_optional_int
from data_utils.paths import discover_local_model
from fast_profile import OUTPUT_ROOT, apply_to_config

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "dyme")

_max_len = env_optional_int("DYME_MAX_COMPLETION_LENGTH")
if _max_len is None:
    _max_len = 96

_dyme_overrides = {
    # Antidegen-style decoding (see config_trimode_antidegen / config_rlsd_chartqa).
    "max_completion_length": _max_len,
    "temperature": env_float("DYME_TEMPERATURE", 0.5),
    "repetition_penalty": env_float("DYME_REPETITION_PENALTY", 1.5),
    "per_device_train_batch_size": env_int("DYME_PER_DEVICE_BATCH", 1),
    "gradient_accumulation_steps": env_int("DYME_GRAD_ACCUM", 16),
    "num_generations": env_int("DYME_NUM_GENERATIONS", 4),
}

CONFIG = apply_to_config(
    base.CONFIG,
    output_dir=OUTPUT_DIR,
    opsd_enabled=False,
)
CONFIG["model"] = {
    **CONFIG["model"],
    "pretrained_model_path": discover_local_model(
        "student",
        base.MODEL_CONFIG["pretrained_model_path"],
    ),
}
CONFIG["training"]["dyme_args"] = {
    **CONFIG["training"]["dyme_args"],
    **_dyme_overrides,
}

# Pure DyME: no embedded SFT cold-start (OPD-only feature).
CONFIG["opsd"]["gate"]["sft_cold_start_frac"] = 0.0
CONFIG["opsd"]["gate"].pop("sft_cold_start_steps", None)

# Less verbose than production when debugging OOM.
CONFIG["opsd"]["debug"]["detail_every"] = env_int("DYME_OPSD_DETAIL_EVERY", 0)

CONFIG["launch"] = {
    "gradient_checkpointing_enable": env_bool("DYME_GRADIENT_CHECKPOINTING", False),
    "pytorch_cuda_alloc_conf": os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    ),
}
