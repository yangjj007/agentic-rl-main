"""
RLSD / anti-leakage ChartQA config (Method 1).

- mode=rlsd: correct → GRPO, wrong → same-prompt OPSD, all-wrong group → online SFT
- No gold answer / visual_facts in teacher privileged context
- Hyperparameters based on config_trimode_antidegen
"""
import os

import config.config_trimode_antidegen as antidegen
from config.config import DEPLOT_CONFIG as _BASE_DEPLOT_CONFIG
from config.env_overrides import (
    env_bool,
    env_float,
    env_int,
    env_list,
    env_optional_float,
    env_optional_int,
    env_str,
)
from data_utils.paths import OUTPUTS_DIR

MODEL_CONFIG = dict(antidegen.MODEL_CONFIG)

TRAINING_CONFIG = dict(antidegen.TRAINING_CONFIG)

OPSD_MODE = env_str("DYME_OPSD_MODE", "rlsd")
PRIVILEGE_PROFILE = env_str("DYME_OPSD_PRIVILEGE_PROFILE", "text")
PRIVILEGED_PROVIDERS = env_list("DYME_OPSD_PROVIDERS", ["format_only"])
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", os.path.join(OUTPUTS_DIR, "rlsd-chartqa"))
REQUIRE_FORMAT_FOR_OPSD = env_bool("DYME_OPSD_REQUIRE_FORMAT", False)
SKIP_DEGENERATE_FOR_OPSD = env_bool("DYME_OPSD_SKIP_DEGENERATE", True)

_reward_weights_raw = os.environ.get("DYME_REWARD_WEIGHTS", "0.5,1.5,1.0")
try:
    _reward_weights = [float(x.strip()) for x in _reward_weights_raw.split(",") if x.strip()]
    if len(_reward_weights) != 3:
        raise ValueError("expected 3 weights")
except ValueError:
    _reward_weights = [0.5, 1.5, 1.0]

# Embedded SFT cold-start + RLSD warmup gates (env overrides optional).
_RLSD_GATE_DEFAULTS = {
    "skip_degenerate_for_opsd": SKIP_DEGENERATE_FOR_OPSD,
    "degen_skip_warmup_steps": 200,
    "sft_warmup_steps": 500,
    "sft_warmup_slots_per_group": 4,
    # First N steps: skip generate, 100% GT injection, pure SFT NLL (no OPSD/GRPO).
    "sft_cold_start_frac": 0.08,
}

DYME_OPSD_CONFIG = {
    **antidegen.DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": OPSD_MODE,
    "text_include_gold": False,
    "privileged_profile": PRIVILEGE_PROFILE,
    "privileged_providers": PRIVILEGED_PROVIDERS,
    "gate": {
        **antidegen.DYME_OPSD_CONFIG.get("gate", {}),
        "per_completion_opsd": True,
        "recoverable_without_privilege": True,
        "require_format_for_opsd": REQUIRE_FORMAT_FOR_OPSD,
        "online_sft_on_all_wrong": True,
        # ChartQA short numeric answers lack "Answer:" — do not block OPSD on format alone
        "opsd_degenerate_require_answer_flag": False,
        **_RLSD_GATE_DEFAULTS,
    },
    "loss": {
        **antidegen.DYME_OPSD_CONFIG.get("loss", {}),
        "acc_gate": True,
        "opsd_weight": env_float("DYME_OPSD_WEIGHT", 1.5),
        "grpo_weight": 1.0,
    },
    "reward_weights": _reward_weights,
    "debug": {
        **antidegen.DYME_OPSD_CONFIG.get("debug", {}),
        "verbose": env_bool("DYME_OPSD_DEBUG", False),
        "detail_every": env_int("DYME_OPSD_DETAIL_EVERY", 10),
    },
}

_dyme_args = {
    **TRAINING_CONFIG["dyme_args"],
    "output_dir": OUTPUT_DIR,
    # Mitigate early RL collapse (newline + bare number + immediate EOS)
    "max_completion_length": 96,
    "temperature": 0.5,
    "repetition_penalty": 1.5,
}
_max_steps = env_optional_int("DYME_MAX_STEPS")
if _max_steps is not None:
    _dyme_args["max_steps"] = _max_steps

_temp = env_optional_float("DYME_TEMPERATURE")
if _temp is not None:
    _dyme_args["temperature"] = _temp
_rep = env_optional_float("DYME_REPETITION_PENALTY")
if _rep is not None:
    _dyme_args["repetition_penalty"] = _rep
_max_len = env_optional_int("DYME_MAX_COMPLETION_LENGTH")
if _max_len is not None:
    _dyme_args["max_completion_length"] = _max_len

# Keep module-level TRAINING_CONFIG in sync so imports of TRAINING_CONFIG["dyme_args"] match CONFIG.
TRAINING_CONFIG = {**TRAINING_CONFIG, "dyme_args": _dyme_args}

# Optional env overrides for gate defaults (see _RLSD_GATE_DEFAULTS above).
_degen_warmup = env_optional_int("DYME_OPSD_DEGEN_WARMUP_STEPS")
if _degen_warmup is not None:
    DYME_OPSD_CONFIG["gate"]["degen_skip_warmup_steps"] = _degen_warmup

_sft_warmup = env_optional_int("DYME_SFT_WARMUP_STEPS")
if _sft_warmup is not None:
    DYME_OPSD_CONFIG["gate"]["sft_warmup_steps"] = _sft_warmup

_sft_slots = env_optional_int("DYME_SFT_WARMUP_SLOTS")
if _sft_slots is not None:
    DYME_OPSD_CONFIG["gate"]["sft_warmup_slots_per_group"] = _sft_slots

_cold_start_steps = env_optional_int("DYME_SFT_COLD_START_STEPS")
if _cold_start_steps is not None:
    DYME_OPSD_CONFIG["gate"]["sft_cold_start_steps"] = _cold_start_steps
    DYME_OPSD_CONFIG["gate"].pop("sft_cold_start_frac", None)
else:
    _cold_start_frac = env_optional_float("DYME_SFT_COLD_START_FRAC")
    if _cold_start_frac is not None:
        DYME_OPSD_CONFIG["gate"]["sft_cold_start_frac"] = _cold_start_frac

LAUNCH_CONFIG = {
    "gradient_checkpointing_enable": env_bool("DYME_GRADIENT_CHECKPOINTING", False),
    "opsd_detail_every": DYME_OPSD_CONFIG["debug"]["detail_every"],
    "opsd_detail_min_free_gb": env_float("DYME_OPSD_DETAIL_MIN_FREE_GB", 4.0),
}

DEPLOT_CONFIG = {
    **_BASE_DEPLOT_CONFIG,
    "enabled": env_bool("DYME_DEPLOT_ENABLED", False),
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **TRAINING_CONFIG,
        "dyme_args": _dyme_args,
        "sft_args": {
            "output_dir": env_str("DYME_SFT_OUTPUT_DIR", os.path.join(OUTPUTS_DIR, "chartqa-sft")),
            "logging_steps": 10,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": env_int("DYME_SFT_EPOCHS", 2),
            "learning_rate": 1e-5,
            "bf16": True,
            "gradient_checkpointing": True,
            "ddp_find_unused_parameters": False,
            "max_grad_norm": 1.0,
            "save_strategy": "epoch",
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "seed": 42,
            "remove_unused_columns": False,
        },
    },
    "rl": antidegen.CONFIG["rl"],
    "opsd": DYME_OPSD_CONFIG,
    "client": antidegen.CONFIG["client"],
    "dataset": antidegen.CONFIG["dataset"],
    "launch": LAUNCH_CONFIG,
    "deplot": DEPLOT_CONFIG,
}
