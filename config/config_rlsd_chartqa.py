"""
RLSD / anti-leakage ChartQA config (Method 1).

- mode=rlsd: correct → GRPO, wrong → same-prompt OPSD, all-wrong group → online SFT
- No gold answer / visual_facts in teacher privileged context
- Hyperparameters based on config_trimode_antidegen
"""
import os

import config.config_trimode_antidegen as antidegen
from data_utils.paths import OUTPUTS_DIR

MODEL_CONFIG = dict(antidegen.MODEL_CONFIG)

TRAINING_CONFIG = dict(antidegen.TRAINING_CONFIG)

_reward_weights_raw = os.environ.get("DYME_REWARD_WEIGHTS", "0.5,1.5,1.0")
try:
    _reward_weights = [float(x.strip()) for x in _reward_weights_raw.split(",") if x.strip()]
    if len(_reward_weights) != 3:
        raise ValueError("expected 3 weights")
except ValueError:
    _reward_weights = [0.5, 1.5, 1.0]

_providers_raw = os.environ.get("DYME_OPSD_PROVIDERS", "format_only").strip()
_privileged_providers = [p.strip() for p in _providers_raw.split(",") if p.strip()] if _providers_raw else []

_skip_degen_env = os.environ.get("DYME_OPSD_SKIP_DEGENERATE", "").strip().lower()
if _skip_degen_env in ("0", "false", "no", "off"):
    _skip_degenerate_for_opsd = False
elif _skip_degen_env in ("1", "true", "yes", "on"):
    _skip_degenerate_for_opsd = True
else:
    _skip_degenerate_for_opsd = True

# Embedded SFT cold-start + RLSD warmup gates (env overrides optional).
_RLSD_GATE_DEFAULTS = {
    "skip_degenerate_for_opsd": _skip_degenerate_for_opsd,
    "degen_skip_warmup_steps": 200,
    "sft_warmup_steps": 500,
    "sft_warmup_slots_per_group": 4,
    # First N steps: skip generate, 100% GT injection, pure SFT NLL (no OPSD/GRPO).
    "sft_cold_start_frac": 0.08,
}

DYME_OPSD_CONFIG = {
    **antidegen.DYME_OPSD_CONFIG,
    "mode": os.environ.get("DYME_OPSD_MODE", "rlsd"),
    "text_include_gold": False,
    "privileged_profile": os.environ.get("DYME_OPSD_PRIVILEGE_PROFILE", "text"),
    "privileged_providers": _privileged_providers,
    "gate": {
        **antidegen.DYME_OPSD_CONFIG.get("gate", {}),
        "per_completion_opsd": True,
        "recoverable_without_privilege": True,
        "require_format_for_opsd": os.environ.get("DYME_OPSD_REQUIRE_FORMAT", "0").strip().lower()
        not in ("0", "false", "no", "off"),
        "online_sft_on_all_wrong": True,
        # ChartQA short numeric answers lack "Answer:" — do not block OPSD on format alone
        "opsd_degenerate_require_answer_flag": False,
        **_RLSD_GATE_DEFAULTS,
    },
    "loss": {
        **antidegen.DYME_OPSD_CONFIG.get("loss", {}),
        "acc_gate": True,
        "opsd_weight": float(os.environ.get("DYME_OPSD_WEIGHT", "1.5")),
        "grpo_weight": 1.0,
    },
    "reward_weights": _reward_weights,
}

_dyme_args = {
    **TRAINING_CONFIG["dyme_args"],
    "output_dir": os.environ.get(
        "DYME_OUTPUT_DIR",
        os.path.join(OUTPUTS_DIR, "rlsd-chartqa"),
    ),
    # Mitigate early RL collapse (newline + bare number + immediate EOS)
    "max_completion_length": 96,
    "temperature": 0.5,
    "repetition_penalty": 1.5,
}
_max_steps_raw = os.environ.get("DYME_MAX_STEPS", "").strip()
if _max_steps_raw:
    _dyme_args["max_steps"] = int(_max_steps_raw)

_temp_raw = os.environ.get("DYME_TEMPERATURE", "").strip()
if _temp_raw:
    _dyme_args["temperature"] = float(_temp_raw)
_rep_raw = os.environ.get("DYME_REPETITION_PENALTY", "").strip()
if _rep_raw:
    _dyme_args["repetition_penalty"] = float(_rep_raw)
_max_len_raw = os.environ.get("DYME_MAX_COMPLETION_LENGTH", "").strip()
if _max_len_raw:
    _dyme_args["max_completion_length"] = int(_max_len_raw)

# Keep module-level TRAINING_CONFIG in sync so imports of TRAINING_CONFIG["dyme_args"] match CONFIG.
TRAINING_CONFIG = {**TRAINING_CONFIG, "dyme_args": _dyme_args}

# Optional env overrides for gate defaults (see _RLSD_GATE_DEFAULTS above).
_degen_warmup_raw = os.environ.get("DYME_OPSD_DEGEN_WARMUP_STEPS", "").strip()
if _degen_warmup_raw:
    DYME_OPSD_CONFIG["gate"]["degen_skip_warmup_steps"] = int(_degen_warmup_raw)

_sft_warmup_raw = os.environ.get("DYME_SFT_WARMUP_STEPS", "").strip()
if _sft_warmup_raw:
    DYME_OPSD_CONFIG["gate"]["sft_warmup_steps"] = int(_sft_warmup_raw)

_sft_slots_raw = os.environ.get("DYME_SFT_WARMUP_SLOTS", "").strip()
if _sft_slots_raw:
    DYME_OPSD_CONFIG["gate"]["sft_warmup_slots_per_group"] = int(_sft_slots_raw)

_cold_start_steps_raw = os.environ.get("DYME_SFT_COLD_START_STEPS", "").strip()
if _cold_start_steps_raw:
    DYME_OPSD_CONFIG["gate"]["sft_cold_start_steps"] = int(_cold_start_steps_raw)
    DYME_OPSD_CONFIG["gate"].pop("sft_cold_start_frac", None)
else:
    _cold_start_frac_raw = os.environ.get("DYME_SFT_COLD_START_FRAC", "").strip()
    if _cold_start_frac_raw:
        DYME_OPSD_CONFIG["gate"]["sft_cold_start_frac"] = float(_cold_start_frac_raw)

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **TRAINING_CONFIG,
        "dyme_args": _dyme_args,
        "sft_args": {
            "output_dir": os.environ.get(
                "DYME_SFT_OUTPUT_DIR",
                os.path.join(OUTPUTS_DIR, "chartqa-sft"),
            ),
            "logging_steps": 10,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": int(os.environ.get("DYME_SFT_EPOCHS", "2")),
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
}
