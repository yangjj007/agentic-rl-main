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
        "skip_degenerate_for_opsd": True,
        "online_sft_on_all_wrong": True,
        # ChartQA short numeric answers lack "Answer:" — do not block OPSD on format alone
        "opsd_degenerate_require_answer_flag": False,
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
    "temperature": float(os.environ.get("DYME_TEMPERATURE", "0.6")),
    "repetition_penalty": float(os.environ.get("DYME_REPETITION_PENALTY", "1.35")),
    "max_completion_length": int(os.environ.get("DYME_MAX_COMPLETION_LENGTH", "128")),
}
_max_steps_raw = os.environ.get("DYME_MAX_STEPS", "").strip()
if _max_steps_raw:
    _dyme_args["max_steps"] = int(_max_steps_raw)

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **TRAINING_CONFIG,
        "dyme_args": _dyme_args,
    },
    "rl": antidegen.CONFIG["rl"],
    "opsd": DYME_OPSD_CONFIG,
    "client": antidegen.CONFIG["client"],
    "dataset": antidegen.CONFIG["dataset"],
}
