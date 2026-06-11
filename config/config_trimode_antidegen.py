"""
TriMode ChartQA config with anti-degeneration overrides (log-validated).

Based on train_trimode_4gpu_20260610_173637.log analysis:
- decoding: shorter max length, lower temperature, higher repetition penalty
- training: lower LR + warmup to mitigate step-1 collapse
- OPSD gate: require_format_for_opsd=False to raise opsd_mask coverage
- reward_weights: emphasize continuous context F1 for group reward spread
"""
import os

import config.config_trimode as trimode
from data_utils.paths import OUTPUTS_DIR

MODEL_CONFIG = dict(trimode.MODEL_CONFIG)

TRAINING_CONFIG = {
    **trimode.TRAINING_CONFIG,
    "dyme_args": {
        **trimode.TRAINING_CONFIG["dyme_args"],
        "output_dir": os.environ.get(
            "DYME_OUTPUT_DIR",
            os.path.join(OUTPUTS_DIR, "dyme-trimode-antidegen"),
        ),
        "max_completion_length": 150,
        "temperature": 0.7,
        "repetition_penalty": 1.25,
        "learning_rate": 5e-5,
        "warmup_steps": 50,
    },
}

_reward_weights_raw = os.environ.get("DYME_REWARD_WEIGHTS", "0.5,1.5,1.0")
try:
    _reward_weights = [float(x.strip()) for x in _reward_weights_raw.split(",") if x.strip()]
    if len(_reward_weights) != 3:
        raise ValueError("expected 3 weights")
except ValueError:
    _reward_weights = [0.5, 1.5, 1.0]

_require_format_raw = os.environ.get("DYME_OPSD_REQUIRE_FORMAT", "0").strip().lower()
_require_format_for_opsd = _require_format_raw not in ("0", "false", "no", "off")

DYME_OPSD_CONFIG = {
    **trimode.DYME_OPSD_CONFIG,
    "gate": {
        **trimode.DYME_OPSD_CONFIG.get("gate", {}),
        "require_format_for_opsd": _require_format_for_opsd,
        "skip_degenerate_for_opsd": True,
    },
    "reward_weights": _reward_weights,
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": trimode.RL_CONFIG,
    "opsd": DYME_OPSD_CONFIG,
    "client": trimode.CLIENT_CONFIG,
    "dataset": trimode.DATASET_CONFIG,
}
