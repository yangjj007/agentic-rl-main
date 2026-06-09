import os

from config import CLIENT_CONFIG, DATASET_CONFIG, DYME_OPSD_CONFIG, MODEL_CONFIG, RL_CONFIG, TRAINING_CONFIG
from data_utils.paths import OUTPUTS_DIR

MODEL_CONFIG = dict(MODEL_CONFIG)

TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    "dyme_args": {
        **TRAINING_CONFIG["dyme_args"],
        "output_dir": os.environ.get("DYME_OUTPUT_DIR", os.path.join(OUTPUTS_DIR, "dyme-trimode")),
    },
}

DYME_OPSD_CONFIG = {
    **DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": os.environ.get("DYME_OPSD_MODE", "trimode"),
    "privileged_providers": os.environ.get("DYME_OPSD_PROVIDERS", "text,visual_facts").split(","),
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "opsd": DYME_OPSD_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
}
