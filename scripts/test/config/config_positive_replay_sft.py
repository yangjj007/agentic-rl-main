"""
Offline SFT warmup on the positive replay buffer exported from teacher-correct candidates.

Usage:
  DYME_REPLAY_TRAIN_DATASET=outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json \
  accelerate launch main_sft.py --config scripts/test/config/config_positive_replay_sft.py
"""
import copy
import os
import sys

_test_cfg_dir = os.path.dirname(os.path.abspath(__file__))
if _test_cfg_dir not in sys.path:
    sys.path.insert(0, _test_cfg_dir)

import config_rlsd_chartqa as base
from config.env_overrides import env_float, env_int, env_str
from data_utils.paths import discover_local_model
from fast_profile import OUTPUT_ROOT

DEFAULT_REPLAY_DATASET = os.path.join(
    "outputs",
    "test-fast",
    "positive-replay-buffer",
    "student_hint_short_full",
    "replay_train.json",
)

REPLAY_TRAIN_DATASET = env_str("DYME_REPLAY_TRAIN_DATASET", DEFAULT_REPLAY_DATASET)
REPLAY_OUTPUT_DIR = env_str(
    "DYME_SFT_OUTPUT_DIR",
    os.path.join(OUTPUT_ROOT, "positive-replay-sft", "warmup"),
)
DEFAULT_STUDENT_MODEL = "/home/deepseek_VG/deepseek/models/llava-0.5b-ov"

CONFIG = copy.deepcopy(base.CONFIG)
CONFIG["model"] = {
    **CONFIG["model"],
    "pretrained_model_path": discover_local_model("student", DEFAULT_STUDENT_MODEL),
}
CONFIG["dataset"] = {
    **CONFIG.get("dataset", {}),
    "train_dataset": REPLAY_TRAIN_DATASET,
    "eval_dataset": None,
}
CONFIG["checkpoint_eval"] = {
    **CONFIG.get("checkpoint_eval", {}),
    "enabled": False,
}
CONFIG["training"]["sft_args"] = {
    **CONFIG["training"]["sft_args"],
    "output_dir": REPLAY_OUTPUT_DIR,
    "num_train_epochs": env_float("DYME_REPLAY_SFT_EPOCHS", 0.5),
    "per_device_train_batch_size": env_int("DYME_REPLAY_SFT_BATCH_SIZE", 2),
    "gradient_accumulation_steps": env_int("DYME_REPLAY_SFT_GRAD_ACCUM", 4),
    "learning_rate": env_float("DYME_REPLAY_SFT_LR", 5e-6),
    "save_strategy": "epoch",
    "remove_unused_columns": False,
}
