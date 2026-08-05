"""One-step single-GPU smoke test for save-time ChartQA evaluation."""
from __future__ import annotations

import os

import config.config as base
from config.env_overrides import env_int, env_str

_output_dir = env_str(
    "DYME_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "single-gpu-checkpoint-eval-smoke"),
)

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": _output_dir,
    "max_steps": env_int("DYME_MAX_STEPS", 1),
    "save_strategy": "steps",
    "save_steps": env_int("DYME_SAVE_STEPS", 1),
    "logging_steps": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "num_generations": 2,
    "generation_batch_size": 2,
    "max_completion_length": 8,
    "temperature": 0.0,
    "remove_unused_columns": False,
}

CONFIG = {
    **base.CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "dataset": {
        **base.CONFIG["dataset"],
        "train_dataset": "/tmp/dyme_checkpoint_eval_smoke/train.json",
    },
    "checkpoint_eval": {
        **base.CONFIG["checkpoint_eval"],
        "enabled": True,
        "split": "validation",
        "batch_size": 1,
        "max_new_tokens": 8,
        "patience": 3,
        "max_samples": 8,
    },
    "opsd": {
        **base.CONFIG["opsd"],
        "gate": {
            **base.CONFIG["opsd"].get("gate", {}),
            "sft_cold_start_steps": 1,
        },
    },
}
