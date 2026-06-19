"""
Shared fast-training defaults for test/config/* baseline wrappers.

Override via environment variables (see plan / test/README.md).
"""
from __future__ import annotations

import copy
import os
from typing import Any

from config.env_overrides import env_float, env_int, env_str
from data_utils.paths import OUTPUTS_DIR, project_path

MAX_TRAIN_SAMPLES = env_int("DYME_FAST_MAX_SAMPLES", 512)
MAX_STEPS = env_int("DYME_FAST_MAX_STEPS", 500)
SFT_EPOCHS = env_int("DYME_FAST_SFT_EPOCHS", 1)
COLD_START_FRAC = env_float("DYME_FAST_COLD_START_FRAC", 0.08)
OUTPUT_ROOT = env_str("DYME_FAST_OUTPUT_ROOT", os.path.join(OUTPUTS_DIR, "test-fast"))
FAST_TRAIN_JSON = env_str(
    "DYME_FAST_TRAIN_JSON",
    project_path(f"data/chartqa/train_fast_{MAX_TRAIN_SAMPLES}.json"),
)


def scaled_gate_defaults(max_steps: int = MAX_STEPS) -> dict[str, Any]:
    """Scale RLSD/OPD warmup gates to fit within a short max_steps budget."""
    return {
        "degen_skip_warmup_steps": max(20, max_steps // 12),
        "sft_warmup_steps": max(40, max_steps // 5),
        "sft_warmup_slots_per_group": 4,
        "sft_cold_start_frac": COLD_START_FRAC,
    }


def fast_dataset_config(base_dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_dataset,
        "train_dataset": FAST_TRAIN_JSON,
        "max_train_samples": MAX_TRAIN_SAMPLES,
    }


def fast_dyme_args(base_dyme_args: dict[str, Any], output_dir: str) -> dict[str, Any]:
    return {
        **base_dyme_args,
        "output_dir": output_dir,
        "max_steps": MAX_STEPS,
    }


def fast_sft_args(base_sft_args: dict[str, Any], output_dir: str) -> dict[str, Any]:
    return {
        **base_sft_args,
        "output_dir": output_dir,
        "num_train_epochs": SFT_EPOCHS,
    }


def apply_fast_opsd_gate(opsd_config: dict[str, Any]) -> dict[str, Any]:
    gate = {
        **opsd_config.get("gate", {}),
        **scaled_gate_defaults(),
    }
    return {**opsd_config, "gate": gate}


def apply_to_config(
    base_config: dict[str, Any],
    *,
    output_dir: str,
    opsd_enabled: bool | None = None,
    disable_deplot: bool = True,
) -> dict[str, Any]:
    """Return a deep-copied CONFIG with fast-training overrides applied."""
    cfg = copy.deepcopy(base_config)
    training = cfg.setdefault("training", {})
    training["dyme_args"] = fast_dyme_args(training.get("dyme_args", {}), output_dir)
    if "sft_args" in training:
        sft_out = os.path.join(OUTPUT_ROOT, "sft")
        training["sft_args"] = fast_sft_args(training["sft_args"], sft_out)
    cfg["dataset"] = fast_dataset_config(cfg.get("dataset", {}))
    if disable_deplot and "deplot" in cfg:
        cfg["deplot"] = {**cfg["deplot"], "enabled": False}
    if opsd_enabled is not None:
        cfg.setdefault("opsd", {})["enabled"] = opsd_enabled
    if cfg.get("opsd", {}).get("enabled"):
        cfg["opsd"] = apply_fast_opsd_gate(cfg["opsd"])
    return cfg
