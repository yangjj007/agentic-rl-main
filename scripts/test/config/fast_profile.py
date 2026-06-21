"""
Shared fast-training defaults for scripts/test/config/* baseline wrappers.

Uses the full ChartQA dataset with fewer epochs (not a sample subset).
Override via environment variables (see scripts/test/README.md).
"""
from __future__ import annotations

import copy
import os
from typing import Any

from config.env_overrides import env_float, env_int, env_str
from data_utils.paths import OUTPUTS_DIR

# Full dataset; reduce epochs instead of subsampling.
RL_EPOCHS = env_int("DYME_FAST_NUM_TRAIN_EPOCHS", 4)
SFT_EPOCHS = env_int("DYME_FAST_SFT_EPOCHS", 4)
COLD_START_FRAC = env_float("DYME_FAST_COLD_START_FRAC", 0.08)
OUTPUT_ROOT = env_str("DYME_FAST_OUTPUT_ROOT", os.path.join(OUTPUTS_DIR, "test-fast"))
# Gate warmup scaling when total step count is not known at import time.
EST_STEPS_PER_EPOCH = env_int("DYME_FAST_EST_STEPS_PER_EPOCH", 600)


def estimated_rl_steps() -> int:
    return max(1, RL_EPOCHS * EST_STEPS_PER_EPOCH)


def scaled_gate_defaults() -> dict[str, Any]:
    """Scale RLSD/OPD warmup gates for a short epoch-budget run."""
    total = estimated_rl_steps()
    return {
        "degen_skip_warmup_steps": max(20, total // 12),
        "sft_warmup_steps": max(40, total // 5),
        "sft_warmup_slots_per_group": 4,
        "sft_cold_start_frac": COLD_START_FRAC,
    }


def fast_dataset_config(base_dataset: dict[str, Any]) -> dict[str, Any]:
    """Keep full production dataset path; do not cap samples."""
    ds = dict(base_dataset)
    ds.pop("max_train_samples", None)
    return ds


def fast_dyme_args(base_dyme_args: dict[str, Any], output_dir: str) -> dict[str, Any]:
    args = {
        **base_dyme_args,
        "output_dir": output_dir,
        "num_train_epochs": RL_EPOCHS,
    }
    # Epoch-based budget; never inherit a smoke/shortrun max_steps cap.
    args.pop("max_steps", None)
    return args


def fast_sft_args(base_sft_args: dict[str, Any], output_dir: str) -> dict[str, Any]:
    return {
        **base_sft_args,
        "output_dir": output_dir,
        "num_train_epochs": SFT_EPOCHS,
    }


def merge_visual_supervision_opsd(opsd_config: dict[str, Any]) -> dict[str, Any]:
    """Attach shared 7B Visual Checker/Refiner settings (same as dyme_probe)."""
    from config.visual_supervision_defaults import build_visual_supervision_config

    return {
        **opsd_config,
        "visual_supervision": build_visual_supervision_config(),
    }


def apply_fast_opsd_gate(opsd_config: dict[str, Any]) -> dict[str, Any]:
    gate = {
        **opsd_config.get("gate", {}),
        **scaled_gate_defaults(),
    }
    gate.pop("sft_cold_start_steps", None)
    return {**opsd_config, "gate": gate}


def apply_dyme_aligned_opd(opsd_config: dict[str, Any]) -> dict[str, Any]:
    """Fast OPD: no embedded cold-start; all-wrong -> SFT; OPD only if teacher probe is correct."""
    opsd = {
        **opsd_config,
        "mode": "dyme_teacher_probe_opd",
        "privileged_providers": ["format_only"],
        "gate": {
            **opsd_config.get("gate", {}),
            "sft_cold_start_frac": 0.0,
            "online_sft_on_all_wrong": True,
        },
        "teacher_probe": {
            **opsd_config.get("teacher_probe", {}),
            "enabled": True,
        },
    }
    opsd.pop("visual_supervision", None)
    return opsd


def apply_to_config(
    base_config: dict[str, Any],
    *,
    output_dir: str,
    opsd_enabled: bool | None = None,
    disable_deplot: bool = True,
    enable_visual_supervision: bool = False,
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
        if enable_visual_supervision:
            cfg["opsd"] = merge_visual_supervision_opsd(cfg["opsd"])
    return cfg
