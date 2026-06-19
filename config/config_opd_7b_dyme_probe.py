"""
DyME-aligned OPD on ChartQA with teacher-correct gating.

Routing:
- all-wrong group -> SFT for every completion
- correct completion -> GRPO
- wrong completion -> 7B teacher answer probe
- teacher-correct wrong completion -> SRKL OPD + optional teacher-trajectory FKL
- teacher-wrong wrong completion -> SFT

Teacher context is no-gold by default: format instructions + visual facts.
"""
import os

import config.config_opd_7b_chartqa as base
from config.env_overrides import env_bool, env_float, env_int, env_list, env_optional_int, env_str
from config.config import DEPLOT_CONFIG as _BASE_DEPLOT_CONFIG
from config.visual_supervision_defaults import build_visual_supervision_config
from data_utils.paths import OUTPUTS_DIR

# --- Training defaults (single source of truth) ---
OPSD_MODE = "dyme_teacher_probe_opd"
PRIVILEGE_PROFILE = "hybrid"
PROBE_PROVIDERS = ["format_only", "visual_facts"]
OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "opd-7b-dyme-probe-chartqa")

TEACHER_PROBE_ENABLED = True
TEACHER_TRAJECTORY_ENABLED = True
LOSS_TYPE = "srkl"
SRKL_ALPHA = 0.1
OPSD_WEIGHT = 1.0
GRPO_WEIGHT = 1.0
TEACHER_TRAJ_FKL_WEIGHT = 0.5

# Full ChartQA run: follow num_train_epochs unless DYME_MAX_STEPS is explicitly set.
NUM_TRAIN_EPOCHS = env_int("DYME_NUM_TRAIN_EPOCHS", 10)

# Optional env overrides (ablation / server paths)
OPSD_MODE = env_str("DYME_OPSD_MODE", OPSD_MODE)
PRIVILEGE_PROFILE = env_str("DYME_OPSD_PRIVILEGE_PROFILE", PRIVILEGE_PROFILE)
PROBE_PROVIDERS = env_list("DYME_TEACHER_PROBE_PROVIDERS", env_list("DYME_OPSD_PROVIDERS", PROBE_PROVIDERS))
OUTPUT_DIR = env_str("DYME_OUTPUT_DIR", OUTPUT_DIR)

TEACHER_PROBE_ENABLED = env_bool("DYME_TEACHER_PROBE", TEACHER_PROBE_ENABLED)
TEACHER_TRAJECTORY_ENABLED = env_bool("DYME_TEACHER_TRAJECTORY", TEACHER_TRAJECTORY_ENABLED)
LOSS_TYPE = env_str("DYME_OPSD_LOSS_TYPE", LOSS_TYPE)
SRKL_ALPHA = env_float("DYME_OPSD_SRKL_ALPHA", SRKL_ALPHA)
OPSD_WEIGHT = env_float("DYME_OPSD_WEIGHT", OPSD_WEIGHT)
GRPO_WEIGHT = env_float("DYME_GRPO_WEIGHT", GRPO_WEIGHT)
TEACHER_TRAJ_FKL_WEIGHT = env_float("DYME_TEACHER_TRAJ_FKL_WEIGHT", TEACHER_TRAJ_FKL_WEIGHT)

MODEL_CONFIG = dict(base.MODEL_CONFIG)

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
}
# Do not inherit max_steps from a prior smoke shell export; full run uses epochs.
_dyme_args.pop("max_steps", None)
_max_steps = env_optional_int("DYME_MAX_STEPS")
if _max_steps is not None:
    _dyme_args["max_steps"] = _max_steps

DYME_OPSD_CONFIG = {
    **base.DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": OPSD_MODE,
    "text_include_gold": False,
    "privileged_profile": PRIVILEGE_PROFILE,
    "privileged_providers": PROBE_PROVIDERS,
    "gate": {
        **base.DYME_OPSD_CONFIG.get("gate", {}),
        "per_completion_opsd": True,
        "online_sft_on_all_wrong": True,
        "require_format_for_opsd": False,
    },
    "loss": {
        **base.DYME_OPSD_CONFIG.get("loss", {}),
        "loss_type": LOSS_TYPE,
        "srkl_alpha": SRKL_ALPHA,
        "opsd_weight": OPSD_WEIGHT,
        "grpo_weight": GRPO_WEIGHT,
        "teacher_traj_fkl_weight": TEACHER_TRAJ_FKL_WEIGHT,
    },
    "teacher_probe": {
        "enabled": TEACHER_PROBE_ENABLED,
        "context_providers": PROBE_PROVIDERS,
        "max_per_batch": env_int("DYME_TEACHER_PROBE_MAX_PER_BATCH", 0),
        "max_new_tokens": env_int("DYME_TEACHER_PROBE_MAX_NEW_TOKENS", 96),
        "do_sample": env_bool("DYME_TEACHER_PROBE_DO_SAMPLE", False),
        "temperature": env_float("DYME_TEACHER_PROBE_TEMPERATURE", 0.0),
        "top_p": env_float("DYME_TEACHER_PROBE_TOP_P", 1.0),
        "repetition_penalty": env_float("DYME_TEACHER_PROBE_REPETITION_PENALTY", 1.2),
        "max_relative_change": env_float("DYME_TEACHER_PROBE_RELAXED_TOL", 0.05),
    },
    "teacher_trajectory": {
        "enabled": TEACHER_TRAJECTORY_ENABLED,
        "loss_type": env_str("DYME_TEACHER_TRAJ_LOSS_TYPE", "fkl"),
        "weight": TEACHER_TRAJ_FKL_WEIGHT,
        "max_new_tokens": env_int("DYME_TEACHER_TRAJ_MAX_NEW_TOKENS", 128),
    },
    "visual_supervision": build_visual_supervision_config(),
    "debug": {
        **base.DYME_OPSD_CONFIG.get("debug", {}),
        "detail_every": env_int("DYME_OPSD_DETAIL_EVERY", 0),
    },
}

LAUNCH_CONFIG = {
    "gradient_checkpointing_enable": env_bool("DYME_GRADIENT_CHECKPOINTING", False),
    "opsd_detail_every": DYME_OPSD_CONFIG["debug"]["detail_every"],
    "opsd_detail_min_free_gb": env_float("DYME_OPSD_DETAIL_MIN_FREE_GB", 4.0),
    "teacher_device_map": env_str("DYME_TEACHER_DEVICE_MAP", "auto"),
    "pytorch_cuda_alloc_conf": env_str("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
}

_teacher_map = LAUNCH_CONFIG["teacher_device_map"]
if _teacher_map and _teacher_map.lower() not in ("none", "null"):
    MODEL_CONFIG["teacher_device_map"] = _teacher_map

DEPLOT_CONFIG = {
    **_BASE_DEPLOT_CONFIG,
    "enabled": env_bool("DYME_DEPLOT_ENABLED", False),
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": {
        **base.CONFIG["training"],
        "dyme_args": _dyme_args,
    },
    "rl": base.CONFIG["rl"],
    "opsd": DYME_OPSD_CONFIG,
    "client": base.CONFIG["client"],
    "dataset": base.CONFIG["dataset"],
    "launch": LAUNCH_CONFIG,
    "deplot": DEPLOT_CONFIG,
}
