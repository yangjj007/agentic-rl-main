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
from data_utils.paths import OUTPUTS_DIR


MODEL_CONFIG = dict(base.MODEL_CONFIG)

_providers_raw = os.environ.get("DYME_TEACHER_PROBE_PROVIDERS", "format_only,visual_facts").strip()
_probe_providers = [p.strip() for p in _providers_raw.split(",") if p.strip()]

_dyme_args = {
    **base.CONFIG["training"]["dyme_args"],
    "output_dir": os.environ.get(
        "DYME_OUTPUT_DIR",
        os.path.join(OUTPUTS_DIR, "opd-7b-dyme-probe-chartqa"),
    ),
}

DYME_OPSD_CONFIG = {
    **base.DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": os.environ.get("DYME_OPSD_MODE", "dyme_teacher_probe_opd"),
    "text_include_gold": False,
    "privileged_profile": os.environ.get("DYME_OPSD_PRIVILEGE_PROFILE", "hybrid"),
    "privileged_providers": _probe_providers,
    "gate": {
        **base.DYME_OPSD_CONFIG.get("gate", {}),
        "per_completion_opsd": True,
        "online_sft_on_all_wrong": True,
        "require_format_for_opsd": False,
    },
    "loss": {
        **base.DYME_OPSD_CONFIG.get("loss", {}),
        "loss_type": os.environ.get("DYME_OPSD_LOSS_TYPE", "srkl"),
        "srkl_alpha": float(os.environ.get("DYME_OPSD_SRKL_ALPHA", "0.1")),
        "opsd_weight": float(os.environ.get("DYME_OPSD_WEIGHT", "1.0")),
        "grpo_weight": float(os.environ.get("DYME_GRPO_WEIGHT", "1.0")),
        "teacher_traj_fkl_weight": float(os.environ.get("DYME_TEACHER_TRAJ_FKL_WEIGHT", "0.5")),
    },
    "teacher_probe": {
        "enabled": os.environ.get("DYME_TEACHER_PROBE", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        "context_providers": _probe_providers,
        "max_per_batch": int(os.environ.get("DYME_TEACHER_PROBE_MAX_PER_BATCH", "0")),
        "max_new_tokens": int(os.environ.get("DYME_TEACHER_PROBE_MAX_NEW_TOKENS", "96")),
        "do_sample": os.environ.get("DYME_TEACHER_PROBE_DO_SAMPLE", "0").strip().lower()
        in ("1", "true", "yes", "on"),
        "temperature": float(os.environ.get("DYME_TEACHER_PROBE_TEMPERATURE", "0.0")),
        "top_p": float(os.environ.get("DYME_TEACHER_PROBE_TOP_P", "1.0")),
        "repetition_penalty": float(os.environ.get("DYME_TEACHER_PROBE_REPETITION_PENALTY", "1.2")),
        "max_relative_change": float(os.environ.get("DYME_TEACHER_PROBE_RELAXED_TOL", "0.05")),
    },
    "teacher_trajectory": {
        "enabled": os.environ.get("DYME_TEACHER_TRAJECTORY", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        "loss_type": os.environ.get("DYME_TEACHER_TRAJ_LOSS_TYPE", "fkl"),
        "weight": float(os.environ.get("DYME_TEACHER_TRAJ_FKL_WEIGHT", "0.5")),
        "max_new_tokens": int(os.environ.get("DYME_TEACHER_TRAJ_MAX_NEW_TOKENS", "128")),
    },
    "visual_supervision": {
        "checker": {
            "enabled": os.environ.get("DYME_VISUAL_CHECKER", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            "model_source": "loaded_teacher",
        },
        "refiner": {
            "enabled": os.environ.get("DYME_VISUAL_REFINER", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            "model_source": "loaded_teacher",
            "include_gold": False,
        },
    },
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
}
