"""Shared visual_supervision block for OPD / DyME configs."""
from __future__ import annotations

from config.env_overrides import env_bool, env_float, env_int, env_str

VISUAL_CHECKER_ENABLED = True
VISUAL_REFINER_ENABLED = True

VISUAL_CHECKER_ENABLED = env_bool("DYME_VISUAL_CHECKER", VISUAL_CHECKER_ENABLED)
VISUAL_REFINER_ENABLED = env_bool("DYME_VISUAL_REFINER", VISUAL_REFINER_ENABLED)


def build_visual_supervision_config() -> dict:
    """7B teacher Visual Checker / Refiner settings (env-overridable)."""
    max_ic_tokens = env_int("DYME_VISUAL_MAX_IC_TOKENS", 512)
    max_refine_tokens = env_int("DYME_VISUAL_MAX_REFINE_TOKENS", 512)
    return {
        "enabled": VISUAL_CHECKER_ENABLED or VISUAL_REFINER_ENABLED,
        "ic_source": env_str("DYME_VISUAL_IC_SOURCE", "auto"),
        "prefetch_ic": env_bool("DYME_VISUAL_PREFETCH_IC", True),
        "dedupe_per_batch": env_bool("DYME_VISUAL_DEDUPE", True),
        "teacher_batch_size": 4,
        "checker": {
            "enabled": VISUAL_CHECKER_ENABLED,
            "model_source": "loaded_teacher",
            "grounding": env_str("DYME_VISUAL_CHECKER_GROUNDING", "image_primary"),
            "aux_evidence": env_str("DYME_VISUAL_CHECKER_AUX", "none"),
            "max_per_batch": env_int("DYME_VISUAL_CHECKER_MAX_PER_BATCH", 0),
            "max_ic_tokens": max_ic_tokens,
            "fallback": "local",
        },
        "refiner": {
            "enabled": VISUAL_REFINER_ENABLED,
            "model_source": "loaded_teacher",
            "scope": "batch_all",
            "fallback": "passthrough",
            "include_gold": False,
            "skip_cold_start": env_bool("DYME_VISUAL_REFINER_SKIP_COLD_START", True),
            "max_ic_tokens": max_ic_tokens,
            "max_refine_tokens": max_refine_tokens,
        },
        "template_pool": {
            "path": env_str("DYME_VISUAL_TEMPLATE_PATH", "best_template.txt"),
            "refresh_interval_sec": env_float("DYME_VISUAL_TEMPLATE_REFRESH_SEC", 60.0),
        },
        "logging": {
            "enabled": env_bool("DYME_VISUAL_LOG", True),
            "sample_count": env_int("DYME_VISUAL_LOG_SAMPLES", 3),
            "preview_chars": env_int("DYME_VISUAL_LOG_PREVIEW_CHARS", 400),
            "save_artifacts": env_bool("DYME_VISUAL_SAVE_ARTIFACTS", True),
            "log_route_binding": env_bool("DYME_VISUAL_LOG_ROUTE", True),
        },
    }
