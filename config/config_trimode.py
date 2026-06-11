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

_detail_every_raw = os.environ.get("DYME_OPSD_DETAIL_EVERY", "10")
try:
    _detail_every = max(0, int(_detail_every_raw))
except ValueError:
    _detail_every = 10

_probe_raw = os.environ.get("DYME_OPSD_PROBE_ON_GENERATE", "1").strip().lower()
_probe_on_generate = _probe_raw not in ("0", "false", "no", "off")

_first_logits_raw = os.environ.get("DYME_OPSD_PROBE_FIRST_TOKEN_LOGITS", "1").strip().lower()
_probe_first_token_logits = _first_logits_raw not in ("0", "false", "no", "off")

_tail_raw = os.environ.get("DYME_OPSD_PROBE_PROMPT_TAIL_TOKENS", "16").strip()
try:
    _probe_prompt_tail_tokens = max(1, int(_tail_raw))
except ValueError:
    _probe_prompt_tail_tokens = 16

_model_ctx_raw = os.environ.get("DYME_OPSD_PROBE_LOG_MODEL_CONTEXT", "1").strip().lower()
_probe_log_model_context = _model_ctx_raw not in ("0", "false", "no", "off")

_health_raw = os.environ.get("DYME_OPSD_HEALTH_MONITOR", "1").strip().lower()
_health_monitor_enabled = _health_raw not in ("0", "false", "no", "off")

_require_format_raw = os.environ.get("DYME_OPSD_REQUIRE_FORMAT", "1").strip().lower()
_require_format_for_opsd = _require_format_raw not in ("0", "false", "no", "off")

DYME_OPSD_CONFIG = {
    **DYME_OPSD_CONFIG,
    "enabled": True,
    "mode": os.environ.get("DYME_OPSD_MODE", "trimode"),
    "privileged_profile": os.environ.get("DYME_OPSD_PRIVILEGE_PROFILE", "hybrid"),
    "privileged_providers": os.environ.get("DYME_OPSD_PROVIDERS", "text,visual_facts").split(","),
    "privileged_image": {
        **DYME_OPSD_CONFIG.get("privileged_image", {}),
        "mode": os.environ.get("DYME_OPSD_PRIVILEGE_IMAGE_MODE", "single"),
        "crop_strategy": os.environ.get("DYME_OPSD_CROP_STRATEGY", "bbox_then_center"),
        "bbox_coord": "normalized",
        "margin_ratio": float(os.environ.get("DYME_OPSD_CROP_MARGIN", "0.25")),
    },
    "privileged_debug": {
        **DYME_OPSD_CONFIG.get("privileged_debug", {}),
        "save_images": os.environ.get("DYME_OPSD_SAVE_PRIVILEGED_IMAGES", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        "image_subdir": os.environ.get("DYME_OPSD_PRIVILEGED_IMAGE_DIR", "logs/images"),
        "max_samples_per_detail": int(os.environ.get("DYME_OPSD_PRIVILEGED_IMAGE_MAX", "2")),
    },
    "gate": {
        **DYME_OPSD_CONFIG.get("gate", {}),
        "require_format_for_opsd": _require_format_for_opsd,
    },
    "debug": {
        **DYME_OPSD_CONFIG.get("debug", {}),
        "detail_every": _detail_every,
        "probe_on_generate": _probe_on_generate,
        "probe_first_token_logits": _probe_first_token_logits,
        "probe_prompt_tail_tokens": _probe_prompt_tail_tokens,
        "probe_log_model_context": _probe_log_model_context,
        "health_monitor": {
            **DYME_OPSD_CONFIG.get("debug", {}).get("health_monitor", {}),
            "enabled": _health_monitor_enabled,
        },
    },
}

CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "opsd": DYME_OPSD_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
}
