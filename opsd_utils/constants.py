MODE_GRPO = 0
MODE_OPSD = 1
MODE_SFT = 2

DEFAULT_OPSD_CONFIG = {
    "enabled": False,
    "mode": "dyme",
    "privileged_profile": "hybrid",
    "privileged_providers": ["text"],
    "privileged_image": {
        "mode": "dual",
        "crop_strategy": "bbox_then_center",
        "bbox_coord": "normalized",
        "margin_ratio": 0.25,
    },
    "privileged_debug": {
        "save_images": True,
        "image_subdir": "logs/images",
        "max_samples_per_detail": 2,
    },
    "gate": {
        "correct_threshold": 0.5,
        "teacher_recoverable": "privileged_available",
        "recoverable_tau": 0.5,
        "use_edge_mask": False,
        "per_completion_opsd": True,
        "require_format_for_opsd": True,
        "skip_degenerate_for_opsd": True,
    },
    "loss": {
        "beta": 0.5,
        "opsd_weight": 2.0,
        "grpo_weight": 1.0,
        "sft_weight": 1.0,
    },
}
