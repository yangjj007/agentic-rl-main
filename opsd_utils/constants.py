MODE_GRPO = 0
MODE_OPSD = 1
MODE_SFT = 2

DEFAULT_OPSD_CONFIG = {
    "enabled": False,
    "mode": "dyme",
    "privileged_providers": ["text"],
    "gate": {
        "correct_threshold": 0.5,
        "teacher_recoverable": "privileged_available",
        "recoverable_tau": 0.5,
        "use_edge_mask": False,
    },
    "loss": {
        "beta": 0.5,
        "opsd_weight": 1.0,
        "grpo_weight": 1.0,
        "sft_weight": 1.0,
    },
}
