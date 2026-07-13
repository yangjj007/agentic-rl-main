from __future__ import annotations

from collections import defaultdict

from opsd_utils.dynamic_trigger_monitor import DynamicTriggerConfig, DynamicTriggerMonitor
from trainer.DyMETrainer import DyMETrainer


def test_trainer_dynamic_trigger_uses_final_optimizer_step_health() -> None:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer._dynamic_trigger_monitor = DynamicTriggerMonitor(
        DynamicTriggerConfig(ema_alpha=1.0, min_progress=0.2, patience_steps=1)
    )
    trainer._dynamic_trigger_last_step = None
    trainer._metrics = {"train": defaultdict(list)}
    trainer.opsd_config = {
        "phase_schedule": {"mode": "progress"},
        "effective_sampling": {"after_step": 294, "start_progress": 0.5, "schedule_mode": "progress"},
        "loss": {
            "weight_decay": {"enabled": True, "start_step": 294, "start_progress": 0.5},
            "route_cap": {
                "enabled": True,
                "after_step": 294,
                "start_progress": 0.5,
                "schedule_mode": "progress",
            },
        },
        "teacher_trajectory": {
            "weight_decay": {"enabled": True, "start_step": 147, "start_progress": 0.25}
        },
    }
    trainer._max_training_steps = lambda: 100

    trainer._record_dynamic_trigger_metrics(
        mode="train",
        global_step=25,
        health_metrics={
            "signal/group_mixed_rate": 0.0,
            "signal/grpo_zero_loss_rate": 1.0,
        },
    )

    metrics = trainer._metrics["train"]
    assert metrics["phase/dynamic_mixed_rate_ema"] == [0.0]
    assert metrics["phase/dynamic_zero_loss_rate_ema"] == [1.0]
    assert metrics["phase/dynamic_sampling_needed_now"] == [1.0]
    assert metrics["phase/training_progress"] == [0.25]


def test_trainer_dynamic_trigger_updates_only_once_per_global_step() -> None:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer._dynamic_trigger_monitor = DynamicTriggerMonitor(
        DynamicTriggerConfig(ema_alpha=1.0, min_progress=0.0, patience_steps=2)
    )
    trainer._dynamic_trigger_last_step = None
    trainer._metrics = {"train": defaultdict(list)}
    trainer.opsd_config = {"phase_schedule": {"mode": "progress"}}
    trainer._max_training_steps = lambda: 10

    health = {"signal/group_mixed_rate": 0.0, "signal/grpo_zero_loss_rate": 1.0}
    trainer._record_dynamic_trigger_metrics(mode="train", global_step=1, health_metrics=health)
    trainer._record_dynamic_trigger_metrics(mode="train", global_step=1, health_metrics=health)

    assert trainer._metrics["train"]["phase/dynamic_sampling_needed_streak"] == [1.0]
