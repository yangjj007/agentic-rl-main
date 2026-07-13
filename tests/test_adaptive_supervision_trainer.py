from __future__ import annotations

from types import SimpleNamespace

import torch

from opsd_utils.adaptive_supervision import AdaptiveSupervisionState
from opsd_utils.global_training_signal import GlobalTrainingSignalCounts
from trainer.DyMETrainer import DyMETrainer


class FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    def __init__(self) -> None:
        self.reduce_calls = 0

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        self.reduce_calls += 1
        return tensor


def make_trainer(
    enabled: bool = True,
    readiness_source: str = "mixed_zero",
) -> DyMETrainer:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer.opsd_config = {
        "adaptive_supervision": {
            "enabled": enabled,
            "readiness_source": readiness_source,
            "ema_alpha": 1.0,
            "target_readiness": 0.20,
            "opsd_initial_weight": 1.5,
            "opsd_final_weight": 0.5,
            "teacher_initial_weight": 0.5,
            "teacher_final_weight": 0.0,
            "opd_initial_cap": 8,
            "opd_final_cap": 2,
        }
    }
    trainer.accelerator = FakeAccelerator()
    trainer._metrics = {"train": {}}
    trainer._init_adaptive_supervision_controller()
    return trainer


def test_trainer_builds_controller_only_when_enabled() -> None:
    assert make_trainer(enabled=True)._adaptive_supervision_controller is not None
    assert make_trainer(enabled=False)._adaptive_supervision_controller is None


def test_trainer_updates_controller_from_global_group_counts() -> None:
    trainer = make_trainer()

    state = trainer._update_adaptive_supervision(
        mode="train",
        global_step=3,
        prompt_count=10,
        mixed_count=4,
        zero_loss_count=5,
    )

    assert isinstance(state, AdaptiveSupervisionState)
    assert state.mixed_rate == 0.4
    assert state.zero_loss_rate == 0.5
    assert state.readiness == 0.2
    assert state.opsd_weight == 0.5
    assert state.teacher_traj_weight == 0.0
    assert state.opd_max_per_prompt == 2


def test_trainer_logs_one_coherent_adaptive_snapshot() -> None:
    trainer = make_trainer()

    state = trainer._update_adaptive_supervision(
        mode="train",
        global_step=1,
        prompt_count=10,
        mixed_count=2,
        zero_loss_count=5,
    )

    metrics = trainer._metrics["train"]
    assert metrics["adaptive/readiness"][-1] == state.readiness
    assert metrics["adaptive/mastery"][-1] == state.mastery
    assert metrics["adaptive/supervision"][-1] == state.supervision
    assert metrics["adaptive/opsd_weight"][-1] == state.opsd_weight
    assert metrics["adaptive/teacher_traj_weight"][-1] == state.teacher_traj_weight
    assert metrics["adaptive/opd_max_per_prompt"][-1] == state.opd_max_per_prompt


def test_adaptive_route_cap_config_uses_snapshot_without_schedule_boundary() -> None:
    trainer = make_trainer()
    trainer._update_adaptive_supervision(
        mode="train",
        global_step=0,
        prompt_count=10,
        mixed_count=2,
        zero_loss_count=5,
    )

    config = trainer._adaptive_opd_route_cap_config()

    assert config["enabled"] is True
    assert config["schedule_mode"] == "step"
    assert config["after_step"] == 0
    assert config["max_per_prompt"] == trainer._adaptive_supervision_state.opd_max_per_prompt


def test_disabled_controller_returns_legacy_fallbacks() -> None:
    trainer = make_trainer(enabled=False)

    assert trainer._update_adaptive_supervision(
        mode="train",
        global_step=0,
        prompt_count=4,
        mixed_count=1,
        zero_loss_count=2,
    ) is None
    assert trainer._adaptive_opd_route_cap_config() is None
    assert trainer._adaptive_loss_weights() is None


def test_adaptive_loss_weights_come_from_same_snapshot() -> None:
    trainer = make_trainer()
    state = trainer._update_adaptive_supervision(
        mode="train",
        global_step=0,
        prompt_count=10,
        mixed_count=2,
        zero_loss_count=5,
    )

    assert trainer._adaptive_loss_weights() == (
        state.opsd_weight,
        state.teacher_traj_weight,
    )


def test_direct_route_source_skips_pre_route_mixed_zero_update() -> None:
    trainer = make_trainer(readiness_source="global_grpo_route")

    state = trainer._update_adaptive_supervision(
        mode="train",
        global_step=3,
        prompt_count=10,
        mixed_count=4,
        zero_loss_count=5,
    )

    assert state == trainer._adaptive_supervision_controller.state
    assert state.update_count == 0
    assert trainer.accelerator.reduce_calls == 0


def test_direct_route_source_updates_and_logs_global_signal() -> None:
    trainer = make_trainer(readiness_source="global_grpo_route")

    state = trainer._update_adaptive_supervision_from_signal(
        mode="train",
        global_step=3,
        signal_rate=0.15,
    )

    assert state.update_count == 1
    assert state.readiness == 0.15
    assert trainer._metrics["train"]["adaptive/signal_rate"][-1] == 0.15
    assert trainer._metrics["train"]["adaptive/signal_ema"][-1] == 0.15


def test_global_route_snapshot_drives_direct_controller() -> None:
    trainer = make_trainer(readiness_source="global_grpo_route")

    trainer._reduce_global_training_signal(
        mode="train",
        global_step=7,
        counts=GlobalTrainingSignalCounts(
            completion_count=20,
            grpo_route_count=6,
            opd_route_count=8,
            sft_route_count=6,
        ),
    )

    state = trainer._adaptive_supervision_state
    assert state.step == 7
    assert state.readiness == 0.30
    assert state.supervision == 0.0


def test_direct_route_source_enables_required_global_signal_collection() -> None:
    trainer = make_trainer(readiness_source="global_grpo_route")

    assert trainer._global_signal_logging_enabled() is True


def test_adaptive_effective_sampler_is_active_from_step_zero_despite_legacy_boundary() -> None:
    trainer = make_trainer(readiness_source="global_grpo_route")
    trainer.opsd_config["effective_sampling"] = {
        "enabled": True,
        "after_step": 294,
        "schedule_mode": "step",
        "start_progress": 0.5,
    }
    trainer.train_dataset = list(range(16))
    trainer.num_generations = 8
    trainer.num_iterations = 1
    trainer.shuffle_dataset = False
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        seed=13,
    )

    sampler = trainer._get_train_sampler()
    sampler.set_step(0, max_steps=588)

    assert sampler.after_step == 294
    assert sampler.always_active is True
    assert sampler.enabled_for_step is True
