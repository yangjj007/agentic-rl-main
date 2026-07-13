from __future__ import annotations

import torch

from opsd_utils.global_training_signal import GlobalTrainingSignalCounts
from trainer.DyMETrainer import DyMETrainer


class ScalingAccelerator:
    device = torch.device("cpu")

    def __init__(self, scale: int = 1) -> None:
        self.scale = scale
        self.reduce_calls = 0

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        self.reduce_calls += 1
        return tensor * self.scale


def make_trainer(scale: int = 1) -> DyMETrainer:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer.accelerator = ScalingAccelerator(scale)
    trainer._metrics = {"train": {}}
    return trainer


def test_trainer_reduces_global_signal_counts_once() -> None:
    trainer = make_trainer(scale=8)

    snapshot = trainer._reduce_global_training_signal(
        mode="train",
        counts=GlobalTrainingSignalCounts(
            prompt_count=4,
            mixed_count=1,
            all_wrong_count=3,
            task_accuracy_zero_count=3,
            total_reward_zero_count=0,
            zero_signal_disagreement_count=3,
            completion_count=32,
            grpo_route_count=4,
            opd_route_count=12,
            sft_route_count=16,
            accuracy_reward_sum=2.0,
            clipped_count=8,
            eos_count=24,
            degenerate_count=6,
        ),
    )

    assert trainer.accelerator.reduce_calls == 1
    assert snapshot.prompt_count == 32
    assert snapshot.completion_count == 256
    assert snapshot.mixed_rate == 0.25
    assert snapshot.task_accuracy_zero_rate == 0.75
    assert snapshot.total_reward_zero_rate == 0.0
    assert snapshot.zero_signal_disagreement_rate == 0.75


def test_trainer_publishes_one_coherent_global_snapshot() -> None:
    trainer = make_trainer()

    snapshot = trainer._reduce_global_training_signal(
        mode="train",
        counts=GlobalTrainingSignalCounts(
            prompt_count=2,
            mixed_count=1,
            all_wrong_count=1,
            task_accuracy_zero_count=1,
            completion_count=16,
            grpo_route_count=4,
            opd_route_count=4,
            sft_route_count=8,
        ),
    )

    metrics = trainer._metrics["train"]
    assert metrics["global_signal/mixed_rate"][-1] == snapshot.mixed_rate
    assert metrics["global_signal/task_accuracy_zero_rate"][-1] == snapshot.task_accuracy_zero_rate
    assert metrics["global_signal/total_reward_zero_rate"][-1] == snapshot.total_reward_zero_rate
    assert metrics["global_signal/grpo_route_rate"][-1] == snapshot.grpo_route_rate
    assert metrics["global_signal/sft_route_rate"][-1] == snapshot.sft_route_rate
