from __future__ import annotations

import math

from opsd_utils.global_training_signal import (
    GlobalTrainingSignalCounts,
    counts_from_local_batch,
    snapshot_from_counts,
)


def test_snapshot_uses_count_weighted_global_denominators() -> None:
    snapshot = snapshot_from_counts(
        GlobalTrainingSignalCounts(
            prompt_count=8,
            mixed_count=2,
            all_wrong_count=5,
            all_correct_count=1,
            total_reward_zero_count=3,
            task_accuracy_zero_count=6,
            zero_signal_disagreement_count=3,
            completion_count=64,
            grpo_route_count=16,
            opd_route_count=24,
            sft_route_count=20,
            skip_route_count=4,
            accuracy_reward_sum=8.0,
            clipped_count=12,
            eos_count=48,
            degenerate_count=10,
        )
    )

    assert snapshot.mixed_rate == 0.25
    assert snapshot.all_wrong_rate == 0.625
    assert snapshot.task_accuracy_zero_rate == 0.75
    assert snapshot.total_reward_zero_rate == 0.375
    assert snapshot.zero_signal_disagreement_rate == 0.375
    assert snapshot.grpo_route_rate == 0.25
    assert snapshot.accuracy_reward_mean == 0.125
    assert snapshot.clipped_rate == 0.1875
    assert snapshot.eos_rate == 0.75


def test_total_reward_variance_does_not_hide_task_zero_signal() -> None:
    snapshot = snapshot_from_counts(
        GlobalTrainingSignalCounts(
            prompt_count=4,
            all_wrong_count=4,
            task_accuracy_zero_count=4,
            total_reward_zero_count=0,
            zero_signal_disagreement_count=4,
        )
    )

    assert snapshot.task_accuracy_zero_rate == 1.0
    assert snapshot.total_reward_zero_rate == 0.0
    assert snapshot.zero_signal_disagreement_rate == 1.0


def test_empty_counts_produce_finite_conservative_rates() -> None:
    snapshot = snapshot_from_counts(GlobalTrainingSignalCounts())

    assert snapshot.prompt_count == 0
    assert snapshot.completion_count == 0
    assert snapshot.task_accuracy_zero_rate == 1.0
    assert snapshot.total_reward_zero_rate == 1.0
    assert snapshot.all_wrong_rate == 1.0
    assert snapshot.grpo_route_rate == 0.0
    assert all(math.isfinite(float(value)) for value in snapshot.__dict__.values())


def test_local_batch_counts_compare_task_and_total_zero_per_prompt() -> None:
    counts = counts_from_local_batch(
        correct_counts=[0, 2, 8],
        total_reward_zero_flags=[False, False, True],
        num_generations=8,
        routes=["sft"] * 8 + ["grpo"] * 8 + ["opd"] * 8,
        accuracy_rewards=[0.0] * 8 + [1.0, 1.0] + [0.0] * 6 + [1.0] * 8,
        clipped_flags=[False] * 12 + [True] * 12,
        eos_flags=[True] * 12 + [False] * 12,
        degenerate_flags=[False] * 20 + [True] * 4,
    )

    assert counts.prompt_count == 3
    assert counts.mixed_count == 1
    assert counts.all_wrong_count == 1
    assert counts.all_correct_count == 1
    assert counts.task_accuracy_zero_count == 2
    assert counts.total_reward_zero_count == 1
    assert counts.zero_signal_disagreement_count == 1
    assert counts.grpo_route_count == 8
    assert counts.opd_route_count == 8
    assert counts.sft_route_count == 8
    assert counts.clipped_count == 12
    assert counts.eos_count == 12
    assert counts.degenerate_count == 4
