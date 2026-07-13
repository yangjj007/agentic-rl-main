from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GlobalTrainingSignalCounts:
    prompt_count: int = 0
    mixed_count: int = 0
    all_wrong_count: int = 0
    all_correct_count: int = 0
    total_reward_zero_count: int = 0
    task_accuracy_zero_count: int = 0
    zero_signal_disagreement_count: int = 0
    completion_count: int = 0
    grpo_route_count: int = 0
    opd_route_count: int = 0
    sft_route_count: int = 0
    skip_route_count: int = 0
    accuracy_reward_sum: float = 0.0
    clipped_count: int = 0
    eos_count: int = 0
    degenerate_count: int = 0


@dataclass(frozen=True)
class GlobalTrainingSignalSnapshot:
    prompt_count: int
    completion_count: int
    mixed_rate: float
    all_wrong_rate: float
    all_correct_rate: float
    total_reward_zero_rate: float
    task_accuracy_zero_rate: float
    zero_signal_disagreement_rate: float
    grpo_route_rate: float
    opd_route_rate: float
    sft_route_rate: float
    skip_route_rate: float
    accuracy_reward_mean: float
    clipped_rate: float
    eos_rate: float
    degenerate_rate: float


def _rate(numerator: float, denominator: int, *, empty: float = 0.0) -> float:
    if denominator <= 0:
        return float(empty)
    return float(numerator) / float(denominator)


def snapshot_from_counts(counts: GlobalTrainingSignalCounts) -> GlobalTrainingSignalSnapshot:
    prompts = max(int(counts.prompt_count), 0)
    completions = max(int(counts.completion_count), 0)
    return GlobalTrainingSignalSnapshot(
        prompt_count=prompts,
        completion_count=completions,
        mixed_rate=_rate(counts.mixed_count, prompts),
        all_wrong_rate=_rate(counts.all_wrong_count, prompts, empty=1.0),
        all_correct_rate=_rate(counts.all_correct_count, prompts),
        total_reward_zero_rate=_rate(counts.total_reward_zero_count, prompts, empty=1.0),
        task_accuracy_zero_rate=_rate(counts.task_accuracy_zero_count, prompts, empty=1.0),
        zero_signal_disagreement_rate=_rate(counts.zero_signal_disagreement_count, prompts),
        grpo_route_rate=_rate(counts.grpo_route_count, completions),
        opd_route_rate=_rate(counts.opd_route_count, completions),
        sft_route_rate=_rate(counts.sft_route_count, completions),
        skip_route_rate=_rate(counts.skip_route_count, completions),
        accuracy_reward_mean=_rate(counts.accuracy_reward_sum, completions),
        clipped_rate=_rate(counts.clipped_count, completions),
        eos_rate=_rate(counts.eos_count, completions),
        degenerate_rate=_rate(counts.degenerate_count, completions),
    )


def counts_from_local_batch(
    *,
    correct_counts: Sequence[int],
    total_reward_zero_flags: Sequence[bool],
    num_generations: int,
    routes: Sequence[str],
    accuracy_rewards: Sequence[float],
    clipped_flags: Sequence[bool],
    eos_flags: Sequence[bool],
    degenerate_flags: Sequence[bool],
) -> GlobalTrainingSignalCounts:
    prompt_count = len(correct_counts)
    if len(total_reward_zero_flags) != prompt_count:
        raise ValueError("total_reward_zero_flags must match correct_counts")
    completion_count = len(routes)
    for name, values in (
        ("accuracy_rewards", accuracy_rewards),
        ("clipped_flags", clipped_flags),
        ("eos_flags", eos_flags),
        ("degenerate_flags", degenerate_flags),
    ):
        if len(values) != completion_count:
            raise ValueError(f"{name} must match routes")

    mixed_flags = [0 < int(count) < int(num_generations) for count in correct_counts]
    all_wrong_flags = [int(count) == 0 for count in correct_counts]
    all_correct_flags = [int(count) >= int(num_generations) for count in correct_counts]
    task_zero_flags = [not mixed for mixed in mixed_flags]
    disagreement = [
        task_zero != bool(total_zero)
        for task_zero, total_zero in zip(task_zero_flags, total_reward_zero_flags)
    ]
    normalized_routes = [str(route).lower() for route in routes]
    return GlobalTrainingSignalCounts(
        prompt_count=prompt_count,
        mixed_count=sum(mixed_flags),
        all_wrong_count=sum(all_wrong_flags),
        all_correct_count=sum(all_correct_flags),
        total_reward_zero_count=sum(bool(value) for value in total_reward_zero_flags),
        task_accuracy_zero_count=sum(task_zero_flags),
        zero_signal_disagreement_count=sum(disagreement),
        completion_count=completion_count,
        grpo_route_count=sum(route == "grpo" for route in normalized_routes),
        opd_route_count=sum(route in ("opd", "opsd") for route in normalized_routes),
        sft_route_count=sum(route == "sft" for route in normalized_routes),
        skip_route_count=sum(route == "skip" for route in normalized_routes),
        accuracy_reward_sum=sum(float(value) for value in accuracy_rewards),
        clipped_count=sum(bool(value) for value in clipped_flags),
        eos_count=sum(bool(value) for value in eos_flags),
        degenerate_count=sum(bool(value) for value in degenerate_flags),
    )
