from __future__ import annotations

from trainer.DyMETrainer import DynamicSignalRepeatSampler


def test_dynamic_signal_sampler_updates_prompt_weights_by_signal() -> None:
    sampler = DynamicSignalRepeatSampler(
        data_source=list(range(4)),
        mini_repeat_count=2,
        batch_size=2,
        seed=123,
        after_step=1,
        mixed_weight=4.0,
        all_wrong_weight=1.0,
        all_correct_weight=0.5,
        reward_std_bonus=2.0,
    )

    sampler.update_prompt_signal(dataset_index=0, correct_count=0, num_generations=8, reward_std=0.0)
    sampler.update_prompt_signal(dataset_index=1, correct_count=3, num_generations=8, reward_std=0.25)
    sampler.update_prompt_signal(dataset_index=2, correct_count=8, num_generations=8, reward_std=0.1)

    assert sampler.prompt_states[:3] == ["all_wrong", "mixed", "all_correct"]
    assert sampler.prompt_weights[1] > sampler.prompt_weights[0]
    assert sampler.prompt_weights[0] > sampler.prompt_weights[2]


def test_dynamic_signal_sampler_repeats_each_selected_index() -> None:
    sampler = DynamicSignalRepeatSampler(
        data_source=list(range(3)),
        mini_repeat_count=3,
        batch_size=1,
        seed=123,
        after_step=0,
        mixed_weight=10.0,
    )
    sampler.set_step(1)
    sampler.update_prompt_signal(dataset_index=2, correct_count=1, num_generations=3, reward_std=0.2)

    values = list(iter(sampler))
    assert len(values) == 9
    for offset in range(0, len(values), 3):
        assert values[offset : offset + 3] == [values[offset]] * 3


def test_dynamic_signal_sampler_can_activate_mid_iterator() -> None:
    sampler = DynamicSignalRepeatSampler(
        data_source=list(range(2)),
        mini_repeat_count=1,
        batch_size=1,
        seed=123,
        after_step=1,
        mixed_weight=1_000_000.0,
        all_wrong_weight=0.001,
        unknown_weight=0.001,
    )
    iterator = iter(sampler)
    first = next(iterator)

    sampler.update_prompt_signal(dataset_index=1, correct_count=1, num_generations=2, reward_std=0.0)
    sampler.set_step(1)
    second = next(iterator)

    assert first in {0, 1}
    assert second == 1


def test_dynamic_signal_sampler_can_activate_by_training_progress() -> None:
    sampler = DynamicSignalRepeatSampler(
        data_source=list(range(2)),
        mini_repeat_count=1,
        batch_size=1,
        seed=123,
        after_step=999,
        schedule_mode="progress",
        start_progress=0.5,
    )

    sampler.set_step(49, max_steps=100)
    assert sampler.enabled_for_step is False

    sampler.set_step(50, max_steps=100)
    assert sampler.enabled_for_step is True
