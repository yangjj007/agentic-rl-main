from __future__ import annotations


def test_effective_group_filter_keeps_limited_all_wrong_rows_after_step() -> None:
    from opsd_utils.effective_group_filter import EffectiveGroupFilterConfig, compute_effective_group_keep_mask

    keep, stats = compute_effective_group_keep_mask(
        correct_counts=[0, 2, 8],
        num_generations=4,
        global_step=100,
        config=EffectiveGroupFilterConfig(
            enabled=True,
            after_step=50,
            all_wrong_keep_per_prompt=1,
            filter_all_correct=True,
        ),
    )

    assert keep == [
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert stats.filtered_all_wrong == 3
    assert stats.filtered_all_correct == 4
    assert stats.kept_all_wrong == 1
    assert stats.kept_mixed == 4


def test_effective_group_filter_disabled_before_after_step() -> None:
    from opsd_utils.effective_group_filter import EffectiveGroupFilterConfig, compute_effective_group_keep_mask

    keep, stats = compute_effective_group_keep_mask(
        correct_counts=[0, 4],
        num_generations=4,
        global_step=49,
        config=EffectiveGroupFilterConfig(enabled=True, after_step=50, all_wrong_keep_per_prompt=1),
    )

    assert keep == [True] * 8
    assert stats.filtered_total == 0


def test_effective_group_filter_can_drop_all_wrong_entirely() -> None:
    from opsd_utils.effective_group_filter import EffectiveGroupFilterConfig, compute_effective_group_keep_mask

    keep, stats = compute_effective_group_keep_mask(
        correct_counts=[0],
        num_generations=4,
        global_step=10,
        config=EffectiveGroupFilterConfig(enabled=True, after_step=0, all_wrong_keep_per_prompt=0),
    )

    assert keep == [False, False, False, False]
    assert stats.filtered_all_wrong == 4
    assert stats.kept_all_wrong == 0


def test_apply_effective_group_filter_zeroes_routes_and_removes_teacher_trajs() -> None:
    import torch

    from opsd_utils.effective_group_filter import apply_effective_group_filter_to_routes

    completion_masks = [torch.ones(2), torch.ones(2), torch.ones(2)]
    advantages = [torch.ones(2), torch.ones(2), torch.ones(2)]
    opsd_mask = [True, True, False]
    sft_replaced = [False, True, False]
    teacher_trajs = {
        0: (torch.tensor([1, 2]), torch.tensor([1, 1]), torch.tensor([1, 1])),
        1: (torch.tensor([3, 4]), torch.tensor([1, 1]), torch.tensor([1, 1])),
    }

    removed = apply_effective_group_filter_to_routes(
        keep_mask=[True, False, False],
        completion_masks=completion_masks,
        advantages=advantages,
        opsd_mask=opsd_mask,
        sft_replaced=sft_replaced,
        teacher_trajs=teacher_trajs,
    )

    assert removed == 1
    assert completion_masks[0].sum().item() == 2
    assert completion_masks[1].sum().item() == 0
    assert completion_masks[2].sum().item() == 0
    assert advantages[1].sum().item() == 0
    assert opsd_mask == [True, False, False]
    assert sft_replaced == [False, False, False]
    assert 0 in teacher_trajs
    assert 1 not in teacher_trajs
