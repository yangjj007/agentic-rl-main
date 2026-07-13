from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT, MODE_SKIP
from opsd_utils.signal_aware_routing import (
    CompletionQuality,
    OpdRouteCapConfig,
    apply_opd_route_cap,
    SignalAwareRoutingConfig,
    apply_signal_aware_routing,
    local_teacher_traj_indices,
)


def test_teacher_correct_degenerate_forced_to_sft_and_traj_removed() -> None:
    modes, kept_trajs, stats = apply_signal_aware_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        qualities=[CompletionQuality(degenerate=True)],
        group_reward_std=[0.2],
        num_generations=1,
        config=SignalAwareRoutingConfig(signal_aware=True, degenerate_hard_override=True),
    )

    assert modes == [MODE_SFT]
    assert kept_trajs == set()
    assert stats.degenerate_hard_overrides == 1
    assert stats.teacher_correct_overrides == 1


def test_teacher_correct_clipped_forced_to_sft() -> None:
    modes, kept_trajs, stats = apply_signal_aware_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        qualities=[CompletionQuality(clipped=True)],
        group_reward_std=[0.2],
        num_generations=1,
        config=SignalAwareRoutingConfig(signal_aware=True, clipped_hard_override=True),
    )

    assert modes == [MODE_SFT]
    assert kept_trajs == set()
    assert stats.clipped_hard_overrides == 1


def test_clean_teacher_correct_wrong_completion_stays_opd() -> None:
    modes, kept_trajs, stats = apply_signal_aware_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        qualities=[CompletionQuality()],
        group_reward_std=[0.2],
        num_generations=1,
        config=SignalAwareRoutingConfig(signal_aware=True, degenerate_hard_override=True),
    )

    assert modes == [MODE_OPSD]
    assert kept_trajs == {0}
    assert stats.total_overrides == 0


def test_clean_correct_completion_stays_grpo_when_reward_std_is_healthy() -> None:
    modes, kept_trajs, stats = apply_signal_aware_routing(
        completion_modes=[MODE_GRPO],
        teacher_traj_indices=set(),
        qualities=[CompletionQuality()],
        group_reward_std=[0.2],
        num_generations=1,
        config=SignalAwareRoutingConfig(signal_aware=True, reward_std_min=0.05),
    )

    assert modes == [MODE_GRPO]
    assert kept_trajs == set()
    assert stats.total_overrides == 0


def test_low_signal_grpo_completion_is_routed_to_sft_without_delaying_pcd_probe() -> None:
    modes, kept_trajs, stats = apply_signal_aware_routing(
        completion_modes=[MODE_GRPO, MODE_OPSD],
        teacher_traj_indices={1},
        qualities=[CompletionQuality(), CompletionQuality()],
        group_reward_std=[0.0],
        num_generations=2,
        config=SignalAwareRoutingConfig(signal_aware=True, reward_std_min=0.05),
    )

    assert modes == [MODE_SFT, MODE_OPSD]
    assert kept_trajs == {1}
    assert stats.signal_aware_sft == 1


def test_teacher_traj_indices_empty_when_teacher_prompt_missing() -> None:
    indices = local_teacher_traj_indices(
        teacher_traj_mask=[False, True, True],
        has_teacher_prompt_ids=False,
    )

    assert indices == []


def test_teacher_traj_indices_kept_when_teacher_prompt_present() -> None:
    indices = local_teacher_traj_indices(
        teacher_traj_mask=[False, True, True],
        has_teacher_prompt_ids=True,
    )

    assert indices == [1, 2]


def test_opd_route_cap_limits_late_opd_per_prompt_and_removes_trajs() -> None:
    modes, kept_trajs, stats = apply_opd_route_cap(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_OPSD, MODE_GRPO, MODE_OPSD],
        teacher_traj_indices={0, 1, 2, 4},
        num_generations=5,
        global_step=294,
        config=OpdRouteCapConfig(enabled=True, max_per_prompt=2, after_step=294),
    )

    assert modes == [MODE_OPSD, MODE_OPSD, MODE_SFT, MODE_GRPO, MODE_SFT]
    assert kept_trajs == {0, 1}
    assert stats.capped == 2
    assert stats.teacher_traj_removed == 2
    assert stats.eligible_prompts == 1


def test_opd_route_cap_disabled_before_after_step() -> None:
    modes, kept_trajs, stats = apply_opd_route_cap(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_OPSD],
        teacher_traj_indices={0, 1, 2},
        num_generations=3,
        global_step=293,
        config=OpdRouteCapConfig(enabled=True, max_per_prompt=1, after_step=294),
    )

    assert modes == [MODE_OPSD, MODE_OPSD, MODE_OPSD]
    assert kept_trajs == {0, 1, 2}
    assert stats.capped == 0


def test_opd_route_cap_routes_mixed_overflow_back_to_grpo() -> None:
    modes, kept_trajs, stats = apply_opd_route_cap(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_OPSD, MODE_OPSD],
        teacher_traj_indices={0, 1, 2, 3},
        group_has_correct=[True],
        num_generations=4,
        global_step=50,
        max_steps=100,
        config=OpdRouteCapConfig(
            enabled=True,
            max_per_prompt=2,
            after_step=294,
            start_progress=0.5,
            schedule_mode="progress",
            overflow_route="mixed_grpo_all_wrong_skip",
        ),
    )

    assert modes == [MODE_OPSD, MODE_OPSD, MODE_GRPO, MODE_GRPO]
    assert kept_trajs == {0, 1}
    assert stats.capped == 2
    assert stats.rerouted_grpo == 2
    assert stats.skipped == 0
    assert stats.teacher_traj_removed == 2


def test_opd_route_cap_skips_all_wrong_overflow() -> None:
    modes, kept_trajs, stats = apply_opd_route_cap(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_OPSD, MODE_OPSD],
        teacher_traj_indices={0, 1, 2, 3},
        group_has_correct=[False],
        num_generations=4,
        global_step=50,
        max_steps=100,
        config=OpdRouteCapConfig(
            enabled=True,
            max_per_prompt=2,
            after_step=294,
            start_progress=0.5,
            schedule_mode="progress",
            overflow_route="mixed_grpo_all_wrong_skip",
        ),
    )

    assert modes == [MODE_OPSD, MODE_OPSD, MODE_SKIP, MODE_SKIP]
    assert kept_trajs == {0, 1}
    assert stats.capped == 2
    assert stats.rerouted_grpo == 0
    assert stats.skipped == 2
    assert stats.teacher_traj_removed == 2
