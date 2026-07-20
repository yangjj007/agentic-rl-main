from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT, MODE_SKIP
from opsd_utils.signal_aware_routing import (
    CompletionQuality,
    ModeStableRouteState,
    OpdRouteCapConfig,
    apply_opd_route_cap,
    SignalUtilityRoutingConfig,
    apply_signal_utility_routing,
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


def test_signal_utility_routes_early_all_wrong_teacher_correct_to_opd() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[False],
        group_reward_std=[0.0],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.0,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert modes == [MODE_OPSD]
    assert kept_trajs == {0}
    utility = stats.utilities[0]
    assert utility.opd > utility.grpo
    assert utility.opd > utility.sft
    assert stats.routed_opd == 1


def test_signal_utility_routes_late_mixed_correct_completion_to_grpo() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_GRPO, MODE_OPSD],
        teacher_traj_indices={1},
        student_correct=[True, False],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality(), CompletionQuality()],
        num_generations=2,
        readiness=0.85,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert modes[0] == MODE_GRPO
    assert stats.utilities[0].grpo > stats.utilities[0].opd
    assert stats.utilities[0].grpo > stats.utilities[0].sft
    assert stats.routed_grpo >= 1


def test_signal_utility_routes_format_bad_teacher_correct_to_sft_and_removes_traj() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[False],
        group_reward_std=[0.0],
        qualities=[CompletionQuality(degenerate=True)],
        num_generations=1,
        readiness=0.0,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert modes == [MODE_SFT]
    assert kept_trajs == set()
    utility = stats.utilities[0]
    assert utility.sft > utility.opd
    assert utility.sft > utility.grpo
    assert stats.routed_sft == 1
    assert stats.teacher_traj_removed == 1


def test_signal_utility_readiness_increases_grpo_and_decreases_opd_for_same_sample() -> None:
    low_modes, _, low_stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[True],
        group_reward_std=[0.08],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.05,
        config=SignalUtilityRoutingConfig(enabled=True),
    )
    high_modes, _, high_stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[True],
        group_reward_std=[0.08],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.85,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert low_modes == [MODE_OPSD]
    assert high_modes == [MODE_GRPO]
    assert high_stats.utilities[0].grpo > low_stats.utilities[0].grpo
    assert high_stats.utilities[0].opd < low_stats.utilities[0].opd


def test_signal_utility_reward_std_increases_grpo_utility() -> None:
    _, _, low_stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[True],
        group_reward_std=[0.0],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.5,
        config=SignalUtilityRoutingConfig(enabled=True),
    )
    _, _, high_stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        student_correct=[False],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.5,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert high_stats.utilities[0].grpo > low_stats.utilities[0].grpo
    assert high_stats.utilities[0].sft < low_stats.utilities[0].sft


def test_signal_utility_logs_raw_utilities_without_invalid_mask_sentinels() -> None:
    _, _, stats = apply_signal_utility_routing(
        completion_modes=[MODE_GRPO, MODE_OPSD],
        teacher_traj_indices={1},
        student_correct=[True, False],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality(), CompletionQuality()],
        num_generations=2,
        readiness=0.85,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert stats.grpo_mean > -10.0
    assert stats.opd_mean > -10.0
    assert stats.sft_mean > -10.0
    assert 0.0 <= stats.margin_mean < 10.0


def test_signal_utility_margin_is_zero_when_only_one_route_is_valid() -> None:
    _, _, stats = apply_signal_utility_routing(
        completion_modes=[MODE_GRPO],
        teacher_traj_indices=set(),
        student_correct=[True],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality(degenerate=True)],
        num_generations=1,
        readiness=0.5,
        config=SignalUtilityRoutingConfig(enabled=True),
    )

    assert stats.margin_mean == 0.0


def test_signal_utility_skips_clipped_completion_without_teacher_when_enabled() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_SFT],
        teacher_traj_indices=set(),
        teacher_correct_indices=set(),
        student_correct=[False],
        group_has_correct=[False],
        group_reward_std=[0.0],
        qualities=[CompletionQuality(clipped=True)],
        num_generations=1,
        readiness=0.5,
        config=SignalUtilityRoutingConfig(
            enabled=True,
            skip_clipped_without_teacher=True,
        ),
    )

    assert modes == [MODE_SKIP]
    assert kept_trajs == set()
    assert stats.routed_skip == 1
    assert stats.utilities[0].selected == MODE_SKIP


def test_mode_stable_utility_keeps_current_route_when_switch_gain_is_small() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        teacher_correct_indices={0},
        student_correct=[False],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.0,
        config=SignalUtilityRoutingConfig(
            enabled=True,
            mode_stable_enabled=True,
            mode_stable_ema_beta=1.0,
            mode_stable_switch_margin=0.20,
            mode_stable_min_hold_steps=0,
        ),
        state_keys=["sample0:0"],
        mode_stable_states={
            "sample0:0": ModeStableRouteState(
                previous_mode=MODE_SFT,
                utility_ema_grpo=0.0,
                utility_ema_opd=0.90,
                utility_ema_sft=0.82,
                hold_steps=3,
            )
        },
    )

    assert modes == [MODE_SFT]
    assert kept_trajs == set()
    assert stats.blocked_switches == 1
    assert stats.switch_gain_mean < 0.20
    assert stats.updated_stable_states["sample0:0"].previous_mode == MODE_SFT


def test_mode_stable_utility_switches_when_switch_gain_exceeds_margin() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        teacher_correct_indices={0},
        student_correct=[False],
        group_has_correct=[False],
        group_reward_std=[0.0],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.0,
        config=SignalUtilityRoutingConfig(
            enabled=True,
            mode_stable_enabled=True,
            mode_stable_ema_beta=1.0,
            mode_stable_switch_margin=0.20,
            mode_stable_min_hold_steps=0,
        ),
        state_keys=["sample0:0"],
        mode_stable_states={
            "sample0:0": ModeStableRouteState(
                previous_mode=MODE_SFT,
                utility_ema_grpo=0.0,
                utility_ema_opd=1.50,
                utility_ema_sft=1.00,
                hold_steps=3,
            )
        },
    )

    assert modes == [MODE_OPSD]
    assert kept_trajs == {0}
    assert stats.switches == 1
    assert stats.updated_stable_states["sample0:0"].previous_mode == MODE_OPSD


def test_mode_stable_utility_does_not_keep_invalid_current_route() -> None:
    modes, kept_trajs, stats = apply_signal_utility_routing(
        completion_modes=[MODE_GRPO],
        teacher_traj_indices=set(),
        teacher_correct_indices=set(),
        student_correct=[True],
        group_has_correct=[True],
        group_reward_std=[0.12],
        qualities=[CompletionQuality()],
        num_generations=1,
        readiness=0.5,
        config=SignalUtilityRoutingConfig(
            enabled=True,
            mode_stable_enabled=True,
            mode_stable_ema_beta=1.0,
            mode_stable_switch_margin=100.0,
            mode_stable_min_hold_steps=10,
        ),
        state_keys=["sample0:0"],
        mode_stable_states={
            "sample0:0": ModeStableRouteState(
                previous_mode=MODE_OPSD,
                utility_ema_grpo=1.00,
                utility_ema_opd=2.00,
                utility_ema_sft=0.00,
                hold_steps=10,
            )
        },
    )

    assert modes == [MODE_GRPO]
    assert kept_trajs == set()
    assert stats.invalid_current_switches == 1
    assert stats.updated_stable_states["sample0:0"].previous_mode == MODE_GRPO
