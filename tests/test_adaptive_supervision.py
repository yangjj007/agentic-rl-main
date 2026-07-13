from __future__ import annotations

import math

from opsd_utils.adaptive_supervision import (
    AdaptiveSupervisionConfig,
    AdaptiveSupervisionController,
)


def make_controller(**overrides) -> AdaptiveSupervisionController:
    values = {
        "ema_alpha": 0.10,
        "target_readiness": 0.20,
        "opsd_initial_weight": 1.50,
        "opsd_final_weight": 0.50,
        "teacher_initial_weight": 0.50,
        "teacher_final_weight": 0.0,
        "opd_initial_cap": 8,
        "opd_final_cap": 2,
    }
    values.update(overrides)
    return AdaptiveSupervisionController(AdaptiveSupervisionConfig(**values))


def test_controller_starts_conservatively_without_step_or_epoch_inputs() -> None:
    state = make_controller().state

    assert state.update_count == 0
    assert state.mixed_ema == 0.0
    assert state.zero_loss_ema == 1.0
    assert state.readiness == 0.0
    assert state.mastery == 0.0
    assert state.supervision == 1.0
    assert state.opsd_weight == 1.5
    assert state.teacher_traj_weight == 0.5
    assert state.opd_max_per_prompt == 8


def test_all_wrong_zero_signal_batches_keep_full_supervision() -> None:
    controller = make_controller()

    for step in range(20):
        state = controller.update(step=step, mixed_rate=0.0, zero_loss_rate=1.0)

    assert state.mastery == 0.0
    assert state.supervision == 1.0
    assert state.opsd_weight == 1.5
    assert state.teacher_traj_weight == 0.5
    assert state.opd_max_per_prompt == 8


def test_learning_signal_smoothly_reduces_all_supervision_actions() -> None:
    controller = make_controller(ema_alpha=1.0)

    state = controller.update(step=0, mixed_rate=0.2, zero_loss_rate=0.5)

    assert state.readiness == 0.1
    assert state.mastery == 0.1
    assert state.supervision == 0.5
    assert state.opsd_weight == 1.0
    assert state.teacher_traj_weight == 0.25
    assert state.opd_max_per_prompt == 5


def test_target_readiness_reaches_exact_final_actions() -> None:
    state = make_controller(ema_alpha=1.0).update(
        step=0,
        mixed_rate=0.4,
        zero_loss_rate=0.5,
    )

    assert state.readiness == 0.2
    assert state.supervision == 0.0
    assert state.opsd_weight == 0.5
    assert state.teacher_traj_weight == 0.0
    assert state.opd_max_per_prompt == 2


def test_mastery_and_actions_do_not_regress_when_signal_falls() -> None:
    controller = make_controller(ema_alpha=1.0)
    mature = controller.update(step=0, mixed_rate=0.4, zero_loss_rate=0.5)
    regressed = controller.update(step=1, mixed_rate=0.0, zero_loss_rate=1.0)

    assert regressed.readiness == 0.0
    assert regressed.mastery == mature.mastery
    assert regressed.supervision == mature.supervision
    assert regressed.opsd_weight == mature.opsd_weight
    assert regressed.teacher_traj_weight == mature.teacher_traj_weight
    assert regressed.opd_max_per_prompt == mature.opd_max_per_prompt


def test_conservative_ema_damps_a_single_early_spike() -> None:
    state = make_controller(ema_alpha=0.1).update(
        step=0,
        mixed_rate=1.0,
        zero_loss_rate=0.0,
    )

    assert math.isclose(state.mixed_ema, 0.1)
    assert math.isclose(state.zero_loss_ema, 0.9)
    assert math.isclose(state.readiness, 0.01)
    assert state.supervision > 0.99
    assert state.opd_max_per_prompt == 8


def test_duplicate_step_update_is_idempotent() -> None:
    controller = make_controller(ema_alpha=1.0)
    first = controller.update(step=4, mixed_rate=0.2, zero_loss_rate=0.5)
    duplicate = controller.update(step=4, mixed_rate=1.0, zero_loss_rate=0.0)

    assert duplicate == first
    assert duplicate.update_count == 1


def test_non_finite_rates_fall_back_to_conservative_signal() -> None:
    state = make_controller(ema_alpha=1.0).update(
        step=0,
        mixed_rate=float("nan"),
        zero_loss_rate=float("inf"),
    )

    assert state.mixed_ema == 0.0
    assert state.zero_loss_ema == 1.0
    assert state.supervision == 1.0


def test_identical_signals_produce_identical_actions_for_four_or_ten_epoch_labels() -> None:
    signals = [(0.0, 1.0), (0.1, 0.8), (0.25, 0.5), (0.4, 0.2)]
    four_epoch = make_controller()
    ten_epoch = make_controller()

    four_states = [
        four_epoch.update(step=i, mixed_rate=mixed, zero_loss_rate=zero)
        for i, (mixed, zero) in enumerate(signals)
    ]
    ten_states = [
        ten_epoch.update(step=i, mixed_rate=mixed, zero_loss_rate=zero)
        for i, (mixed, zero) in enumerate(signals)
    ]

    assert four_states == ten_states


def test_direct_signal_reaches_final_actions_at_target_rate() -> None:
    state = make_controller(ema_alpha=1.0, target_readiness=0.30).update_signal(
        step=0,
        signal_rate=0.30,
    )

    assert state.mixed_rate == 0.30
    assert state.mixed_ema == 0.30
    assert state.zero_loss_rate == 0.0
    assert state.zero_loss_ema == 0.0
    assert state.readiness == 0.30
    assert state.supervision == 0.0
    assert state.opsd_weight == 0.5
    assert state.teacher_traj_weight == 0.0
    assert state.opd_max_per_prompt == 2


def test_direct_signal_mastery_does_not_regress() -> None:
    controller = make_controller(ema_alpha=1.0, target_readiness=0.30)
    mature = controller.update_signal(step=0, signal_rate=0.15)
    regressed = controller.update_signal(step=1, signal_rate=0.0)

    assert regressed.readiness == 0.0
    assert regressed.mastery == mature.mastery
    assert regressed.supervision == mature.supervision


def test_direct_signal_duplicate_step_is_idempotent() -> None:
    controller = make_controller(ema_alpha=1.0, target_readiness=0.30)
    first = controller.update_signal(step=4, signal_rate=0.10)
    duplicate = controller.update_signal(step=4, signal_rate=0.30)

    assert duplicate == first
    assert duplicate.update_count == 1
