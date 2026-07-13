"""Tests for TrainingHealthMonitor alerts and correlation."""
from opsd_utils.health_monitor import (
    ALERT_GEN_CLIP_COLLAPSE,
    ALERT_GEN_REPEAT_DEGEN,
    ALERT_RL_ZERO_SIGNAL,
    ALERT_TEMPLATE_PARTIAL_DRIFT,
    ALERT_TEMPLATE_ANSWER_COLLAPSE,
    ALERT_TEMPLATE_SKELETON_COLLAPSE,
    TrainingHealthMonitor,
)


def test_clip_collapse_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(1)
    alerts = hm.record_generate(
        1,
        {"clipped_rate": 0.85, "eos_terminated_rate": 0.1, "degenerate_rate": 0.2, "repeat_loop_count": 0},
        {"p_greedy_first": 0.99, "p_eos_first": 1e-6},
    )
    assert ALERT_GEN_CLIP_COLLAPSE in alerts


def test_repeat_degen_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(2)
    alerts = hm.record_generate(
        2,
        {"clipped_rate": 0.3, "eos_terminated_rate": 0.5, "degenerate_rate": 0.6, "repeat_loop_count": 1},
        {},
    )
    assert ALERT_GEN_REPEAT_DEGEN in alerts


def test_template_skeleton_collapse_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(2)
    alerts = hm.record_generate(
        2,
        {
            "clipped_rate": 0.0,
            "eos_terminated_rate": 1.0,
            "degenerate_rate": 0.3,
            "repeat_loop_count": 0,
            "full_cot_template_rate": 0.95,
            "empty_cot_skeleton_rate": 0.25,
            "malformed_answer_section_rate": 0.25,
        },
        {},
    )
    assert ALERT_TEMPLATE_SKELETON_COLLAPSE in alerts
    metrics = hm.finish_step(2)
    assert metrics["completions/full_cot_template_rate"] == 0.95
    assert metrics["completions/empty_cot_skeleton_rate"] == 0.25


def test_template_malformed_answer_collapse_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(3)
    alerts = hm.record_generate(
        3,
        {
            "clipped_rate": 0.0,
            "eos_terminated_rate": 1.0,
            "degenerate_rate": 0.1,
            "repeat_loop_count": 0,
            "full_cot_template_rate": 0.95,
            "empty_cot_skeleton_rate": 0.05,
            "malformed_answer_section_rate": 0.3,
        },
        {},
    )
    assert ALERT_TEMPLATE_ANSWER_COLLAPSE in alerts


def test_partial_template_drift_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(4)
    alerts = hm.record_generate(
        4,
        {
            "clipped_rate": 1.0,
            "eos_terminated_rate": 0.0,
            "degenerate_rate": 1.0,
            "repeat_loop_count": 0,
            "full_cot_template_rate": 0.0,
            "partial_cot_template_rate": 0.75,
            "goal_without_answer_rate": 0.75,
        },
        {},
    )
    assert ALERT_TEMPLATE_PARTIAL_DRIFT in alerts
    metrics = hm.finish_step(4)
    assert metrics["completions/partial_cot_template_rate"] == 0.75
    assert metrics["completions/goal_without_answer_rate"] == 0.75


def test_rl_zero_signal_alert():
    hm = TrainingHealthMonitor({"enabled": True, "log_alerts_immediately": False})
    hm.reset_step(3)
    hm.record_loss(3, {"advantages_abs_mean": 0.0, "grpo_zero_loss_rate": 0.95})
    assert ALERT_RL_ZERO_SIGNAL in hm._step_alerts


def test_correlate_hints_after_history():
    hm = TrainingHealthMonitor({"enabled": True, "window": 5, "log_every_step": False})
    hm.reset_step(0)
    hm.record_generate(
        0,
        {"clipped_rate": 0.1, "eos_terminated_rate": 0.9, "degenerate_rate": 0.1, "repeat_loop_count": 0},
        {"p_greedy_first": 0.8, "p_eos_first": 0.01},
    )
    hm.record_optimizer(0, 0.5, 8e-5)
    hm.finish_step(0)

    hm.reset_step(1)
    hm.record_generate(
        1,
        {"clipped_rate": 0.9, "eos_terminated_rate": 0.05, "degenerate_rate": 0.5, "repeat_loop_count": 1},
        {"p_greedy_first": 0.995, "p_eos_first": 1e-6},
    )
    hm.record_optimizer(1, 2.5, 8e-5)
    corr = hm.correlate()
    assert "delta_clipped_rate" in corr or "root_cause_hints" in corr


def test_finish_step_returns_metrics_keys():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(1)
    hm.record_generate(
        1,
        {"clipped_rate": 0.2, "eos_terminated_rate": 0.8, "degenerate_rate": 0.1, "repeat_loop_count": 0},
        {"p_greedy_first": 0.9, "p_eos_first": 0.001},
    )
    hm.record_optimizer(1, 1.0, 8e-5)
    metrics = hm.finish_step(1)
    assert "completions/degenerate_rate" in metrics
    assert "health/alert_count" in metrics


def test_finish_step_returns_teacher_probe_diagnostic_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(2)
    hm.record_routing(
        2,
        {
            "teacher_probe_skipped_no_evidence_rate": 0.25,
            "teacher_probe_deplot_real_rate": 0.75,
            "teacher_probe_visual_fact_used_rate": 0.5,
            "teacher_probe_generated_tokens_mean": 42.0,
            "teacher_probe_generated_tokens_p95": 91.0,
            "teacher_probe_clipped_rate": 0.125,
        },
    )

    metrics = hm.finish_step(2)

    assert metrics["routing/teacher_probe_skipped_no_evidence_rate"] == 0.25
    assert metrics["routing/teacher_probe_deplot_real_rate"] == 0.75
    assert metrics["routing/teacher_probe_visual_fact_used_rate"] == 0.5
    assert metrics["teacher_probe/generated_tokens_mean"] == 42.0
    assert metrics["teacher_probe/generated_tokens_p95"] == 91.0
    assert metrics["teacher_probe/clipped_rate"] == 0.125


def test_finish_step_returns_adaptive_opsd_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(3)
    hm.record_loss(
        3,
        {
            "reward_std_mean": 0.05,
            "opsd_effective_weight": 2.7,
            "opsd_adaptive_multiplier": 1.8,
        },
    )

    metrics = hm.finish_step(3)

    assert metrics["signal/reward_std_mean"] == 0.05
    assert metrics["loss/opsd_effective_weight"] == 2.7
    assert metrics["loss/opsd_adaptive_multiplier"] == 1.8


def test_finish_step_returns_routing_count_and_group_signal_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(4)
    hm.record_routing(
        4,
        {
            "grpo_route_rate": 0.25,
            "opd_route_rate": 0.5,
            "sft_route_rate": 0.25,
            "total_completion_count": 8,
            "wrong_completion_count": 6,
            "probe_candidate_count": 4,
            "teacher_correct_count": 2,
            "opd_route_count": 3,
            "sft_route_count": 3,
            "grpo_route_count": 2,
            "group_all_wrong_rate": 0.75,
            "group_mixed_rate": 0.25,
            "reward_std_lt_0_01_rate": 0.5,
            "reward_std_lt_0_05_rate": 0.75,
            "reward_std_lt_0_10_rate": 1.0,
        },
    )

    metrics = hm.finish_step(4)

    assert metrics["routing/grpo_route_rate"] == 0.25
    assert metrics["routing/opd_route_rate"] == 0.5
    assert metrics["routing/total_completion_count"] == 8.0
    assert metrics["routing/probe_candidate_count"] == 4.0
    assert metrics["signal/group_all_wrong_rate"] == 0.75
    assert metrics["signal/reward_std_lt_0_05_rate"] == 0.75


def test_finish_step_returns_route_guard_and_perception_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(5)
    hm.record_routing(
        5,
        {
            "degenerate_hard_override_rate": 0.125,
            "clipped_hard_override_rate": 0.25,
            "teacher_correct_overridden_rate": 0.375,
            "signal_aware_sft_rate": 0.5,
            "perception_reward_mean": 0.4,
            "perception_reward_skipped_rate": 0.1,
            "perception_judge_parse_fail_rate": 0.05,
            "diagnostic_deplot_overlap_mean": 0.2,
        },
    )

    metrics = hm.finish_step(5)

    assert metrics["routing/degenerate_hard_override_rate"] == 0.125
    assert metrics["routing/clipped_hard_override_rate"] == 0.25
    assert metrics["routing/teacher_correct_overridden_rate"] == 0.375
    assert metrics["routing/signal_aware_sft_rate"] == 0.5
    assert metrics["reward/perception_mean"] == 0.4
    assert metrics["reward/perception_skipped_rate"] == 0.1
    assert metrics["reward/perception_judge_parse_fail_rate"] == 0.05
    assert metrics["reward/diagnostic_deplot_overlap_mean"] == 0.2


def test_finish_step_returns_teacher_sft_repair_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(6)
    hm.record_routing(
        6,
        {
            "teacher_sft_repair_rate": 0.125,
            "teacher_sft_repair_all_wrong_rate": 0.125,
            "teacher_sft_repair_slot_utilization": 1.0,
            "teacher_correct_to_opd_rate": 0.5,
            "teacher_correct_to_sft_repair_rate": 0.125,
            "repaired_prompt_to_mixed_rate": 0.25,
            "repaired_prompt_still_all_wrong_rate": 0.75,
            "teacher_sft_privileged_tag_rate": 0.0,
            "teacher_sft_target_student_short_rate": 0.625,
            "teacher_sft_target_answer_only_rate": 0.375,
            "teacher_sft_target_full_hint_format_rate": 0.125,
            "teacher_sft_target_exact_answer_line_rate": 1.0,
        },
    )

    metrics = hm.finish_step(6)

    assert metrics["routing/teacher_sft_repair_rate"] == 0.125
    assert metrics["routing/teacher_sft_repair_all_wrong_rate"] == 0.125
    assert metrics["routing/teacher_sft_repair_slot_utilization"] == 1.0
    assert metrics["routing/teacher_correct_to_opd_rate"] == 0.5
    assert metrics["routing/teacher_correct_to_sft_repair_rate"] == 0.125
    assert metrics["repair/repaired_prompt_to_mixed_rate"] == 0.25
    assert metrics["repair/repaired_prompt_still_all_wrong_rate"] == 0.75
    assert metrics["repair/teacher_sft_target_student_short_rate"] == 0.625
    assert metrics["repair/teacher_sft_target_answer_only_rate"] == 0.375
    assert metrics["repair/teacher_sft_target_full_hint_format_rate"] == 0.125
    assert metrics["repair/teacher_sft_target_exact_answer_line_rate"] == 1.0
    assert metrics["leakage/teacher_sft_privileged_tag_rate"] == 0.0


def test_finish_step_returns_effective_group_filter_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(7)
    hm.record_routing(
        7,
        {
            "effective_group_filter_enabled": 1.0,
            "effective_group_filtered_rate": 0.375,
            "effective_group_all_wrong_filtered_rate": 0.25,
            "effective_group_all_correct_filtered_rate": 0.125,
            "effective_group_kept_all_wrong_rate": 0.125,
            "effective_group_teacher_traj_removed_rate": 0.5,
        },
    )

    metrics = hm.finish_step(7)

    assert metrics["filter/effective_group_enabled"] == 1.0
    assert metrics["filter/effective_group_filtered_rate"] == 0.375
    assert metrics["filter/effective_group_all_wrong_filtered_rate"] == 0.25
    assert metrics["filter/effective_group_all_correct_filtered_rate"] == 0.125
    assert metrics["filter/effective_group_kept_all_wrong_rate"] == 0.125
    assert metrics["filter/effective_group_teacher_traj_removed_rate"] == 0.5


def test_finish_step_returns_phase_and_dynamic_trigger_metrics():
    hm = TrainingHealthMonitor({"enabled": True, "metrics_every_step": True, "log_every_step": False})
    hm.reset_step(8)
    hm.record_routing(
        8,
        {
            "training_progress": 0.5,
            "max_training_steps": 600,
            "teacher_traj_decay_active": 0.0,
            "effective_sampling_active": 1.0,
            "opd_decay_active": 1.0,
            "opd_route_cap_active": 1.0,
            "opd_route_cap_grpo_rate": 0.25,
            "opd_route_cap_skip_rate": 0.125,
            "skip_route_rate": 0.125,
        },
    )
    hm.record_loss(
        8,
        {
            "dynamic_mixed_rate_ema": 0.35,
            "dynamic_zero_loss_rate_ema": 0.20,
            "dynamic_mixed_ready": 1.0,
            "dynamic_zero_loss_ready": 1.0,
            "dynamic_joint_ready": 1.0,
            "dynamic_ready_streak": 20,
            "dynamic_would_trigger": 1.0,
        },
    )

    metrics = hm.finish_step(8)

    assert metrics["phase/training_progress"] == 0.5
    assert metrics["phase/max_training_steps"] == 600.0
    assert metrics["phase/effective_sampling_active"] == 1.0
    assert metrics["phase/opd_route_cap_active"] == 1.0
    assert metrics["phase/dynamic_mixed_rate_ema"] == 0.35
    assert metrics["phase/dynamic_zero_loss_rate_ema"] == 0.20
    assert metrics["phase/dynamic_ready_streak"] == 20.0
    assert metrics["phase/dynamic_would_trigger"] == 1.0
    assert metrics["routing/opd_route_cap_grpo_rate"] == 0.25
    assert metrics["routing/opd_route_cap_skip_rate"] == 0.125
    assert metrics["routing/skip_route_rate"] == 0.125
