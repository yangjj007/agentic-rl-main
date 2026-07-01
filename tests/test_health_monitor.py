"""Tests for TrainingHealthMonitor alerts and correlation."""
from opsd_utils.health_monitor import (
    ALERT_GEN_CLIP_COLLAPSE,
    ALERT_GEN_REPEAT_DEGEN,
    ALERT_RL_ZERO_SIGNAL,
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
