"""Tests for anti-degeneration trimode config and reward_weights wiring."""
import importlib

import pytest


def test_trimode_antidegen_config_loads():
    mod = importlib.import_module("config.config_trimode_antidegen")
    cfg = mod.CONFIG
    dyme = cfg["training"]["dyme_args"]
    assert dyme["max_completion_length"] == 150
    assert dyme["temperature"] == 0.7
    assert dyme["repetition_penalty"] == 1.25
    assert dyme["learning_rate"] == 5e-5
    assert dyme["warmup_steps"] == 50
    assert cfg["opsd"]["gate"]["require_format_for_opsd"] is False
    assert cfg["opsd"]["reward_weights"] == [0.5, 1.5, 1.0]


def test_trimode_require_format_env(monkeypatch):
    import config.config_trimode as trimode_mod

    monkeypatch.setenv("DYME_OPSD_REQUIRE_FORMAT", "0")
    mod = importlib.reload(trimode_mod)
    assert mod.DYME_OPSD_CONFIG["gate"]["require_format_for_opsd"] is False

    monkeypatch.setenv("DYME_OPSD_REQUIRE_FORMAT", "1")
    mod = importlib.reload(trimode_mod)
    assert mod.DYME_OPSD_CONFIG["gate"]["require_format_for_opsd"] is True


def test_loader_trimode_antidegen_alias():
    from config.loader import load_config

    cfg = load_config("trimode_antidegen")
    assert cfg["training"]["dyme_args"]["max_completion_length"] == 150


def test_reward_weights_must_have_three_values():
    bad = [1.0, 2.0]
    with pytest.raises(ValueError, match="length 3"):
        if len(bad) != 3:
            raise ValueError(f"opsd_config reward_weights must have length 3 (format, context, acc), got {bad}")


def test_probe_config_eval_format_reward_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.delenv("DYME_EVAL_FORMAT_REWARD", raising=False)
    monkeypatch.delenv("DYME_EVAL_FORMAT_REWARD_WEIGHT", raising=False)
    mod = importlib.reload(probe_mod)
    assert mod.DYME_OPSD_CONFIG["eval_format_reward"]["enabled"] is False
    assert mod.DYME_OPSD_CONFIG["eval_format_reward"]["weight"] == 0.1

    monkeypatch.setenv("DYME_EVAL_FORMAT_REWARD", "1")
    monkeypatch.setenv("DYME_EVAL_FORMAT_REWARD_WEIGHT", "0.25")
    mod = importlib.reload(probe_mod)
    assert mod.DYME_OPSD_CONFIG["eval_format_reward"]["enabled"] is True
    assert mod.DYME_OPSD_CONFIG["eval_format_reward"]["weight"] == 0.25


def test_probe_config_teacher_traj_decay_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.setenv("DYME_TEACHER_TRAJ_WEIGHT_DECAY", "1")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_DECAY_START_STEP", "294")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_DECAY_END_STEP", "441")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_FINAL_WEIGHT", "0.0")
    mod = importlib.reload(probe_mod)

    cfg = mod.DYME_OPSD_CONFIG["teacher_trajectory"]
    assert cfg["weight_decay"]["enabled"] is True
    assert cfg["weight_decay"]["start_step"] == 294
    assert cfg["weight_decay"]["end_step"] == 441
    assert cfg["weight_decay"]["final_weight"] == 0.0


def test_probe_config_progress_schedule_and_dynamic_trigger_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.setenv("DYME_PHASE_SCHEDULE_MODE", "progress")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_DECAY_START_PROGRESS", "0.25")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_DECAY_END_PROGRESS", "0.50")
    monkeypatch.setenv("DYME_OPSD_DECAY_START_PROGRESS", "0.50")
    monkeypatch.setenv("DYME_OPSD_DECAY_END_PROGRESS", "0.75")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_START_PROGRESS", "0.50")
    monkeypatch.setenv("DYME_OPSD_ROUTE_CAP_START_PROGRESS", "0.50")
    monkeypatch.setenv("DYME_OPSD_OVERFLOW_ROUTE", "mixed_grpo_all_wrong_skip")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_EMA_ALPHA", "0.15")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_MIN_PROGRESS", "0.20")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_PATIENCE", "12")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_SAMPLING_MIXED_MAX", "0.18")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN", "0.72")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_RL_MIXED_MIN", "0.32")
    monkeypatch.setenv("DYME_DYNAMIC_TRIGGER_RL_ZERO_MAX", "0.28")
    mod = importlib.reload(probe_mod)

    cfg = mod.DYME_OPSD_CONFIG
    assert cfg["phase_schedule"]["mode"] == "progress"
    assert cfg["teacher_trajectory"]["weight_decay"]["start_progress"] == 0.25
    assert cfg["teacher_trajectory"]["weight_decay"]["end_progress"] == 0.50
    assert cfg["loss"]["weight_decay"]["start_progress"] == 0.50
    assert cfg["loss"]["weight_decay"]["end_progress"] == 0.75
    assert cfg["loss"]["route_cap"]["schedule_mode"] == "progress"
    assert cfg["loss"]["route_cap"]["start_progress"] == 0.50
    assert cfg["loss"]["route_cap"]["overflow_route"] == "mixed_grpo_all_wrong_skip"
    assert cfg["effective_sampling"]["schedule_mode"] == "progress"
    assert cfg["effective_sampling"]["start_progress"] == 0.50
    assert cfg["dynamic_trigger_monitor"] == {
        "enabled": True,
        "ema_alpha": 0.15,
        "min_progress": 0.20,
        "patience_steps": 12,
        "sampling_mixed_max": 0.18,
        "sampling_zero_loss_min": 0.72,
        "rl_mixed_min": 0.32,
        "rl_zero_loss_max": 0.28,
    }


def test_probe_config_adaptive_supervision_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.setenv("DYME_ADAPTIVE_SUPERVISION", "1")
    monkeypatch.setenv("DYME_ADAPTIVE_READINESS_SOURCE", "global_grpo_route")
    monkeypatch.setenv("DYME_ADAPTIVE_EMA_ALPHA", "0.15")
    monkeypatch.setenv("DYME_ADAPTIVE_TARGET_READINESS", "0.24")
    monkeypatch.setenv("DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT", "1.4")
    monkeypatch.setenv("DYME_ADAPTIVE_OPSD_FINAL_WEIGHT", "0.4")
    monkeypatch.setenv("DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT", "0.45")
    monkeypatch.setenv("DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT", "0.05")
    monkeypatch.setenv("DYME_ADAPTIVE_OPSD_INITIAL_CAP", "8")
    monkeypatch.setenv("DYME_ADAPTIVE_OPSD_FINAL_CAP", "1")
    mod = importlib.reload(probe_mod)

    cfg = mod.DYME_OPSD_CONFIG["adaptive_supervision"]
    assert cfg == {
        "enabled": True,
        "readiness_source": "global_grpo_route",
        "ema_alpha": 0.15,
        "target_readiness": 0.24,
        "opsd_initial_weight": 1.4,
        "opsd_final_weight": 0.4,
        "teacher_initial_weight": 0.45,
        "teacher_final_weight": 0.05,
        "opd_initial_cap": 8,
        "opd_final_cap": 1,
    }


def test_probe_config_global_signal_logging_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.setenv("DYME_GLOBAL_SIGNAL_LOGGING", "1")
    mod = importlib.reload(probe_mod)

    assert mod.DYME_OPSD_CONFIG["global_signal_logging"] == {"enabled": True}


def test_probe_config_chart_cot_quality_gate_env(monkeypatch):
    import config.config_opd_7b_dyme_probe as probe_mod

    monkeypatch.setenv("DYME_CHART_COT_VERIFY", "1")
    monkeypatch.setenv("DYME_CHART_COT_GATE_MODE", "gate")
    monkeypatch.setenv("DYME_CHART_COT_REQUIRE_Q3", "1")
    monkeypatch.setenv("DYME_CHART_COT_LOG_SAMPLES", "1")
    monkeypatch.setenv("DYME_CHART_COT_MAX_LOG_SAMPLES", "5")
    mod = importlib.reload(probe_mod)

    assert mod.DYME_OPSD_CONFIG["chart_cot_quality_gate"] == {
        "enabled": True,
        "mode": "gate",
        "require_quality": "Q3",
        "log_samples": True,
        "max_log_samples": 5,
    }
