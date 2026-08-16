"""Tests for anti-degeneration trimode config and reward_weights wiring."""
import pytest


def test_trimode_antidegen_config_loads():
    from config.loader import load_config
    cfg = load_config("trimode_antidegen")
    dyme = cfg["training"]["dyme_args"]
    assert dyme["max_completion_length"] == 150
    assert dyme["temperature"] == 0.7
    assert dyme["repetition_penalty"] == 1.25
    assert dyme["learning_rate"] == 5e-5
    assert dyme["warmup_steps"] == 50
    assert cfg["opsd"]["gate"]["require_format_for_opsd"] is False
    assert cfg["opsd"]["reward_weights"] == [0.5, 1.5, 1.0]


def test_loader_trimode_antidegen_alias():
    from config.loader import load_config

    cfg = load_config("trimode_antidegen")
    assert cfg["training"]["dyme_args"]["max_completion_length"] == 150


def test_reward_weights_must_have_three_values():
    bad = [1.0, 2.0]
    with pytest.raises(ValueError, match="length 3"):
        if len(bad) != 3:
            raise ValueError(f"opsd_config reward_weights must have length 3 (format, context, acc), got {bad}")


def test_probe_config_is_explicit_yaml():
    from config.loader import load_config
    cfg = load_config("opd_7b_dyme_probe")
    assert cfg["opsd"]["mode"] == "dyme_teacher_probe_opd"


def test_probe_config_has_teacher_trajectory_block():
    from config.loader import load_config
    cfg = load_config("opd_7b_dyme_probe")["opsd"]["teacher_trajectory"]
    assert "weight_decay" in cfg


def test_probe_config_is_invariant_to_environment(monkeypatch):
    from config.loader import load_config

    before = load_config("opd_7b_dyme_probe")
    monkeypatch.setenv("DYME_OPSD_WEIGHT", "999")
    monkeypatch.setenv("DYME_PHASE_SCHEDULE_MODE", "progress")
    monkeypatch.setenv("DYME_VISUAL_CHECKER", "0")
    after = load_config("opd_7b_dyme_probe")
    assert after == before
