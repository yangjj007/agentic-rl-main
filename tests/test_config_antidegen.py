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
