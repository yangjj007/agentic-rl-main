"""Regression tests for the explicit YAML teacher-probe recipe."""
from __future__ import annotations

from config.loader import load_config


def test_teacher_probe_recipe_has_explicit_batch_and_generation_limits() -> None:
    probe = load_config("opd_7b_dyme_probe")["opsd"]["teacher_probe"]
    assert probe["enabled"] is True
    assert probe["batch_size"] == 1
    assert probe["max_per_batch"] == 0
    assert probe["max_new_tokens"] == 96
    assert probe["context_providers"] == ["format_only", "visual_facts_deplot"]


def test_teacher_probe_recipe_is_not_overridden_by_environment(monkeypatch) -> None:
    before = load_config("opd_7b_dyme_probe")
    monkeypatch.setenv("DYME_TEACHER_PROBE_BATCH_SIZE", "8")
    monkeypatch.setenv("DYME_TEACHER_PROBE_PROVIDERS", "oracle_hint")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_MAX_NEW_TOKENS", "256")
    after = load_config("opd_7b_dyme_probe")
    assert after == before


def test_teacher_probe_recipe_has_explicit_repair_and_sampling_policy() -> None:
    opsd = load_config("opd_7b_dyme_probe")["opsd"]
    repair = opsd["teacher_correct_repair"]
    assert repair["mode"] == "opd"
    assert repair["scope"] == "all_wrong"
    assert repair["slots_per_prompt"] == 1
    assert repair["target_max_tokens"] == 256
    assert repair["sanitize_privileged"] is True
    assert opsd["effective_sampling"]["enabled"] is False


def test_teacher_probe_recipe_has_explicit_loss_schedule_and_route_cap() -> None:
    loss = load_config("opd_7b_dyme_probe")["opsd"]["loss"]
    assert loss["weight_decay"]["enabled"] is False
    assert loss["route_cap"]["enabled"] is False
