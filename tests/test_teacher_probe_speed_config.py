from __future__ import annotations

import importlib
import os


def test_teacher_probe_batch_size_env_reaches_config(monkeypatch) -> None:
    monkeypatch.setenv("DYME_TEACHER_PROBE_BATCH_SIZE", "8")
    module = importlib.import_module("config.config_opd_7b_dyme_probe")
    module = importlib.reload(module)

    try:
        assert module.CONFIG["opsd"]["teacher_probe"]["batch_size"] == 8
    finally:
        os.environ.pop("DYME_TEACHER_PROBE_BATCH_SIZE", None)
        importlib.reload(module)


def test_oracle_hint_teacher_probe_env_reaches_config(monkeypatch) -> None:
    monkeypatch.setenv("DYME_TEACHER_PROBE_PROVIDERS", "format_only,oracle_hint,visual_facts_deplot")
    monkeypatch.setenv("DYME_TEACHER_PROBE_PROMPT_PROFILE", "chartqa_oracle_hint")
    monkeypatch.setenv("DYME_TEACHER_PROBE_MAX_NEW_TOKENS", "192")
    monkeypatch.setenv("DYME_TEACHER_TRAJ_MAX_NEW_TOKENS", "256")
    module = importlib.import_module("config.config_opd_7b_dyme_probe")
    module = importlib.reload(module)

    try:
        opsd = module.CONFIG["opsd"]
        assert opsd["privileged_providers"] == [
            "format_only",
            "oracle_hint",
            "visual_facts_deplot",
        ]
        assert opsd["teacher_probe"]["context_providers"] == [
            "format_only",
            "oracle_hint",
            "visual_facts_deplot",
        ]
        assert opsd["teacher_probe"]["prompt_profile"] == "chartqa_oracle_hint"
        assert opsd["teacher_probe"]["max_new_tokens"] == 192
        assert opsd["teacher_trajectory"]["max_new_tokens"] == 256
        assert opsd["text_include_gold"] is False
    finally:
        for name in (
            "DYME_TEACHER_PROBE_PROVIDERS",
            "DYME_TEACHER_PROBE_PROMPT_PROFILE",
            "DYME_TEACHER_PROBE_MAX_NEW_TOKENS",
            "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS",
        ):
            os.environ.pop(name, None)
        importlib.reload(module)


def test_teacher_sft_repair_env_reaches_config(monkeypatch) -> None:
    monkeypatch.setenv("DYME_TEACHER_CORRECT_REPAIR_MODE", "traj_sft")
    monkeypatch.setenv("DYME_TEACHER_SFT_REPAIR_SCOPE", "all_wrong")
    monkeypatch.setenv("DYME_TEACHER_SFT_REPAIR_SLOTS", "1")
    monkeypatch.setenv("DYME_TEACHER_SFT_TARGET_MAX_TOKENS", "256")
    monkeypatch.setenv("DYME_TEACHER_SFT_SANITIZE_PRIVILEGED", "1")
    monkeypatch.setenv("DYME_TEACHER_SFT_TARGET_CONSTRAINT", "chartqa_hint")
    monkeypatch.setenv("DYME_TEACHER_SFT_TARGET_STYLE", "student_short")
    module = importlib.import_module("config.config_opd_7b_dyme_probe")
    module = importlib.reload(module)

    try:
        cfg = module.CONFIG["opsd"]["teacher_correct_repair"]
        assert cfg["mode"] == "traj_sft"
        assert cfg["scope"] == "all_wrong"
        assert cfg["slots_per_prompt"] == 1
        assert cfg["target_max_tokens"] == 256
        assert cfg["sanitize_privileged"] is True
        assert cfg["target_constraint"] == "chartqa_hint"
        assert cfg["target_style"] == "student_short"
    finally:
        for name in (
            "DYME_TEACHER_CORRECT_REPAIR_MODE",
            "DYME_TEACHER_SFT_REPAIR_SCOPE",
            "DYME_TEACHER_SFT_REPAIR_SLOTS",
            "DYME_TEACHER_SFT_TARGET_MAX_TOKENS",
            "DYME_TEACHER_SFT_SANITIZE_PRIVILEGED",
            "DYME_TEACHER_SFT_TARGET_CONSTRAINT",
            "DYME_TEACHER_SFT_TARGET_STYLE",
        ):
            os.environ.pop(name, None)
        importlib.reload(module)


def test_opsd_decay_and_cap_env_reaches_config(monkeypatch) -> None:
    monkeypatch.setenv("DYME_OPSD_WEIGHT_DECAY", "1")
    monkeypatch.setenv("DYME_OPSD_DECAY_START_STEP", "294")
    monkeypatch.setenv("DYME_OPSD_DECAY_END_STEP", "441")
    monkeypatch.setenv("DYME_OPSD_FINAL_WEIGHT", "0.5")
    monkeypatch.setenv("DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP", "294")
    monkeypatch.setenv("DYME_OPSD_MAX_PER_PROMPT", "2")
    module = importlib.import_module("config.config_opd_7b_dyme_probe")
    module = importlib.reload(module)

    try:
        loss = module.CONFIG["opsd"]["loss"]
        assert loss["weight_decay"]["enabled"] is True
        assert loss["weight_decay"]["start_step"] == 294
        assert loss["weight_decay"]["end_step"] == 441
        assert loss["weight_decay"]["final_weight"] == 0.5
        assert loss["route_cap"]["enabled"] is True
        assert loss["route_cap"]["max_per_prompt"] == 2
        assert loss["route_cap"]["after_step"] == 294
        assert loss["route_cap"]["overflow_route"] == "sft"
    finally:
        for name in (
            "DYME_OPSD_WEIGHT_DECAY",
            "DYME_OPSD_DECAY_START_STEP",
            "DYME_OPSD_DECAY_END_STEP",
            "DYME_OPSD_FINAL_WEIGHT",
            "DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP",
            "DYME_OPSD_MAX_PER_PROMPT",
        ):
            os.environ.pop(name, None)
        importlib.reload(module)


def test_effective_sampling_env_reaches_config(monkeypatch) -> None:
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING", "1")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_AFTER_STEP", "294")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT", "4.5")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT", "1.1")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT", "0.6")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT", "1.0")
    monkeypatch.setenv("DYME_EFFECTIVE_SAMPLING_REWARD_STD_BONUS", "2.5")
    module = importlib.import_module("config.config_opd_7b_dyme_probe")
    module = importlib.reload(module)

    try:
        cfg = module.CONFIG["opsd"]["effective_sampling"]
        assert cfg["enabled"] is True
        assert cfg["after_step"] == 294
        assert cfg["mixed_weight"] == 4.5
        assert cfg["all_wrong_weight"] == 1.1
        assert cfg["all_correct_weight"] == 0.6
        assert cfg["unknown_weight"] == 1.0
        assert cfg["reward_std_bonus"] == 2.5
    finally:
        for name in (
            "DYME_EFFECTIVE_SAMPLING",
            "DYME_EFFECTIVE_SAMPLING_AFTER_STEP",
            "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT",
            "DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT",
            "DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT",
            "DYME_EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT",
            "DYME_EFFECTIVE_SAMPLING_REWARD_STD_BONUS",
        ):
            os.environ.pop(name, None)
        importlib.reload(module)
