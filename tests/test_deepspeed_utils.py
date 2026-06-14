"""Smoke tests for DeepSpeed accelerate config detection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.deepspeed_utils import (
    deepspeed_zero_stage,
    gradient_checkpointing_enable_kwargs,
    is_deepspeed_accelerate_config,
    should_colocate_teacher_with_student,
    uses_deepspeed_json_file,
)


def test_zero2_config_detected(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config_zero2.yaml")
    assert is_deepspeed_accelerate_config()
    assert uses_deepspeed_json_file()
    assert deepspeed_zero_stage() == 2
    assert should_colocate_teacher_with_student("auto")
    assert gradient_checkpointing_enable_kwargs() == {"use_reentrant": False}


def test_ddp_config_not_deepspeed(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config.yaml")
    assert not is_deepspeed_accelerate_config()
    assert not should_colocate_teacher_with_student("auto")
