"""Tests for batch data health diagnostics."""
import torch

from opsd_utils.diagnostics import (
    _detect_char_repeat,
    summarize_batch_data_health,
)


def test_detect_char_repeat_cjk():
    assert _detect_char_repeat("Goal: " + "其" * 10)


def test_summarize_batch_data_health_empty_vf():
    samples = [
        {"prompt": "q1", "visual_fact_hint": ""},
        {"prompt": "q2", "visual_fact_hint": "bar value 3"},
    ]
    stats = summarize_batch_data_health(samples)
    assert stats["visual_fact_empty_rate"] == 0.5
    assert stats["batch_size"] == 2


def test_summarize_batch_data_health_pixel_nan():
    samples = [{"prompt": "q", "visual_fact_hint": "x"}]
    pixel = torch.tensor([float("nan"), 1.0, 2.0])
    stats = summarize_batch_data_health(samples, pixel_values=pixel)
    assert stats["pixel_has_nan"] is True
