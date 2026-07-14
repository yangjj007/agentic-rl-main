"""Tests for batch data health diagnostics."""
import torch

from opsd_utils.diagnostics import (
    _detect_char_repeat,
    summarize_batch_data_health,
)
from data_utils.chart.deplot_pipeline import build_deplot_visual_fact


def test_detect_char_repeat_cjk():
    assert _detect_char_repeat("Goal: " + "其" * 10)


def test_summarize_batch_data_health_empty_vf():
    samples = [
        {"prompt": "q1", "visual_fact_hint": "Goal: leaked hint\nAnswer: 3"},
        {
            "prompt": "q2",
            "visual_fact_deplot": build_deplot_visual_fact(
                {"question": "q2"}, "Label | Value\nA | 3"
            ),
        },
    ]
    stats = summarize_batch_data_health(samples)
    assert stats["visual_fact_empty_rate"] == 0.5
    assert stats["batch_size"] == 2


def test_summarize_batch_data_health_pixel_nan():
    samples = [{"prompt": "q", "visual_fact": {"objects": []}}]
    pixel = torch.tensor([float("nan"), 1.0, 2.0])
    stats = summarize_batch_data_health(samples, pixel_values=pixel)
    assert stats["pixel_has_nan"] is True
