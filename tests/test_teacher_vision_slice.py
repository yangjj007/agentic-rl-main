import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.opsd_loss import slice_teacher_vision_inputs


def test_slice_dual_image_batch():
    # batch=2, each sample has 2 images, 7 patches each (LLaVA-OV layout)
    pixel_values = torch.zeros(4, 7, 3, 384, 384)
    image_sizes = torch.tensor([[800, 600], [400, 300], [800, 600], [400, 300]])
    counts = [2, 2]

    p0, s0 = slice_teacher_vision_inputs(pixel_values, image_sizes, 0, counts)
    p1, s1 = slice_teacher_vision_inputs(pixel_values, image_sizes, 1, counts)

    assert p0.shape == (2, 7, 3, 384, 384)
    assert p1.shape == (2, 7, 3, 384, 384)
    assert s0.shape == (2, 2)
    assert s1.shape == (2, 2)


def test_slice_mixed_image_counts():
    pixel_values = torch.zeros(3, 7, 3, 384, 384)
    image_sizes = torch.tensor([[800, 600], [800, 600], [400, 300]])
    counts = [1, 2]

    p0, s0 = slice_teacher_vision_inputs(pixel_values, image_sizes, 0, counts)
    p1, s1 = slice_teacher_vision_inputs(pixel_values, image_sizes, 1, counts)

    assert p0.shape == (1, 7, 3, 384, 384)
    assert p1.shape == (2, 7, 3, 384, 384)
    assert s0.shape == (1, 2)
    assert s1.shape == (2, 2)
