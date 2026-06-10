import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.privileged.image_utils import resolve_teacher_images
from opsd_utils.privileged.profiles import effective_profile


def test_text_profile_single_image():
    img = Image.new("RGB", (64, 64))
    sample = {"image": img, "hint": "h", "answer": "Answer: 1"}
    images, meta = resolve_teacher_images(sample, "text")
    assert len(images) == 1
    assert meta["num_teacher_images"] == 1


def test_hybrid_profile_dual_image():
    img = Image.new("RGB", (64, 64))
    sample = {"image": img, "evidence_bbox": [0.1, 0.1, 0.9, 0.9]}
    images, meta = resolve_teacher_images(sample, "hybrid")
    assert len(images) == 2
    assert meta["has_bbox"] is True


def test_no_image_empty():
    sample = {"hint": "only text"}
    assert effective_profile(sample, "hybrid") == "text"
    images, meta = resolve_teacher_images(sample, "text")
    assert images == []
    assert meta["num_teacher_images"] == 0
