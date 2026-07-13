"""Tests for completion logit slicing shared by GRPO and OPSD."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.opsd_loss import slice_student_completion_logits


def test_slice_matches_legacy_grpo_path():
    logits_to_keep = 4
    full = torch.randn(2, 20, 8)
    legacy = full[:, -logits_to_keep - 1 :, :]
    legacy = legacy[:, :-1, :]
    legacy = legacy[:, -logits_to_keep:, :]
    assert torch.allclose(legacy, slice_student_completion_logits(full, logits_to_keep))
