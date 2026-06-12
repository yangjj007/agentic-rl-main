"""Smoke test cross-model OPD hook in opsd_loss."""

import os
import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.opsd_loss import compute_vlm_opsd_loss_masked_batch


def test_opsd_loss_accepts_teacher_model_kwarg():
    """Teacher model kwarg is wired; full forward is integration-tested on GPU."""
    batch_size = 1
    seq = 4
    vocab = 8
    device = torch.device("cpu")

    student = MagicMock()
    teacher = MagicMock()

    def _fake_forward(**kwargs):
        logits = torch.zeros(1, seq + 2, vocab, device=device)
        out = MagicMock()
        out.logits = logits
        return out

    student.side_effect = _fake_forward
    teacher.side_effect = _fake_forward

    inputs = {
        "prompt_ids": torch.ones(batch_size, seq, dtype=torch.long),
        "prompt_mask": torch.ones(batch_size, seq, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 8, 8),
        "teacher_prompt_ids": torch.ones(batch_size, seq, dtype=torch.long),
        "teacher_prompt_mask": torch.ones(batch_size, seq, dtype=torch.long),
        "completion_ids": torch.ones(batch_size, 2, dtype=torch.long),
        "completion_mask": torch.ones(batch_size, 2, dtype=torch.long),
        "acc_rewards": torch.tensor([0.0]),
        "teacher_num_images": torch.tensor([1], dtype=torch.long),
    }

    loss = compute_vlm_opsd_loss_masked_batch(
        student,
        [0],
        [0],
        inputs,
        beta=0.5,
        processor=None,
        teacher_model=teacher,
        acc_gate=True,
    )
    assert isinstance(loss, torch.Tensor)
    assert teacher.called, "cross-model OPD must forward through teacher_model"
    assert student.called, "OPSD must forward through student model"
