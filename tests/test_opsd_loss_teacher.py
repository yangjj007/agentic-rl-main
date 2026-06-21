"""Smoke test cross-model OPD hook in opsd_loss."""

import os
import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.vocab_align import align_cross_model_logits
from opsd_utils.opsd_loss import (
    _completion_tensors_for_opsd_step,
    _teacher_row,
    compute_vlm_opsd_loss_masked_batch,
    generalized_jsd_loss,
    token_distillation_loss,
)


def test_opsd_loss_accepts_teacher_model_kwarg():
    """Teacher model kwarg is wired; full forward is integration-tested on GPU."""
    batch_size = 1
    seq = 4
    vocab = 8
    device = torch.device("cpu")

    student = MagicMock()
    teacher = MagicMock()
    _param = torch.nn.Parameter(torch.zeros(1, device=device))

    def _parameters():
        return iter([_param])

    student.parameters = _parameters
    teacher.parameters = _parameters

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


def test_teacher_row_compact_indices():
    inputs = {"teacher_compact_indices": [0, 5, 12]}
    assert _teacher_row(inputs, 5) == 1
    assert _teacher_row(inputs, 0) == 0
    assert _teacher_row(inputs, 99) == 0


def test_dummy_opsd_completion_uses_single_token():
    inputs = {
        "completion_ids": torch.tensor([[10, 11, 12, 13]]),
        "completion_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    logits = torch.randn(1, 4, 8)
    comp_ids, comp_mask, pre = _completion_tensors_for_opsd_step(
        inputs, 0, is_real=False, student_completion_logits=logits
    )
    assert comp_ids.shape == (1, 1)
    assert comp_mask.shape == (1, 1)
    assert pre.shape == (1, 1, 8)


def test_generalized_jsd_loss_mismatched_vocab_sizes():
    student = torch.randn(1, 5, 152000, requires_grad=True)
    teacher = torch.randn(1, 5, 152128)
    mask = torch.ones(1, 5)
    s, t = align_cross_model_logits(student, teacher)
    assert s.shape[-1] == t.shape[-1] == 152000
    loss = generalized_jsd_loss(s, t, mask)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_token_distillation_loss_srkl_and_fkl():
    student = torch.randn(1, 4, 64, requires_grad=True)
    teacher = torch.randn(1, 4, 64)
    mask = torch.ones(1, 4)
    srkl = token_distillation_loss(student, teacher, mask, loss_type="srkl", srkl_alpha=0.1)
    fkl = token_distillation_loss(student, teacher, mask, loss_type="fkl")
    assert srkl.ndim == 0 and srkl.requires_grad
    assert fkl.ndim == 0 and fkl.requires_grad
