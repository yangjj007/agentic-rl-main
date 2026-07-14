"""Smoke test cross-model OPD hook in opsd_loss."""

import os
import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.vocab_align import align_cross_model_logits
from opsd_utils.opsd_loss import (
    _build_token_reliability_mask,
    _combine_grpo_opsd_losses,
    _teacher_row,
    _trim_to_effective_completion,
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


def test_trim_to_effective_completion_drops_padding():
    comp_ids = torch.tensor([[10, 11, 12, 0, 0]])
    comp_mask = torch.tensor([[1, 1, 1, 0, 0]])
    logits = torch.randn(1, 5, 8)
    ids, mask, lg, eff = _trim_to_effective_completion(comp_ids, comp_mask, logits)
    assert eff == 3
    assert ids.shape == (1, 3)
    assert mask.shape == (1, 3)
    assert lg.shape == (1, 3, 8)


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


def test_token_reliability_mask_upweights_numeric_and_answer_tokens():
    class ToyTokenizer:
        pieces = {
            1: "The",
            2: " answer",
            3: " is",
            4: " 42",
            5: "%",
            6: ".",
        }

        def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
            return self.pieces[int(ids[0])]

    completion_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    completion_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    weights = _build_token_reliability_mask(
        completion_ids,
        completion_mask,
        tokenizer=ToyTokenizer(),
        token_weighting={
            "enabled": True,
            "numeric_weight": 2.0,
            "answer_weight": 1.5,
            "min_weight": 0.75,
        },
    )

    assert weights.tolist() == [[0.75, 1.5, 0.75, 2.0, 2.0, 0.0]]


def test_answer_anchor_token_mask_focuses_after_answer_marker():
    class ToyTokenizer:
        pieces = {
            1: " 12",
            2: " reasoning",
            3: " answer",
            4: " is",
            5: " 34",
            6: ".",
            7: "<pad>",
        }

        def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
            return self.pieces[int(ids[0])]

    weights = _build_token_reliability_mask(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
        tokenizer=ToyTokenizer(),
        token_weighting={
            "enabled": True,
            "mode": "answer_anchor",
            "numeric_weight": 3.0,
            "answer_weight": 2.0,
            "min_weight": 0.05,
        },
    )

    assert torch.allclose(
        weights,
        torch.tensor([[0.05, 0.05, 2.0, 2.0, 3.0, 2.0, 0.0]]),
    )


def test_loss_mixer_applies_grpo_weight_even_without_opsd_samples():
    grpo_loss = torch.tensor(3.0)

    mixed = _combine_grpo_opsd_losses(
        grpo_loss,
        grpo_weight=0.0,
        opsd_loss=None,
        opsd_weight=1.5,
    )

    assert mixed.item() == 0.0


def test_loss_mixer_adds_weighted_opsd_when_present():
    grpo_loss = torch.tensor(3.0)
    opsd_loss = torch.tensor(2.0)

    mixed = _combine_grpo_opsd_losses(
        grpo_loss,
        grpo_weight=0.25,
        opsd_loss=opsd_loss,
        opsd_weight=1.5,
    )

    assert mixed.item() == 3.75
