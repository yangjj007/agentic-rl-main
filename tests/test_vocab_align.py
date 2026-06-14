"""Tests for cross-model vocab alignment diagnostics."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.opsd_loss import generalized_jsd_loss
from opsd_utils.vocab_align import (
    align_cross_model_logits,
    reset_vocab_align_debug,
    verify_shared_tokenizer_alignment,
)


class _Tok:
    def __init__(self, vocab_size: int, offset: int = 0):
        self._size = vocab_size
        self._offset = offset

    def __len__(self):
        return self._size

    def decode(self, ids, skip_special_tokens=False):
        i = ids[0]
        return f"tok_{i + self._offset}"

    def convert_ids_to_tokens(self, i):
        return f"tok_{i + self._offset}"


def test_align_slice_renorm_via_log_softmax():
    reset_vocab_align_debug()
    student = torch.randn(1, 3, 100, requires_grad=True)
    teacher = torch.randn(1, 3, 128)
    s, t = align_cross_model_logits(student, teacher, log_renorm_check=False)
    assert s.shape[-1] == t.shape[-1] == 100
    t_probs = F.softmax(t[0, 0], dim=-1)
    assert abs(float(t_probs.sum()) - 1.0) < 1e-4


def test_generalized_jsd_renormalizes_after_slice():
    reset_vocab_align_debug()
    student = torch.randn(1, 5, 152000, requires_grad=True)
    teacher = torch.randn(1, 5, 152128)
    mask = torch.ones(1, 5)
    loss = generalized_jsd_loss(student, teacher, mask)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_tokenizer_alignment_detects_mismatch():
    st = _Tok(200, offset=0)
    tt = _Tok(200, offset=1)
    report = verify_shared_tokenizer_alignment(
        st, tt, shared_vocab=200, full_scan=True, sample_stride=1
    )
    assert not report["aligned"]
    assert report["mismatch_count"] > 0


def test_tokenizer_alignment_passes_identical():
    st = _Tok(1000, offset=0)
    tt = _Tok(1200, offset=0)
    report = verify_shared_tokenizer_alignment(
        st, tt, shared_vocab=1000, full_scan=False, sample_stride=100
    )
    assert report["aligned"]
