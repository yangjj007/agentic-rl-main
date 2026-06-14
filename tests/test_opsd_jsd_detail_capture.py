"""OPSD-DETAIL JSD diagnostics must not run extra model forwards."""

import os
import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils import debug_log as opsd_debug
from opsd_utils import diagnostics as opsd_diagnostics
from opsd_utils.opsd_loss import compute_vlm_opsd_loss_masked_batch


def test_jsd_detail_reuses_loss_logits_no_extra_forward(monkeypatch):
    opsd_debug.configure(enabled=True, detail_every=10, rank=0, world_size=1)

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

    student.reset_mock()
    teacher.reset_mock()

    loss = compute_vlm_opsd_loss_masked_batch(
        student,
        [0],
        [0],
        inputs,
        beta=0.5,
        processor=None,
        teacher_model=teacher,
        acc_gate=True,
        global_step=20,
        tokenizer=None,
    )
    assert isinstance(loss, torch.Tensor)
    assert student.call_count == 1
    assert teacher.call_count == 1

    opsd_diagnostics.log_opsd_jsd_diagnostics(global_step=20)
    assert student.call_count == 1
    assert teacher.call_count == 1


def test_jsd_detail_memory_guard_skips_capture(monkeypatch):
    opsd_debug.configure(enabled=True, detail_every=10, rank=0, world_size=1)
    monkeypatch.setattr(opsd_diagnostics, "check_detail_cuda_memory", lambda **_: (False, "mock_low_mem", 0.5))

    opsd_diagnostics.begin_opsd_jsd_detail_capture(10, [0], max_samples=1)
    assert opsd_diagnostics._OPSD_JSD_DETAIL_CAPTURE["skipped_memory"] is True
