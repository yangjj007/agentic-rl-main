from __future__ import annotations

from types import SimpleNamespace

import torch

from trainer.DyMETrainer import DyMETrainer


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    image_token_id = 151646


class _Processor:
    tokenizer = _Tokenizer()

    def batch_decode(self, token_ids, skip_special_tokens: bool = True):
        return [" ".join(str(int(tok)) for tok in row.tolist()) for row in token_ids]


class _Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.calls = 0
        self.batch_sizes: list[int] = []

    def get_input_embeddings(self):
        return self.embed

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.calls += 1
        self.batch_sizes.append(int(input_ids.shape[0]))
        suffix = torch.full(
            (input_ids.shape[0], 2),
            5,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix], dim=1)


def _trainer_with_teacher(teacher: _Teacher) -> DyMETrainer:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer.teacher_model = teacher
    trainer.processing_class = _Processor()
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer._perf_timing_enabled = False
    return trainer


def test_teacher_generate_batch_uses_one_generate_for_stackable_rows():
    teacher = _Teacher()
    trainer = _trainer_with_teacher(teacher)
    teacher_tensors = {
        "teacher_prompt_ids": torch.tensor([[1, 3, 4], [1, 6, 7]]),
        "teacher_prompt_mask": torch.ones(2, 3, dtype=torch.long),
    }

    outputs, fallback = trainer._teacher_generate_batch_from_tensors(
        teacher_tensors,
        [0, 1],
        max_new_tokens=2,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert fallback is False
    assert teacher.calls == 1
    assert teacher.batch_sizes == [2]
    assert len(outputs) == 2
    assert [int(mask.sum().item()) for _ids, mask, _text in outputs] == [2, 2]


def test_teacher_generate_batch_falls_back_for_mixed_patch_shapes():
    teacher = _Teacher()
    trainer = _trainer_with_teacher(teacher)
    teacher_tensors = {
        "teacher_prompt_ids": torch.tensor([[1, 3, 4], [1, 6, 7]]),
        "teacher_prompt_mask": torch.ones(2, 3, dtype=torch.long),
        "teacher_pixel_values_list": [
            torch.zeros(1, 7, 3, 4, 4),
            torch.zeros(1, 5, 3, 4, 4),
        ],
        "teacher_image_sizes_list": [
            torch.tensor([[800, 600]]),
            torch.tensor([[640, 480]]),
        ],
        "teacher_num_images": torch.tensor([1, 1]),
    }

    outputs, fallback = trainer._teacher_generate_batch_from_tensors(
        teacher_tensors,
        [0, 1],
        max_new_tokens=2,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert fallback is True
    assert teacher.calls == 2
    assert teacher.batch_sizes == [1, 1]
    assert len(outputs) == 2
