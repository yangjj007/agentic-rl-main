import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.teacher_batching import (
    get_teacher_vision_for_sample,
    split_tensor_dict_for_opsd,
    stack_teacher_processor_batches,
)


def test_split_tensor_dict_dual_image_chunks_legacy_stacked():
    """8-sample batch, 2 teacher images each -> 16 image rows; split into 4 GA chunks."""
    batch_size = 8
    teacher_num_images = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2])
    teacher_pixel_values = torch.arange(16 * 2).reshape(16, 2)
    teacher_prompt_ids = torch.zeros(batch_size, 10)

    chunks = split_tensor_dict_for_opsd(
        {
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_pixel_values": teacher_pixel_values,
            "teacher_num_images": teacher_num_images,
        },
        num_chunks=4,
    )
    assert len(chunks) == 4
    assert chunks[0]["teacher_prompt_ids"].shape[0] == 2
    assert chunks[0]["teacher_pixel_values"].shape[0] == 4
    assert chunks[1]["teacher_pixel_values"].shape[0] == 4
    assert chunks[0]["teacher_num_images"].tolist() == [2, 2]
    assert chunks[0]["teacher_pixel_values"][2:4].shape[0] == 2


def test_split_tensor_dict_vision_list_mixed_patches():
    """Per-sample vision lists preserve variable patch counts across GA split."""
    batch_size = 4
    pv_list = [
        torch.zeros(2, 7, 3, 4, 4),
        torch.zeros(2, 5, 3, 4, 4),
        torch.zeros(2, 7, 3, 4, 4),
        torch.zeros(2, 3, 3, 4, 4),
    ]
    chunks = split_tensor_dict_for_opsd(
        {
            "teacher_prompt_ids": torch.zeros(batch_size, 10),
            "teacher_pixel_values_list": pv_list,
            "teacher_num_images": torch.tensor([2, 2, 2, 2]),
        },
        num_chunks=2,
    )
    assert len(chunks) == 2
    assert len(chunks[0]["teacher_pixel_values_list"]) == 2
    assert chunks[0]["teacher_pixel_values_list"][0].shape == (2, 7, 3, 4, 4)
    assert chunks[0]["teacher_pixel_values_list"][1].shape == (2, 5, 3, 4, 4)
    assert chunks[1]["teacher_pixel_values_list"][1].shape == (2, 3, 3, 4, 4)


def test_get_teacher_vision_for_sample_from_list():
    inputs = {
        "prompt_ids": torch.zeros(2, 5),
        "teacher_pixel_values_list": [
            torch.zeros(2, 7, 3, 4, 4),
            torch.zeros(2, 5, 3, 4, 4),
        ],
        "teacher_image_sizes_list": [
            torch.tensor([[800, 600], [400, 300]]),
            torch.tensor([[640, 480], [320, 240]]),
        ],
        "teacher_num_images": torch.tensor([2, 2]),
    }
    pv, sz = get_teacher_vision_for_sample(inputs, 1, [2, 2])
    assert pv.shape == (2, 5, 3, 4, 4)
    assert sz.shape == (2, 2)


def test_stack_teacher_processor_batches_keeps_per_sample_pixels():
    per_sample = [
        {
            "input_ids": torch.zeros(1, 5),
            "attention_mask": torch.ones(1, 5),
            "pixel_values": torch.zeros(2, 7, 3, 4, 4),
        },
        {
            "input_ids": torch.zeros(1, 7),
            "attention_mask": torch.ones(1, 7),
            "pixel_values": torch.zeros(2, 5, 3, 4, 4),
        },
    ]

    class _Tok:
        pad_token_id = 0

    class _Proc:
        tokenizer = _Tok()

    out = stack_teacher_processor_batches(_Proc(), per_sample)
    assert out["input_ids"].shape == (2, 7)
    assert len(out["pixel_values_list"]) == 2
    assert out["pixel_values_list"][0].shape == (2, 7, 3, 4, 4)
    assert out["pixel_values_list"][1].shape == (2, 5, 3, 4, 4)
    assert out["batch_num_images"] == [2, 2]
