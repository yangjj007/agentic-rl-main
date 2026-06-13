import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.teacher_batching import (
    _image_feature_row_count,
    align_teacher_prompt_image_tokens,
    as_batch_num_images_tensor,
    expected_image_feature_count,
    get_teacher_vision_for_sample,
    split_tensor_dict_for_opsd,
    stack_teacher_processor_batches,
    student_batch_num_images_tensor,
    truncate_image_tokens,
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


def test_as_batch_num_images_tensor_shape():
    pv = torch.zeros(2, 1, 3, 384, 384)
    bn = as_batch_num_images_tensor(2, pv)
    assert bn is not None
    assert bn.shape == (1,)
    assert bn.tolist() == [2]
    assert bn.dtype == torch.long


def test_as_batch_num_images_tensor_none_cases():
    assert as_batch_num_images_tensor(2, None) is None
    assert as_batch_num_images_tensor(None, torch.zeros(1)) is None


def test_student_batch_num_images_tensor_collator_layout():
    """Processor-batched student pixels: dim0 is batch size, one image per row."""
    pv = torch.zeros(4, 7, 3, 384, 384)
    bn = student_batch_num_images_tensor(pv, batch_rows=4)
    assert bn is not None
    assert bn.tolist() == [1, 1, 1, 1]


def test_student_batch_num_images_tensor_stacked_images():
    """Per-sample vision tensor with multiple images (dim0 = num images)."""
    pv = torch.zeros(2, 7, 3, 384, 384)
    bn = student_batch_num_images_tensor(pv, batch_rows=1)
    assert bn is not None
    assert bn.tolist() == [2]


def test_image_feature_row_count_list_return():
    feats = [torch.zeros(3, 64), torch.zeros(7, 64)]
    assert _image_feature_row_count(feats) == 10


def test_image_feature_row_count_pooler_output():
    out = type("Out", (), {"pooler_output": torch.zeros(5, 64)})()
    assert _image_feature_row_count(out) == 5


def test_expected_image_feature_count_passes_batch_num_images():
    captured: dict = {}

    class _Core:
        config = type(
            "C",
            (),
            {
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "full",
                "vision_aspect_ratio": "anyres_max_9",
            },
        )()

        def get_image_features(self, pixel_values, image_sizes, **kwargs):
            captured["batch_num_images"] = kwargs.get("batch_num_images")
            captured["return_dict"] = kwargs.get("return_dict")
            return [torch.zeros(6, 64), torch.zeros(4, 64)]

    class _Model:
        model = _Core()

    pv = torch.zeros(2, 1, 3, 4, 4)
    sizes = torch.tensor([[800, 600], [400, 300]])
    bn = as_batch_num_images_tensor(2, pv)
    count = expected_image_feature_count(_Model(), pv, sizes, batch_num_images=bn)
    assert count == 10
    assert captured["batch_num_images"].tolist() == [2]
    assert captured.get("return_dict") is None


def test_align_passes_batch_num_images():
    captured: dict = {}

    class _Core:
        config = type(
            "C",
            (),
            {
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "full",
                "vision_aspect_ratio": "anyres_max_9",
            },
        )()

        def get_image_features(self, pixel_values, image_sizes, **kwargs):
            captured["batch_num_images"] = kwargs.get("batch_num_images")
            return [torch.zeros(4, 64)]

    class _Model:
        model = _Core()

    class _Tok:
        pad_token_id = 0
        image_token_id = 151646

    class _Proc:
        tokenizer = _Tok()
        image_token = "<image>"

    img_id = 151646
    ids = torch.tensor([[img_id] * 4 + [1, 2]])
    mask = torch.ones(1, 6, dtype=torch.long)
    pv = torch.zeros(2, 1, 3, 4, 4)
    sizes = torch.tensor([[800, 600], [400, 300]])
    bn = as_batch_num_images_tensor(2, pv)
    out_ids, out_mask = align_teacher_prompt_image_tokens(
        _Model(),
        _Proc(),
        ids,
        mask,
        pv,
        sizes,
        batch_num_images=bn,
    )
    assert captured["batch_num_images"].tolist() == [2]
    assert out_ids.shape == ids.shape
    assert out_mask.shape == mask.shape


def test_truncate_image_tokens_keeps_first_n():
    img_id = 151646
    pad_id = 0
    ids = torch.tensor([[img_id] * 10 + [1, 2, 3]])
    mask = torch.ones(1, 13, dtype=torch.long)
    out_ids, out_mask = truncate_image_tokens(ids, mask, img_id, 4, pad_id)
    assert int((out_ids == img_id).sum()) == 4
    assert out_ids.shape[1] == 7
    assert int(out_mask.sum()) == 7
