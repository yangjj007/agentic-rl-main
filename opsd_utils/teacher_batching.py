"""Helpers for LLaVA-OV teacher batches (per-sample vision tensors)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug

# Legacy stacked tensors (dim0 = total images). Prefer *_list keys.
TEACHER_IMAGE_STACKED_KEYS = frozenset({"teacher_pixel_values", "teacher_image_sizes"})
TEACHER_VISION_LIST_KEYS = frozenset({"teacher_pixel_values_list", "teacher_image_sizes_list"})


def batch_size_from_tensor_dict(tensor_dict: dict[str, Any]) -> int:
    for key in ("prompt_ids", "teacher_prompt_ids", "completion_ids"):
        tensor = tensor_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    for value in tensor_dict.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, list) and value:
            return len(value)
    return 0


def teacher_image_counts_from_dict(
    tensor_dict: dict[str, Any],
    batch_size: int,
) -> list[int]:
    counts = tensor_dict.get("teacher_num_images")
    if counts is None:
        return [1] * batch_size
    if isinstance(counts, torch.Tensor):
        return [int(max(1, c)) for c in counts.detach().cpu().tolist()]
    return [int(max(1, c)) for c in counts]


def image_offsets(counts: list[int]) -> list[int]:
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return offsets


def split_tensor_dict_for_opsd(
    tensor_dict: dict[str, Any],
    num_chunks: int,
) -> list[dict[str, Any]]:
    """
    Split batch tensors for gradient accumulation.

    Teacher vision is stored per-sample in *_list keys to preserve
    input_ids <-> pixel_values alignment (variable patch counts).
    """
    batch_size = batch_size_from_tensor_dict(tensor_dict)
    if batch_size == 0 or num_chunks <= 0:
        return [dict(tensor_dict)]
    if batch_size % num_chunks != 0:
        opsd_debug.log(
            "teacher_batching",
            "split_tensor_dict_for_opsd uneven batch",
            batch_size=batch_size,
            num_chunks=num_chunks,
        )
    chunk_batch = max(1, batch_size // num_chunks)
    img_counts = teacher_image_counts_from_dict(tensor_dict, batch_size)
    img_offs = image_offsets(img_counts)

    chunks: list[dict[str, Any]] = []
    for i in range(num_chunks):
        b0 = i * chunk_batch
        b1 = min((i + 1) * chunk_batch, batch_size)
        if b0 >= batch_size:
            break
        img0, img1 = img_offs[b0], img_offs[b1]
        chunk: dict[str, Any] = {}
        for key, value in tensor_dict.items():
            if value is None:
                chunk[key] = None
            elif key in TEACHER_VISION_LIST_KEYS:
                chunk[key] = value[b0:b1]
            elif key in TEACHER_IMAGE_STACKED_KEYS:
                chunk[key] = value[img0:img1]
            elif isinstance(value, torch.Tensor):
                chunk[key] = value[b0:b1]
            else:
                chunk[key] = value
        chunks.append(chunk)

    opsd_debug.log(
        "teacher_batching",
        "split_tensor_dict_for_opsd",
        batch_size=batch_size,
        num_chunks=num_chunks,
        chunk_batch=chunk_batch,
        teacher_num_images=img_counts,
        image_offsets=img_offs,
        uses_vision_lists=tensor_dict.get("teacher_pixel_values_list") is not None,
        output_chunks=len(chunks),
    )
    return chunks


def stack_teacher_processor_batches(
    processor,
    per_sample_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad input_ids per sample; keep pixel_values as per-sample list (no cross-sample cat)."""
    if not per_sample_batches:
        return {}

    pad_id = processor.tokenizer.pad_token_id
    max_len = max(int(b["input_ids"].shape[1]) for b in per_sample_batches)

    input_ids_list: list[torch.Tensor] = []
    attn_list: list[torch.Tensor] = []
    pixel_list: list[torch.Tensor] = []
    size_list: list[torch.Tensor] = []
    batch_num_images: list[int] = []

    for batch in per_sample_batches:
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=pad_id)
            attn = F.pad(attn, (0, pad_len), value=0)
        input_ids_list.append(ids)
        attn_list.append(attn)

        if "pixel_values" in batch:
            pv = batch["pixel_values"]
            pixel_list.append(pv)
            batch_num_images.append(int(pv.shape[0]))
        else:
            batch_num_images.append(1)
        if "image_sizes" in batch:
            size_list.append(batch["image_sizes"])

    patch_counts = [int(pv.shape[1]) for pv in pixel_list] if pixel_list else []
    if len(set(patch_counts)) > 1:
        opsd_debug.log(
            "teacher_batching",
            "per-sample teacher patch counts (kept aligned via list storage)",
            patch_counts_per_sample=patch_counts,
            num_samples=len(per_sample_batches),
        )

    out: dict[str, Any] = {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "attention_mask": torch.cat(attn_list, dim=0),
        "batch_num_images": batch_num_images,
        "pixel_values_list": pixel_list,
        "image_sizes_list": size_list,
    }
    return out


def process_teacher_sample(processor, text: str, images: list[Any]) -> dict[str, Any]:
    """Tokenize one teacher sample (supports multi-image)."""
    if images:
        return processor(text=[text], images=images, return_tensors="pt", padding=True)
    return processor(text=[text], return_tensors="pt", padding=True)


def get_teacher_vision_for_sample(
    inputs: dict[str, Any],
    local: int,
    num_images_per_sample: Optional[list[int]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (pixel_values, image_sizes) for one batch row, aligned with teacher_prompt_ids."""
    pv_list = inputs.get("teacher_pixel_values_list")
    if pv_list is not None:
        if local >= len(pv_list):
            return None, None
        t_pixel = pv_list[local]
        sizes_list = inputs.get("teacher_image_sizes_list") or []
        t_sizes = sizes_list[local] if local < len(sizes_list) else None
        return t_pixel, t_sizes

    # Legacy stacked layout
    from opsd_utils.opsd_loss import slice_teacher_vision_inputs

    batch_size = inputs["prompt_ids"].shape[0]
    counts = num_images_per_sample or teacher_image_counts_from_dict(inputs, batch_size)
    return slice_teacher_vision_inputs(
        inputs.get("teacher_pixel_values"),
        inputs.get("teacher_image_sizes"),
        local,
        counts,
    )
