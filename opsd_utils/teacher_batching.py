"""Helpers for LLaVA-OV teacher batches (image-stacked pixel_values)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug

TEACHER_IMAGE_STACKED_KEYS = frozenset({"teacher_pixel_values", "teacher_image_sizes"})


def batch_size_from_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> int:
    for key in ("prompt_ids", "teacher_prompt_ids", "completion_ids"):
        tensor = tensor_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    for tensor in tensor_dict.values():
        if tensor is not None:
            return int(tensor.shape[0])
    return 0


def teacher_image_counts_from_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]],
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
    tensor_dict: dict[str, Optional[torch.Tensor]],
    num_chunks: int,
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Split batch tensors for gradient accumulation.

    Teacher vision tensors use dim-0 = total images (not batch size), so slice by
    cumulative teacher_num_images offsets instead of batch chunk_size.
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

    chunks: list[dict[str, Optional[torch.Tensor]]] = []
    for i in range(num_chunks):
        b0 = i * chunk_batch
        b1 = min((i + 1) * chunk_batch, batch_size)
        if b0 >= batch_size:
            break
        img0, img1 = img_offs[b0], img_offs[b1]
        chunk: dict[str, Optional[torch.Tensor]] = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                chunk[key] = None
            elif key in TEACHER_IMAGE_STACKED_KEYS:
                chunk[key] = tensor[img0:img1]
            else:
                chunk[key] = tensor[b0:b1]
        chunks.append(chunk)

    opsd_debug.log(
        "teacher_batching",
        "split_tensor_dict_for_opsd",
        batch_size=batch_size,
        num_chunks=num_chunks,
        chunk_batch=chunk_batch,
        teacher_num_images=img_counts,
        image_offsets=img_offs,
        output_chunks=len(chunks),
    )
    return chunks


def _max_patch_count(pixel_parts: list[torch.Tensor]) -> int:
    return max(int(pv.shape[1]) for pv in pixel_parts)


def _pad_pixel_values_patch_dim(pixel_values: torch.Tensor, target_patches: int) -> torch.Tensor:
    """Pad LLaVA-OV pixel_values (N, P, C, H, W) along patch dim to target_patches."""
    cur = int(pixel_values.shape[1])
    if cur >= target_patches:
        return pixel_values
    pad_p = target_patches - cur
    # F.pad last dims first: W, H, C, P, N
    return F.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, pad_p))


def stack_teacher_processor_batches(
    processor,
    per_sample_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad input_ids per sample and concat image-stacked pixel_values."""
    if not per_sample_batches:
        return {}

    pad_id = processor.tokenizer.pad_token_id
    max_len = max(int(b["input_ids"].shape[1]) for b in per_sample_batches)

    input_ids_list: list[torch.Tensor] = []
    attn_list: list[torch.Tensor] = []
    pixel_parts: list[torch.Tensor] = []
    size_parts: list[torch.Tensor] = []
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
            pixel_parts.append(pv)
            batch_num_images.append(int(pv.shape[0]))
        else:
            batch_num_images.append(1)
        if "image_sizes" in batch:
            size_parts.append(batch["image_sizes"])

    out: dict[str, Any] = {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "attention_mask": torch.cat(attn_list, dim=0),
        "batch_num_images": batch_num_images,
    }
    if pixel_parts:
        max_patches = _max_patch_count(pixel_parts)
        padded_parts = [_pad_pixel_values_patch_dim(pv, max_patches) for pv in pixel_parts]
        patch_counts = [int(pv.shape[1]) for pv in pixel_parts]
        if len(set(patch_counts)) > 1:
            opsd_debug.log(
                "teacher_batching",
                "pad teacher pixel_values patch dim before concat",
                patch_counts_per_sample=patch_counts,
                max_patches=max_patches,
                num_samples=len(pixel_parts),
            )
        out["pixel_values"] = torch.cat(padded_parts, dim=0)
    if size_parts:
        out["image_sizes"] = torch.cat(size_parts, dim=0)
    return out


def process_teacher_sample(processor, text: str, images: list[Any]) -> dict[str, Any]:
    """Tokenize one teacher sample (supports multi-image)."""
    if images:
        return processor(text=[text], images=images, return_tensors="pt", padding=True)
    return processor(text=[text], return_tensors="pt", padding=True)
