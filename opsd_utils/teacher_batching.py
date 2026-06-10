"""Helpers for LLaVA-OV teacher batches (per-sample vision tensors)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from opsd_utils import debug_log as opsd_debug

# Legacy stacked tensors (dim0 = total images). Prefer *_list keys.
TEACHER_IMAGE_STACKED_KEYS = frozenset({"teacher_pixel_values", "teacher_image_sizes"})
TEACHER_VISION_LIST_KEYS = frozenset({"teacher_pixel_values_list", "teacher_image_sizes_list"})


def image_token_id(processor) -> int:
    tok = processor.tokenizer
    if getattr(tok, "image_token_id", None) is not None:
        return int(tok.image_token_id)
    if hasattr(processor, "image_token_id"):
        return int(processor.image_token_id)
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is not None:
        return int(convert(getattr(processor, "image_token", "<image>")))
    return 151646


def count_image_tokens(input_ids: torch.Tensor, processor) -> int:
    img_id = image_token_id(processor)
    return int((input_ids == img_id).sum().item())


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
    image_token_counts: list[int] = []

    for batch in per_sample_batches:
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=pad_id)
            attn = F.pad(attn, (0, pad_len), value=0)
        input_ids_list.append(ids)
        attn_list.append(attn)
        image_token_counts.append(count_image_tokens(ids, processor))

        if "pixel_values" in batch:
            pv = batch["pixel_values"]
            pixel_list.append(pv)
            batch_num_images.append(int(pv.shape[0]))
        else:
            batch_num_images.append(0)
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
        "image_token_counts": image_token_counts,
    }
    return out


def _messages_for_teacher(teacher_text: str, images: list[Image.Image]) -> list[dict]:
    """Build chat messages with PIL images embedded (single processor tokenize path)."""
    content: list[dict] = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": teacher_text})
    return [{"role": "user", "content": content}]


def process_teacher_sample(processor, teacher_text: str, images: list[Any]) -> dict[str, Any]:
    """Tokenize one teacher sample via processor.apply_chat_template(tokenize=True)."""
    pil_images = [img for img in images if isinstance(img, Image.Image)]
    messages = _messages_for_teacher(teacher_text, pil_images)
    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    n_img_tok = count_image_tokens(batch["input_ids"], processor)
    opsd_debug.log(
        "teacher_batching",
        "process_teacher_sample",
        num_images=len(pil_images),
        input_ids_shape=tuple(batch["input_ids"].shape),
        pixel_values_shape=tuple(batch["pixel_values"].shape) if "pixel_values" in batch else None,
        image_token_count=n_img_tok,
    )
    return batch


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def as_batch_num_images_tensor(
    num_images: int | None,
    pixel_values: Optional[torch.Tensor],
    batch_rows: int = 1,
) -> Optional[torch.Tensor]:
    """Build batch_num_images for LLaVA-OV (per-sample image count in each batch row)."""
    if pixel_values is None or num_images is None:
        return None
    n = int(max(1, num_images))
    device = pixel_values.device
    return torch.tensor([n] * batch_rows, device=device, dtype=torch.long)


def _image_feature_row_count(result) -> int:
    """Total vision placeholder rows from LLaVA-OV get_image_features return value."""
    if hasattr(result, "pooler_output") and result.pooler_output is not None:
        packed = result.pooler_output
    else:
        packed = result
    if isinstance(packed, (list, tuple)):
        if not packed:
            return 0
        return int(torch.cat(packed, dim=0).shape[0])
    if isinstance(packed, torch.Tensor):
        return int(packed.shape[0])
    return 0


@torch.no_grad()
def expected_image_feature_count(
    model,
    pixel_values,
    image_sizes,
    batch_num_images: Optional[torch.Tensor] = None,
) -> int:
    """Vision feature rows after LLaVA-OV pack (matches model forward placeholder count)."""
    if pixel_values is None:
        return 0
    inner = _unwrap_model(model)
    if not hasattr(inner, "model"):
        return 0
    core = inner.model
    vision_feature_layer = getattr(core.config, "vision_feature_layer", -1)
    vision_feature_select_strategy = getattr(core.config, "vision_feature_select_strategy", "full")
    vision_aspect_ratio = getattr(core.config, "vision_aspect_ratio", "anyres_max_9")
    # transformers 4.57.x: returns list[Tensor] (already packed per image); no return_dict kwarg.
    # transformers 5.x: may return BaseModelOutputWithPooling with pooler_output.
    result = core.get_image_features(
        pixel_values,
        image_sizes,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        vision_aspect_ratio=vision_aspect_ratio,
        batch_num_images=batch_num_images,
    )
    return _image_feature_row_count(result)


def truncate_image_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id_value: int,
    max_image_tokens: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the first max_image_tokens image placeholders; drop extras (anyres_max mismatch fix)."""
    if max_image_tokens < 0:
        return input_ids, attention_mask
    trimmed_rows: list[list[int]] = []
    trimmed_masks: list[list[int]] = []
    for row_ids, row_mask in zip(input_ids, attention_mask):
        valid = [(int(t), int(m)) for t, m in zip(row_ids.tolist(), row_mask.tolist()) if m]
        if not valid:
            trimmed_rows.append(row_ids.tolist())
            trimmed_masks.append(row_mask.tolist())
            continue
        new_ids: list[int] = []
        kept_img = 0
        for tok, _ in valid:
            if tok == image_token_id_value:
                if kept_img < max_image_tokens:
                    new_ids.append(tok)
                    kept_img += 1
            else:
                new_ids.append(tok)
        trimmed_rows.append(new_ids)
        trimmed_masks.append([1] * len(new_ids))
    max_len = max(len(r) for r in trimmed_rows)
    out_ids = []
    out_mask = []
    for row, mask in zip(trimmed_rows, trimmed_masks):
        pad_len = max_len - len(row)
        out_ids.append(row + [pad_token_id] * pad_len)
        out_mask.append(mask + [0] * pad_len)
    return (
        torch.tensor(out_ids, dtype=input_ids.dtype, device=input_ids.device),
        torch.tensor(out_mask, dtype=attention_mask.dtype, device=attention_mask.device),
    )


def align_teacher_prompt_image_tokens(
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values,
    image_sizes,
    batch_num_images: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sync image placeholder count in input_ids to vision feature count."""
    if pixel_values is None:
        return input_ids, attention_mask
    img_id = image_token_id(processor)
    n_tokens = int((input_ids == img_id).sum().item())
    n_features = expected_image_feature_count(
        model, pixel_values, image_sizes, batch_num_images=batch_num_images
    )
    if n_features <= 0 or n_tokens == n_features:
        return input_ids, attention_mask
    opsd_debug.log(
        "teacher_batching",
        "align teacher image tokens to vision features",
        image_tokens=n_tokens,
        image_features=n_features,
        delta=n_tokens - n_features,
    )
    pad_id = int(processor.tokenizer.pad_token_id)
    return truncate_image_tokens(input_ids, attention_mask, img_id, n_features, pad_id)


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
