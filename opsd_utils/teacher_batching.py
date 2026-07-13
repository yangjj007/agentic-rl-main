"""Helpers for LLaVA-OV teacher batches (per-sample vision tensors)."""
from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from opsd_utils import debug_log as opsd_debug
from opsd_utils.deepspeed_utils import should_colocate_teacher_with_student

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
            if value.dim() == 0:
                continue
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


def expand_teacher_tensors_to_full_batch(
    teacher_tensors: dict[str, Any],
    build_indices: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """
    Scatter compact teacher rows (built only for OPSD / dummy anchor indices)
    into tensors aligned with the full student batch for GA splitting.
    """
    if not teacher_tensors or not build_indices or batch_size <= 0:
        return teacher_tensors
    compact_ids = teacher_tensors.get("teacher_prompt_ids")
    if compact_ids is None:
        return teacher_tensors
    compact_n = int(compact_ids.shape[0])
    if compact_n >= batch_size and build_indices == list(range(batch_size)):
        return teacher_tensors

    idx_to_compact = {bi: i for i, bi in enumerate(build_indices)}
    fallback_compact = idx_to_compact[build_indices[0]]
    reorder = [idx_to_compact.get(i, fallback_compact) for i in range(batch_size)]

    out = dict(teacher_tensors)
    compact_mask = teacher_tensors["teacher_prompt_mask"]
    reorder_tensor = torch.tensor(reorder, device=compact_ids.device, dtype=torch.long)
    out["teacher_prompt_ids"] = compact_ids.index_select(0, reorder_tensor)
    out["teacher_prompt_mask"] = compact_mask.index_select(0, reorder_tensor)

    compact_counts = teacher_tensors.get("teacher_num_images")
    if compact_counts is not None:
        if isinstance(compact_counts, torch.Tensor):
            counts_list = [int(max(1, c)) for c in compact_counts.detach().cpu().tolist()]
        else:
            counts_list = [int(max(1, c)) for c in compact_counts]
        out["teacher_num_images"] = torch.tensor(
            [counts_list[i] for i in reorder],
            device=compact_ids.device,
            dtype=torch.long,
        )

    compact_prefix_lens = teacher_tensors.get("teacher_response_prefix_lens")
    if compact_prefix_lens is not None:
        if isinstance(compact_prefix_lens, torch.Tensor):
            prefix_lens = [int(max(0, c)) for c in compact_prefix_lens.detach().cpu().tolist()]
        else:
            prefix_lens = [int(max(0, c)) for c in compact_prefix_lens]
        out["teacher_response_prefix_lens"] = torch.tensor(
            [prefix_lens[i] for i in reorder],
            device=compact_ids.device,
            dtype=torch.long,
        )

    pv_list = teacher_tensors.get("teacher_pixel_values_list")
    if pv_list is not None:
        out["teacher_pixel_values_list"] = [pv_list[i] for i in reorder]
    sz_list = teacher_tensors.get("teacher_image_sizes_list")
    if sz_list is not None:
        out["teacher_image_sizes_list"] = [sz_list[i] for i in reorder]

    opsd_debug.log(
        "teacher_batching",
        "expand_teacher_tensors_to_full_batch",
        batch_size=batch_size,
        build_indices=build_indices,
        compact_n=compact_n,
    )
    return out


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
            elif isinstance(value, torch.Tensor) and value.dim() == 0:
                chunk[key] = value
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
        if ids.shape[1] < max_len:
            ids, attn = _pad_batch_token_rows(
                [ids], [attn], pad_id, padding_side="left", target_len=max_len
            )
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
        "response_prefix_lens": [
            int(batch.get("response_prefix_len", 0) or 0) for batch in per_sample_batches
        ],
    }
    return out


def _messages_for_teacher(teacher_text: str, images: list[Image.Image]) -> list[dict]:
    """Build chat messages with PIL images embedded (single processor tokenize path)."""
    content: list[dict] = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": teacher_text})
    return [{"role": "user", "content": content}]


def _tokenize_response_prefix(processor, response_prefix: str) -> torch.Tensor | None:
    text = str(response_prefix or "")
    if not text:
        return None
    encoded = processor.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = encoded.get("input_ids")
    if ids is None or ids.numel() == 0:
        return None
    return ids


def process_teacher_sample(
    processor,
    teacher_text: str,
    images: list[Any],
    *,
    response_prefix: str = "",
) -> dict[str, Any]:
    """Tokenize one teacher sample via processor.apply_chat_template(tokenize=True)."""
    pil_images = [img for img in images if isinstance(img, Image.Image)]
    messages = _messages_for_teacher(teacher_text, pil_images)
    tok = processor.tokenizer
    prev_padding_side = getattr(tok, "padding_side", "right")
    if prev_padding_side != "left":
        tok.padding_side = "left"
    try:
        batch = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    finally:
        tok.padding_side = prev_padding_side
    prefix_ids = _tokenize_response_prefix(processor, response_prefix)
    prefix_len = int(prefix_ids.shape[1]) if prefix_ids is not None else 0
    if prefix_ids is not None:
        prefix_ids = prefix_ids.to(batch["input_ids"].device)
        prefix_mask = torch.ones_like(prefix_ids, dtype=batch["attention_mask"].dtype)
        batch["input_ids"] = torch.cat([batch["input_ids"], prefix_ids], dim=1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], prefix_mask], dim=1)
    batch["response_prefix_len"] = prefix_len
    batch["response_prefix_text"] = str(response_prefix or "")
    n_img_tok = count_image_tokens(batch["input_ids"], processor)
    opsd_debug.log(
        "teacher_batching",
        "process_teacher_sample",
        num_images=len(pil_images),
        input_ids_shape=tuple(batch["input_ids"].shape),
        pixel_values_shape=tuple(batch["pixel_values"].shape) if "pixel_values" in batch else None,
        image_token_count=n_img_tok,
        response_prefix_len=prefix_len,
    )
    return batch


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def resolve_teacher_device_map(
    device_map: Optional[str],
    *,
    local_rank: int,
    num_gpus: int,
) -> str:
    """
    Place frozen teacher relative to this rank's trainable student.

    Modes:
    - ``same`` / ``colocate`` / ``local``: teacher on ``cuda:{local_rank}`` (use with DeepSpeed ZeRO).
    - ``auto`` (default DDP): complementary GPU when ``num_gpus >= 2``.
    - ``auto`` + DeepSpeed Accelerate config: colocate on ``cuda:{local_rank}``.
    - explicit ``cuda:N``: honored unless it collides with the student device.
    """
    student_dev = f"cuda:{local_rank}"
    raw = (device_map or os.environ.get("DYME_TEACHER_DEVICE_MAP", "")).strip()

    if raw.lower() in ("same", "colocate", "local"):
        return student_dev

    if should_colocate_teacher_with_student(raw):
        return student_dev

    if raw.lower() in ("", "auto", "complement", "opposite"):
        if num_gpus >= 2:
            return f"cuda:{(local_rank + 1) % num_gpus}"
        return student_dev

    if raw == student_dev and num_gpus >= 2:
        resolved = f"cuda:{(local_rank + 1) % num_gpus}"
        opsd_debug.log(
            "teacher_placement",
            "teacher device collides with student; using complement GPU",
            requested=raw,
            local_rank=local_rank,
            resolved=resolved,
        )
        return resolved
    return raw


def log_teacher_placement(
    *,
    local_rank: int,
    num_gpus: int,
    teacher_path: str,
    resolved_device: str,
    requested_map: Optional[str],
) -> None:
    student_dev = f"cuda:{local_rank}"
    req = requested_map or os.environ.get("DYME_TEACHER_DEVICE_MAP", "") or "auto"
    print(
        f"[DyME] rank={local_rank}/{num_gpus}: frozen teacher "
        f"{teacher_path} on {resolved_device} "
        f"(student DDP device {student_dev}, requested_map={req!r})",
        flush=True,
    )


def _as_torch_device(dev) -> Optional[torch.device]:
    return dev if isinstance(dev, torch.device) else None


def model_inference_device(model) -> torch.device:
    """Device for teacher/student forward inputs (embeddings or vision tower)."""
    inner = _unwrap_model(model)
    get_emb = getattr(inner, "get_input_embeddings", None)
    if callable(get_emb):
        emb = get_emb()
        if emb is not None and hasattr(emb, "weight"):
            dev = _as_torch_device(emb.weight.device)
            if dev is not None:
                return dev
    core = getattr(inner, "model", inner)
    tower = getattr(core, "vision_tower", None)
    if tower is not None:
        params_fn = getattr(tower, "parameters", None)
        if callable(params_fn):
            for p in params_fn():
                dev = _as_torch_device(p.device)
                if dev is not None and dev.type != "meta":
                    return dev
    params_fn = getattr(inner, "parameters", None)
    if callable(params_fn):
        for p in params_fn():
            dev = _as_torch_device(p.device)
            if dev is not None and dev.type != "meta":
                return dev
    return torch.device("cpu")


def model_inference_dtype(model) -> torch.dtype:
    inner = _unwrap_model(model)
    params_fn = getattr(inner, "parameters", None)
    if callable(params_fn):
        for p in params_fn():
            dev = _as_torch_device(p.device)
            if dev is not None and dev.type != "meta":
                return p.dtype
    return torch.bfloat16


def move_pixel_values_to_model_device(model, pixel_values):
    """Align vision tensor device/dtype with the model (cross-model OPD safety)."""
    if pixel_values is None or not isinstance(pixel_values, torch.Tensor):
        return pixel_values
    device = model_inference_device(model)
    dtype = model_inference_dtype(model)
    return pixel_values.to(device=device, dtype=dtype)


def move_batch_num_images_to_model_device(model, batch_num_images: Optional[torch.Tensor]):
    if batch_num_images is None or not isinstance(batch_num_images, torch.Tensor):
        return batch_num_images
    return batch_num_images.to(device=model_inference_device(model))


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


def student_batch_num_images_tensor(
    pixel_values: Optional[torch.Tensor],
    batch_rows: int,
) -> Optional[torch.Tensor]:
    """
    Infer batch_num_images for student / collator-batched pixel_values.

    HF processor output uses dim0 as batch size (one chart image per row); do not
    treat dim0 as the image count when it equals batch_rows.
    """
    if pixel_values is None or batch_rows <= 0:
        return None
    device = pixel_values.device
    pv_batch = int(pixel_values.shape[0])
    if pixel_values.ndim >= 4 and pv_batch == batch_rows:
        return torch.ones(batch_rows, device=device, dtype=torch.long)
    n_img = int(max(1, pv_batch))
    return torch.tensor([n_img] * batch_rows, device=device, dtype=torch.long)


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
    pixel_values = move_pixel_values_to_model_device(model, pixel_values)
    batch_num_images = move_batch_num_images_to_model_device(model, batch_num_images)
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
        if pad_len > 0:
            row = [pad_token_id] * pad_len + row
            mask = [0] * pad_len + mask
        out_ids.append(row)
        out_mask.append(mask)
    return (
        torch.tensor(out_ids, dtype=input_ids.dtype, device=input_ids.device),
        torch.tensor(out_mask, dtype=attention_mask.dtype, device=attention_mask.device),
    )


def _slice_row_image_sizes(image_sizes, index: int):
    if image_sizes is None:
        return None
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.dim() == 0:
            return image_sizes
        if image_sizes.shape[0] > index:
            return image_sizes[index : index + 1]
        return image_sizes
    if isinstance(image_sizes, (list, tuple)) and len(image_sizes) > index:
        return image_sizes[index]
    return image_sizes


def _row_batch_num_images(
    batch_num_images: Optional[torch.Tensor],
    index: int,
    pixel_values_row,
) -> Optional[torch.Tensor]:
    if batch_num_images is None:
        return student_batch_num_images_tensor(pixel_values_row, 1)
    if batch_num_images.numel() == 1:
        return batch_num_images
    if batch_num_images.shape[0] > index:
        return batch_num_images[index : index + 1]
    return batch_num_images


def _align_prompt_image_tokens_one_row(
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values,
    image_sizes,
    batch_num_images: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align image placeholders for a single batch row (shape [1, L])."""
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
        "align image tokens to vision features (one row)",
        image_tokens=n_tokens,
        image_features=n_features,
        delta=n_tokens - n_features,
        batch_num_images=batch_num_images.detach().cpu().tolist() if batch_num_images is not None else None,
    )
    pad_id = int(processor.tokenizer.pad_token_id)
    return truncate_image_tokens(input_ids, attention_mask, img_id, n_features, pad_id)


def _pad_batch_token_rows(
    ids_rows: list[torch.Tensor],
    mask_rows: list[torch.Tensor],
    pad_id: int,
    *,
    padding_side: str = "left",
    target_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length token rows for batched decoder-only generate (default: left)."""
    max_len = target_len if target_len is not None else max(int(row.shape[1]) for row in ids_rows)
    out_ids: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    side = (padding_side or "left").lower()
    for ids, mask in zip(ids_rows, mask_rows):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            if side == "left":
                ids = F.pad(ids, (pad_len, 0), value=pad_id)
                mask = F.pad(mask, (pad_len, 0), value=0)
            else:
                ids = F.pad(ids, (0, pad_len), value=pad_id)
                mask = F.pad(mask, (0, pad_len), value=0)
        out_ids.append(ids)
        out_masks.append(mask)
    return torch.cat(out_ids, dim=0), torch.cat(out_masks, dim=0)


def _pad_aligned_batch_rows(
    ids_rows: list[torch.Tensor],
    mask_rows: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pad_batch_token_rows(ids_rows, mask_rows, pad_id, padding_side="left")


def align_teacher_prompt_image_tokens(
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values,
    image_sizes,
    batch_num_images: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sync image placeholder count in input_ids to vision feature count (per sample)."""
    if pixel_values is None:
        return input_ids, attention_mask

    batch_rows = int(input_ids.shape[0])
    if batch_rows <= 1:
        return _align_prompt_image_tokens_one_row(
            model,
            processor,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
            batch_num_images=batch_num_images,
        )

    # Batched prompts must be aligned per row; global token/feature totals hide per-row mismatch.
    pad_id = int(processor.tokenizer.pad_token_id)
    ids_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for row in range(batch_rows):
        row_pv = pixel_values[row : row + 1]
        row_sizes = _slice_row_image_sizes(image_sizes, row)
        row_bn = _row_batch_num_images(batch_num_images, row, row_pv)
        row_ids, row_mask = _align_prompt_image_tokens_one_row(
            model,
            processor,
            input_ids[row : row + 1],
            attention_mask[row : row + 1],
            row_pv,
            row_sizes,
            batch_num_images=row_bn,
        )
        ids_rows.append(row_ids)
        mask_rows.append(row_mask)
    return _pad_aligned_batch_rows(ids_rows, mask_rows, pad_id)


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


def _teacher_num_images_for_row(inputs: dict[str, Any], row: int, pixel_values) -> int:
    counts = inputs.get("teacher_num_images")
    if isinstance(counts, torch.Tensor):
        if counts.dim() == 0:
            return int(max(0, counts.item()))
        if row < counts.shape[0]:
            return int(max(0, counts[row].item()))
    elif isinstance(counts, (list, tuple)) and row < len(counts):
        return int(max(0, counts[row]))
    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() > 0:
        return int(max(0, pixel_values.shape[0]))
    return 0


def stack_teacher_vision_for_generate(
    teacher_tensors: dict[str, Any],
    rows: list[int],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Stack teacher vision rows for one batched LLaVA-OV ``generate`` call.

    This intentionally supports only the safe common case: one image per row
    with identical processed pixel tensor shapes. Mixed patch counts or
    multi-image rows force the trainer to fall back to smaller micro-batches.
    """
    if not rows:
        return None, None, None

    vision_inputs = teacher_tensors
    if "prompt_ids" not in vision_inputs and "teacher_prompt_ids" in vision_inputs:
        vision_inputs = {**teacher_tensors, "prompt_ids": teacher_tensors["teacher_prompt_ids"]}

    pixel_rows: list[torch.Tensor] = []
    size_rows: list[Optional[torch.Tensor]] = []
    saw_text_only = False
    for row in rows:
        pixel_values, image_sizes = get_teacher_vision_for_sample(
            vision_inputs,
            row,
            teacher_image_counts_from_dict(vision_inputs, batch_size_from_tensor_dict(vision_inputs)),
        )
        if pixel_values is None:
            saw_text_only = True
            continue
        if saw_text_only:
            raise ValueError("cannot batch mixed text-only and vision teacher rows")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("teacher pixel_values must be a tensor for batched generate")
        num_images = _teacher_num_images_for_row(vision_inputs, row, pixel_values)
        if num_images != 1:
            raise ValueError("batched teacher generate currently supports one image per row")
        if pixel_rows and tuple(pixel_values.shape) != tuple(pixel_rows[0].shape):
            raise ValueError(
                f"teacher vision shape mismatch: {tuple(pixel_values.shape)} != {tuple(pixel_rows[0].shape)}"
            )
        if image_sizes is not None and not isinstance(image_sizes, torch.Tensor):
            raise ValueError("teacher image_sizes must be tensor-like for batched generate")
        pixel_rows.append(pixel_values)
        size_rows.append(image_sizes)

    if not pixel_rows:
        return None, None, None

    if saw_text_only:
        raise ValueError("cannot batch mixed text-only and vision teacher rows")

    pixel_values = torch.cat(pixel_rows, dim=0)
    batch_num_images = torch.ones(len(pixel_rows), device=pixel_values.device, dtype=torch.long)

    if all(size is None for size in size_rows):
        image_sizes = None
    elif all(isinstance(size, torch.Tensor) for size in size_rows):
        normalized_sizes: list[torch.Tensor] = []
        first_shape: tuple[int, ...] | None = None
        for size in size_rows:
            assert isinstance(size, torch.Tensor)
            size_i = size.unsqueeze(0) if size.dim() == 1 else size
            if first_shape is None:
                first_shape = tuple(size_i.shape)
            elif tuple(size_i.shape) != first_shape:
                raise ValueError("teacher image_sizes shape mismatch")
            normalized_sizes.append(size_i)
        image_sizes = torch.cat(normalized_sizes, dim=0)
    else:
        raise ValueError("cannot batch mixed missing and present teacher image_sizes")

    return pixel_values, image_sizes, batch_num_images
