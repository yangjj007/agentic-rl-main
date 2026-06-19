"""Shared 7B teacher generate helpers for visual supervision."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from data_utils.paths import resolve_image_path
from opsd_utils.teacher_batching import (
    align_teacher_prompt_image_tokens,
    as_batch_num_images_tensor,
    model_inference_device,
    move_batch_num_images_to_model_device,
    move_pixel_values_to_model_device,
    process_teacher_sample,
    stack_teacher_processor_batches,
)


@dataclass
class TeacherGenerateRequest:
    prompt_text: str
    images: list[Any] = field(default_factory=list)
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


def _load_pil_image(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        path = resolve_image_path(image)
        img = Image.open(path)
    else:
        path = getattr(image, "filename", None)
        if not path:
            return None
        img = Image.open(resolve_image_path(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _images_to_pil(images: Optional[list[Any]]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for img in images or []:
        loaded = _load_pil_image(img)
        if loaded is not None:
            out.append(loaded)
    return out


def _pad_aligned_batch_rows(
    ids_rows: list[torch.Tensor],
    mask_rows: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(row.shape[1]) for row in ids_rows)
    out_ids: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    for ids, mask in zip(ids_rows, mask_rows):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=pad_id)
            mask = F.pad(mask, (0, pad_len), value=0)
        out_ids.append(ids)
        out_masks.append(mask)
    return torch.cat(out_ids, dim=0), torch.cat(out_masks, dim=0)


def _align_stacked_batch(
    teacher_model,
    processor,
    stacked: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Any, Any, Optional[torch.Tensor]]:
    input_ids = stacked["input_ids"]
    attention_mask = stacked["attention_mask"]
    pv_list = stacked.get("pixel_values_list") or []
    size_list = stacked.get("image_sizes_list") or []
    batch_num_images = stacked.get("batch_num_images") or []
    batch_rows = int(input_ids.shape[0])

    if not pv_list:
        return input_ids, attention_mask, None, None, None

    pad_id = int(processor.tokenizer.pad_token_id)
    ids_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for row in range(batch_rows):
        pv = pv_list[row]
        sizes = size_list[row] if row < len(size_list) else None
        n_img = batch_num_images[row] if row < len(batch_num_images) else int(pv.shape[0])
        bn = as_batch_num_images_tensor(int(max(1, n_img)), pv, batch_rows=1)
        row_ids, row_mask = align_teacher_prompt_image_tokens(
            teacher_model,
            processor,
            input_ids[row : row + 1],
            attention_mask[row : row + 1],
            pv,
            sizes,
            batch_num_images=bn,
        )
        ids_rows.append(row_ids)
        mask_rows.append(row_mask)
    aligned_ids, aligned_mask = _pad_aligned_batch_rows(ids_rows, mask_rows, pad_id)

    shapes = {tuple(pv.shape) for pv in pv_list}
    if len(shapes) == 1:
        pixel_values = torch.cat(pv_list, dim=0)
        image_sizes = torch.cat(size_list, dim=0) if size_list else None
        bn_tensor = torch.tensor(
            [
                int(max(1, batch_num_images[i] if i < len(batch_num_images) else 1))
                for i in range(batch_rows)
            ],
            dtype=torch.long,
        )
        return aligned_ids, aligned_mask, pixel_values, image_sizes, bn_tensor

    return aligned_ids, aligned_mask, pv_list, size_list, None


def _decode_new_tokens(
    processor,
    generated: torch.Tensor,
    prompt_len: int,
) -> list[str]:
    new_ids = generated[:, prompt_len:]
    return [t.strip() for t in processor.batch_decode(new_ids, skip_special_tokens=True)]


def _generate_rows_sequential(
    teacher_model,
    processor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    pv_list: list[torch.Tensor],
    size_list: list[Any],
    batch_num_images: list[int],
    *,
    prompt_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> list[str]:
    texts: list[str] = []
    for row in range(prompt_ids.shape[0]):
        pv = pv_list[row]
        sizes = size_list[row] if row < len(size_list) else None
        n_img = batch_num_images[row] if row < len(batch_num_images) else int(pv.shape[0])
        bn = as_batch_num_images_tensor(int(max(1, n_img)), pv, batch_rows=1)
        row_ids = prompt_ids[row : row + 1]
        row_mask = prompt_mask[row : row + 1]
        row_ids, row_mask = align_teacher_prompt_image_tokens(
            teacher_model,
            processor,
            row_ids,
            row_mask,
            pv,
            sizes,
            batch_num_images=bn,
        )
        teacher_device = model_inference_device(teacher_model)
        row_ids = row_ids.to(teacher_device)
        row_mask = row_mask.to(teacher_device)
        pv = move_pixel_values_to_model_device(teacher_model, pv)
        bn = move_batch_num_images_to_model_device(teacher_model, bn)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            generated = teacher_model.generate(
                input_ids=row_ids,
                attention_mask=row_mask,
                pixel_values=pv,
                image_sizes=sizes,
                batch_num_images=bn,
                **gen_kwargs,
            )
        texts.append(_decode_new_tokens(processor, generated, row_ids.shape[1])[0])
    return texts


def teacher_generate_batch(
    teacher_model,
    processor,
    requests: list[TeacherGenerateRequest],
    *,
    recorder: Any = None,
    timing_kind: str = "teacher",
) -> tuple[list[str], float]:
    """Batched teacher generate; keeps chart images in the forward when provided."""
    if not requests:
        return [], 0.0

    t0 = time.perf_counter()
    per_sample_batches: list[dict[str, Any]] = []
    max_new_tokens = max(r.max_new_tokens for r in requests)
    do_sample = any(r.do_sample for r in requests)
    temperature = max(r.temperature for r in requests)
    top_p = min(r.top_p for r in requests)
    repetition_penalty = max(r.repetition_penalty for r in requests)

    for req in requests:
        pil_images = _images_to_pil(req.images)
        per_sample_batches.append(process_teacher_sample(processor, req.prompt_text, pil_images))

    stacked = stack_teacher_processor_batches(processor, per_sample_batches)
    prompt_len = int(stacked["input_ids"].shape[1])
    aligned_ids, aligned_mask, pixel_values, image_sizes, batch_num_images = _align_stacked_batch(
        teacher_model,
        processor,
        stacked,
    )

    try:
        if isinstance(pixel_values, list):
            texts = _generate_rows_sequential(
                teacher_model,
                processor,
                aligned_ids,
                aligned_mask,
                pixel_values,
                image_sizes if isinstance(image_sizes, list) else [],
                stacked.get("batch_num_images") or [],
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            teacher_device = model_inference_device(teacher_model)
            aligned_ids = aligned_ids.to(teacher_device)
            aligned_mask = aligned_mask.to(teacher_device)
            pixel_values = move_pixel_values_to_model_device(teacher_model, pixel_values)
            batch_num_images = move_batch_num_images_to_model_device(teacher_model, batch_num_images)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
                gen_kwargs["top_p"] = top_p
            forward_kwargs: dict[str, Any] = {
                "input_ids": aligned_ids,
                "attention_mask": aligned_mask,
            }
            if pixel_values is not None:
                forward_kwargs["pixel_values"] = pixel_values
                forward_kwargs["image_sizes"] = image_sizes
                forward_kwargs["batch_num_images"] = batch_num_images
            with torch.no_grad():
                generated = teacher_model.generate(**forward_kwargs, **gen_kwargs)
            texts = _decode_new_tokens(processor, generated, prompt_len)
    except Exception:
        texts = []
        for req in requests:
            text, _ = teacher_generate_one(
                teacher_model,
                processor,
                req.prompt_text,
                req.images,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                recorder=None,
            )
            texts.append(text)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if recorder is not None and hasattr(recorder, "record_teacher_timing"):
        recorder.record_teacher_timing(
            timing_kind,
            latency_ms=latency_ms,
            n_calls=len(requests),
            batch_size=len(requests),
        )
    return texts, latency_ms


def teacher_generate_one(
    teacher_model,
    processor,
    prompt_text: str,
    images: Optional[list[Any]] = None,
    *,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    recorder: Any = None,
    timing_kind: str = "teacher",
) -> tuple[str, float]:
    """Run one teacher forward; returns (decoded_text, latency_ms)."""
    req = TeacherGenerateRequest(
        prompt_text=prompt_text,
        images=list(images or []),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    texts, latency_ms = teacher_generate_batch(
        teacher_model,
        processor,
        [req],
        recorder=recorder,
        timing_kind=timing_kind,
    )
    return texts[0] if texts else "", latency_ms


def teacher_generate_batched_chunks(
    teacher_model,
    processor,
    requests: list[TeacherGenerateRequest],
    *,
    chunk_size: int = 4,
    recorder: Any = None,
    timing_kind: str = "teacher",
) -> tuple[list[str], float]:
    """Run teacher_generate_batch in fixed-size chunks."""
    if not requests:
        return [], 0.0
    chunk_size = max(1, int(chunk_size))
    all_texts: list[str] = []
    total_ms = 0.0
    for start in range(0, len(requests), chunk_size):
        chunk = requests[start : start + chunk_size]
        texts, latency_ms = teacher_generate_batch(
            teacher_model,
            processor,
            chunk,
            recorder=recorder,
            timing_kind=timing_kind,
        )
        all_texts.extend(texts)
        total_ms += latency_ms
    return all_texts, total_ms
