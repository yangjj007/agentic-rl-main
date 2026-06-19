"""Shared 7B teacher generate helpers for visual supervision."""
from __future__ import annotations

import time
from typing import Any, Optional

import torch
from PIL import Image

from data_utils.paths import resolve_image_path
from opsd_utils.teacher_batching import (
    align_teacher_prompt_image_tokens,
    as_batch_num_images_tensor,
    model_inference_device,
    move_batch_num_images_to_model_device,
    move_pixel_values_to_model_device,
    process_teacher_sample,
)


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
) -> tuple[str, float]:
    """Run one teacher forward; returns (decoded_text, latency_ms)."""
    pil_images: list[Image.Image] = []
    for img in images or []:
        loaded = _load_pil_image(img)
        if loaded is not None:
            pil_images.append(loaded)

    t0 = time.perf_counter()
    batch = process_teacher_sample(processor, prompt_text, pil_images)
    prompt_ids = batch["input_ids"]
    prompt_mask = batch["attention_mask"]
    pixel_values = batch.get("pixel_values")
    image_sizes = batch.get("image_sizes")
    n_img = int(pixel_values.shape[0]) if pixel_values is not None else 0
    batch_num_images = as_batch_num_images_tensor(max(1, n_img), pixel_values)

    if pixel_values is not None:
        prompt_ids, prompt_mask = align_teacher_prompt_image_tokens(
            teacher_model,
            processor,
            prompt_ids,
            prompt_mask,
            pixel_values,
            image_sizes,
            batch_num_images=batch_num_images,
        )

    teacher_device = model_inference_device(teacher_model)
    prompt_ids = prompt_ids.to(teacher_device)
    prompt_mask = prompt_mask.to(teacher_device)
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
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask,
    }
    if pixel_values is not None:
        forward_kwargs["pixel_values"] = pixel_values
        forward_kwargs["image_sizes"] = image_sizes
        forward_kwargs["batch_num_images"] = batch_num_images

    with torch.no_grad():
        generated = teacher_model.generate(**forward_kwargs, **gen_kwargs)

    new_ids = generated[:, prompt_ids.shape[1] :]
    text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return text, latency_ms
