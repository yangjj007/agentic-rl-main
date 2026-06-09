import os
from typing import Any, Optional

import torch
from PIL import Image

from data_utils.paths import resolve_image_path
from opsd_utils import debug_log as opsd_debug


def _load_image(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, str):
        path = resolve_image_path(image)
        if os.path.exists(path):
            img = Image.open(path)
            return img.convert("RGB")
    return None


def tokenize_teacher_prompt(processor, student_prompt: str, privileged_suffix: str, image: Any) -> dict:
    """Tokenize teacher multimodal prompt = student question + privileged suffix."""
    pil_image = _load_image(image)
    teacher_text = student_prompt
    if privileged_suffix.strip():
        teacher_text = f"{student_prompt}\n\n{privileged_suffix.strip()}"

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt",
        has_image=pil_image is not None,
        suffix_len=len(privileged_suffix.strip()),
        teacher_text_len=len(teacher_text),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": teacher_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if pil_image is not None:
        batch = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    else:
        batch = processor(text=[text], return_tensors="pt", padding=True)
    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt result",
        input_ids_shape=tuple(batch["input_ids"].shape),
        has_pixel_values="pixel_values" in batch,
    )
    return batch


def build_teacher_prompt_batch(
    processor,
    samples: list[dict[str, Any]],
    indices: list[int],
    provider_names: list[str],
    device,
) -> dict[str, Any]:
    """Build padded teacher prompt tensors for OPSD samples at given indices."""
    from opsd_utils.privileged import build_privileged_context

    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch enter",
        num_indices=len(indices),
        indices=indices,
        num_samples=len(samples),
        provider_names=provider_names,
        device=str(device),
    )

    if not indices:
        opsd_debug.log("teacher_prompt", "empty indices, return {}")
        return {}

    prompt_ids_list = []
    prompt_mask_list = []
    pixel_values_list = []
    image_sizes_list = []
    has_images = False

    for idx in indices:
        sample = samples[idx]
        suffix, teacher_image = build_privileged_context(sample, provider_names)
        image = teacher_image if teacher_image is not None else sample.get("image")
        opsd_debug.log(
            "teacher_prompt",
            "build sample teacher prompt",
            idx=idx,
            suffix_preview=(suffix[:160] + "...") if len(suffix) > 160 else suffix,
            has_teacher_image=teacher_image is not None,
            image_type=type(image).__name__,
        )
        batch = tokenize_teacher_prompt(processor, sample["prompt"], suffix, image)
        prompt_ids_list.append(batch["input_ids"][0])
        prompt_mask_list.append(batch["attention_mask"][0])
        if "pixel_values" in batch:
            has_images = True
            pixel_values_list.append(batch["pixel_values"])
        if "image_sizes" in batch:
            image_sizes_list.append(batch["image_sizes"])

    from torch.nn.utils.rnn import pad_sequence

    pad_id = processor.tokenizer.pad_token_id
    prompt_ids = pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_id).to(device)
    prompt_mask = pad_sequence(prompt_mask_list, batch_first=True, padding_value=0).to(device)

    out = {"teacher_prompt_ids": prompt_ids, "teacher_prompt_mask": prompt_mask}
    if has_images:
        out["teacher_pixel_values"] = torch.cat(pixel_values_list, dim=0).to(device)
    if image_sizes_list:
        if isinstance(image_sizes_list[0], torch.Tensor):
            out["teacher_image_sizes"] = torch.cat(image_sizes_list, dim=0).to(device)
        else:
            out["teacher_image_sizes"] = image_sizes_list
    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch done",
        teacher_prompt_ids_shape=tuple(prompt_ids.shape),
        teacher_prompt_mask_shape=tuple(prompt_mask.shape),
        has_teacher_pixel_values="teacher_pixel_values" in out,
    )
    return out
