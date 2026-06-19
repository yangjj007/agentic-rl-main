"""Batch dedupe and IC prefetch for visual supervision."""
from __future__ import annotations

from typing import Any, Optional

from reward_utils.visual_ic import extract_visual_facts_teacher


def image_cache_key(image: Any) -> str:
    if image is None:
        return ""
    if isinstance(image, str):
        return image
    return str(getattr(image, "filename", image))


def ic_cache_key(image: Any, question: str) -> tuple[str, str]:
    return (image_cache_key(image), str(question))


def refine_dedupe_key(question: str, hint: str, image: Any = None) -> tuple[str, str, str]:
    return (str(question).strip(), str(hint or "").strip(), image_cache_key(image))


def prefetch_ic_unique(
    *,
    teacher_model,
    processor,
    samples: list[dict],
    images: list[Any],
    questions: list[str],
    ic_source: str,
    max_new_tokens: int,
    cache: dict,
    recorder: Any = None,
) -> int:
    """Warm IC cache for unique (image, question) pairs. Returns teacher vision call count."""
    if teacher_model is None or processor is None:
        return 0
    if ic_source not in ("teacher_image", "auto"):
        return 0

    seen: set[tuple[str, str]] = set()
    calls = 0
    n = min(len(samples), len(images), len(questions))
    for idx in range(n):
        key = ic_cache_key(images[idx], questions[idx])
        if key in seen or key in cache:
            continue
        seen.add(key)
        extract_visual_facts_teacher(
            teacher_model=teacher_model,
            processor=processor,
            sample=samples[idx],
            question=questions[idx],
            image=images[idx],
            ic_source=ic_source,
            max_new_tokens=max_new_tokens,
            cache=cache,
            recorder=recorder,
            sample_idx=idx,
        )
        calls += 1
    return calls
