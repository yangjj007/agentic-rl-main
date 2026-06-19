"""Batch dedupe and IC prefetch for visual supervision."""
from __future__ import annotations

from typing import Any, Optional

from reward_utils.teacher_generate import TeacherGenerateRequest, teacher_generate_batched_chunks
from reward_utils.visual_ic import build_prompt_s1, extract_visual_facts_teacher, _parse_ic_json, _ic_stats, _ic_text_from_sample
import json


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
    teacher_batch_size: int = 4,
) -> int:
    """Warm IC cache for unique (image, question) pairs. Returns teacher vision call count."""
    if teacher_model is None or processor is None:
        return 0
    if ic_source not in ("teacher_image", "auto"):
        return 0

    seen: set[tuple[str, str]] = set()
    pending: list[tuple[int, str, Any, dict]] = []
    n = min(len(samples), len(images), len(questions))
    for idx in range(n):
        key = ic_cache_key(images[idx], questions[idx])
        if key in seen or key in cache:
            continue
        seen.add(key)
        if ic_source == "auto":
            ic_text, fb = _ic_text_from_sample(samples[idx])
            if ic_text:
                cache[key] = ic_text
                meta = _ic_stats(None, ic_text)
                meta.update(
                    sample_idx=idx,
                    image=image_cache_key(images[idx]),
                    question_preview=questions[idx][:120],
                    ic_source=f"auto_{fb}",
                    parse_ok=True,
                    ic_preview=ic_text[:400],
                )
                if recorder is not None:
                    recorder.record_ic(**meta)
                continue
        pending.append((idx, questions[idx], images[idx], samples[idx]))

    if not pending:
        return 0

    requests = [
        TeacherGenerateRequest(
            prompt_text=build_prompt_s1(question),
            images=[image],
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
        )
        for _idx, question, image, _sample in pending
    ]
    texts, _ = teacher_generate_batched_chunks(
        teacher_model,
        processor,
        requests,
        chunk_size=teacher_batch_size,
        recorder=recorder,
        timing_kind="ic",
    )

    calls = len(pending)
    for (idx, question, image, sample), output in zip(pending, texts):
        key = ic_cache_key(image, question)
        ic_obj, err = _parse_ic_json(output)
        meta: dict[str, Any] = {
            "sample_idx": idx,
            "image": image_cache_key(image),
            "question_preview": question[:120],
            "ic_source": "teacher_image",
            "parse_ok": False,
        }
        if ic_obj is not None:
            ic_text = json.dumps(ic_obj, ensure_ascii=False)
            cache[key] = ic_text
            meta.update(_ic_stats(ic_obj, ic_text))
            meta.update(parse_ok=True, ic_preview=ic_text[:400], raw_teacher_output=output[:200])
        else:
            ic_text, fb = _ic_text_from_sample(sample)
            meta.update(_ic_stats(ic_obj, ic_text))
            meta.update(parse_ok=bool(ic_text), error=err, fallback=fb, raw_teacher_output=output[:200])
            if ic_text:
                cache[key] = ic_text
        if recorder is not None:
            recorder.record_ic(**meta)
    return calls
