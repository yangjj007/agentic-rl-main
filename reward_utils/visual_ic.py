"""Online I_c extraction via 7B teacher + Prompt S1."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from data_utils.chart.deplot_pipeline import format_deplot_for_teacher, is_deplot_placeholder
from data_utils.privileged_schema import parse_visual_fact
from reward_utils.teacher_generate import teacher_generate_one

PROMPT_S1 = """You are a helpful assistant that analyzes images and provides visual facts.
Your response MUST be a single, valid JSON object.
The JSON object should contain:
1. "description": A detailed and accurate description of the image.
2. "objects": A list of key objects, including their name, attributes, and approximate position in the image.

Example format:
{
"description": "A person riding a bicycle on a city street.... (detailed description here)",
"objects": [
{"name": "person", "attributes": ["wearing helmet", "blue shirt"], "position": "center"},
{"name": "bicycle", "attributes": ["red", "mountain bike"], "position": "center"},
{"name": "street", "attributes": ["asphalt", "wet"], "position": "bottom"}
]
}

Analyze the attached image and provide the visual facts in the required JSON format.
For context, the user will be asked this question about the image (do not answer the question, just use it for context):
"__QUESTION__"
"""


def build_prompt_s1(question: str) -> str:
    """Insert question without str.format (JSON braces in template are literal)."""
    return PROMPT_S1.replace("__QUESTION__", str(question or ""))


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_ic_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    raw = _strip_code_fence(text)
    if not raw:
        return None, "empty_output"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError:
            return None, "json_decode"
    return None, "json_decode"


def ic_text_from_offline_sample(sample: dict[str, Any]) -> tuple[str, str]:
    """DyME-aligned offline I_c: DePlot table > hint visual facts > hint."""
    deplot_vf = sample.get("visual_fact_deplot")
    if deplot_vf and not is_deplot_placeholder(deplot_vf):
        text = format_deplot_for_teacher(deplot_vf)
        if text:
            return text, "deplot"

    for key in ("visual_fact_hint", "visual_fact", "visual_facts", "hint"):
        raw = sample.get(key)
        if raw:
            text = parse_visual_fact(raw)
            if text:
                return text, f"hint_{key}"
    return "", "empty"


def _ic_text_from_sample(sample: dict[str, Any]) -> tuple[str, str]:
    return ic_text_from_offline_sample(sample)


def _ic_stats(ic_obj: Optional[dict], ic_text: str) -> dict[str, Any]:
    objects = []
    if isinstance(ic_obj, dict):
        objects = ic_obj.get("objects") or []
    return {
        "ic_chars": len(ic_text),
        "ic_objects_count": len(objects) if isinstance(objects, list) else 0,
    }


def extract_visual_facts_teacher(
    *,
    teacher_model,
    processor,
    sample: dict[str, Any],
    question: str,
    image: Any,
    ic_source: str = "auto",
    max_new_tokens: int = 768,
    cache: Optional[dict[tuple[str, str], str]] = None,
    recorder: Any = None,
    sample_idx: int = 0,
) -> tuple[str, dict[str, Any]]:
    """
    Returns (ic_text_for_prompts, meta).
    ic_text is a JSON-ish string used in prompt_thinking_reward / prompt_refine.
    """
    image_key = image if isinstance(image, str) else getattr(image, "filename", str(image))
    cache_key = (str(image_key), str(question))
    meta: dict[str, Any] = {
        "sample_idx": sample_idx,
        "image": image_key,
        "question_preview": question[:120],
        "ic_source": ic_source,
        "parse_ok": False,
    }

    if cache is not None and cache_key in cache:
        ic_text = cache[cache_key]
        meta.update(_ic_stats(None, ic_text))
        meta.update(parse_ok=True, cache_hit=True, ic_preview=ic_text[:400])
        if recorder is not None:
            recorder.record_ic(**meta)
        return ic_text, meta

    if ic_source in ("auto", "teacher_image"):
        ic_text, fb = _ic_text_from_sample(sample)
        if ic_text:
            meta.update(_ic_stats(None, ic_text))
            meta.update(parse_ok=True, ic_source=f"offline_{fb}", ic_preview=ic_text[:400])
            if cache is not None:
                cache[cache_key] = ic_text
            if recorder is not None:
                recorder.record_ic(**meta)
            return ic_text, meta

    if ic_source == "auto" or teacher_model is None or processor is None:
        ic_text, fb = _ic_text_from_sample(sample)
        meta.update(_ic_stats(None, ic_text))
        meta.update(parse_ok=bool(ic_text), fallback=fb, error="teacher_unavailable")
        if recorder is not None:
            recorder.record_ic(**meta)
        return ic_text, meta

    prompt = build_prompt_s1(question)
    try:
        output, latency_ms = teacher_generate_one(
            teacher_model,
            processor,
            prompt,
            [image],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            recorder=recorder,
            timing_kind="ic",
        )
        ic_obj, err = _parse_ic_json(output)
        if ic_obj is not None:
            ic_text = json.dumps(ic_obj, ensure_ascii=False)
            meta.update(_ic_stats(ic_obj, ic_text))
            meta.update(
                parse_ok=True,
                latency_ms=round(latency_ms, 1),
                ic_preview=ic_text[:400],
                raw_teacher_output=output[:200],
            )
            if cache is not None:
                cache[cache_key] = ic_text
        else:
            ic_text, fb = _ic_text_from_sample(sample)
            meta.update(_ic_stats(ic_obj, ic_text))
            meta.update(
                parse_ok=False,
                error=err,
                fallback=fb,
                latency_ms=round(latency_ms, 1),
                raw_teacher_output=output[:200],
            )
    except Exception as exc:
        ic_text, fb = _ic_text_from_sample(sample)
        meta.update(_ic_stats(None, ic_text))
        meta.update(parse_ok=False, error=str(exc)[:120], fallback=fb)

    if recorder is not None:
        recorder.record_ic(**meta)
    return ic_text, meta
