"""Optional perception-grounding reward helpers.

DePlot is diagnostic-only here: it may be measured for overlap, but never
provides reward credit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_%-]*|\d+(?:\.\d+)?%?")
_ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*.*$")


@dataclass(frozen=True)
class PerceptionRewardResult:
    rewards: list[float]
    stats: dict[str, float]


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "") if len(m.group(0)) > 1}


def _reference_answer_variants(reference_answer: str | None) -> set[str]:
    raw = str(reference_answer or "")
    raw = re.sub(r"(?i)\banswer\s*:", "", raw).strip()
    values = {raw.lower()} if raw else set()
    values.update(_tokens(raw))
    return {v for v in values if v}


def sanitize_trusted_hint(hint: str, reference_answer: str | None = None) -> str:
    """Remove explicit answer leakage from a trusted human hint."""
    cleaned = _ANSWER_LINE_RE.sub("", hint or "")
    for value in sorted(_reference_answer_variants(reference_answer), key=len, reverse=True):
        if not value:
            continue
        if re.fullmatch(r"[A-Za-z0-9_%-]+", value):
            pattern = rf"\b{re.escape(value)}\b"
        else:
            pattern = re.escape(value)
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_perception_judge_prompt(
    *,
    source: str,
    sample: dict[str, Any],
    question: str,
    response: str,
) -> str:
    """Build a no-answer prompt for an image teacher judge."""
    if source != "image_teacher":
        raise ValueError(f"Unsupported judge prompt source: {source}")
    reasoning = re.split(r"(?i)\banswer\s*:", response or "", maxsplit=1)[0].strip()
    return (
        "Judge whether the student's reasoning is visually grounded in the image.\n"
        "You will see only the image, the question, and the student's reasoning.\n"
        "Do not answer the chart question. Do not use any reference answer, hint, or DePlot table.\n"
        "Return exactly one label: high, medium, or low.\n\n"
        f"Question: {question or sample.get('question', '')}\n"
        f"Student reasoning:\n{reasoning}\n"
    )


def parse_perception_judge_score(text: str) -> tuple[float, bool]:
    """Parse image-teacher grounding labels into reward values."""
    low = (text or "").strip().lower()
    if "high" in low:
        return 1.0, True
    if "medium" in low:
        return 0.5, True
    if "low" in low:
        return 0.0, True
    return 0.0, False


def _diagnostic_deplot_overlap(sample: dict[str, Any], response: str) -> float:
    deplot = str(sample.get("visual_fact_deplot") or "")
    deplot_tokens = _tokens(deplot)
    if not deplot_tokens:
        return 0.0
    response_tokens = _tokens(response)
    return len(deplot_tokens & response_tokens) / max(len(deplot_tokens), 1)


def _trusted_hint_reward(sample: dict[str, Any], response: str) -> tuple[float, bool]:
    raw_hint = (
        sample.get("trusted_hint")
        or sample.get("hint")
        or ""
    )
    hint = sanitize_trusted_hint(str(raw_hint), reference_answer=str(sample.get("answer") or ""))
    hint_tokens = _tokens(hint)
    if not hint_tokens:
        return 0.0, True
    reasoning = re.split(r"(?i)\banswer\s*:", response or "", maxsplit=1)[0]
    response_tokens = _tokens(reasoning)
    if not response_tokens:
        return 0.0, False
    precision = len(hint_tokens & response_tokens) / max(len(response_tokens), 1)
    recall = len(hint_tokens & response_tokens) / max(len(hint_tokens), 1)
    if precision + recall <= 0:
        return 0.0, False
    return 2 * precision * recall / (precision + recall), False


def score_perception_rewards(
    *,
    samples: Sequence[dict[str, Any]],
    responses: Sequence[str],
    source: str,
) -> PerceptionRewardResult:
    """Score optional perception reward without falling back to DePlot."""
    rewards: list[float] = []
    skipped = 0
    deplot_overlaps: list[float] = []

    for sample, response in zip(samples, responses):
        deplot_overlaps.append(_diagnostic_deplot_overlap(sample, response))
        if source == "trusted_hint":
            reward, was_skipped = _trusted_hint_reward(sample, response)
        elif source == "image_teacher":
            reward, was_skipped = 0.0, True
        else:
            reward, was_skipped = 0.0, True
        rewards.append(float(max(0.0, min(1.0, reward))))
        skipped += int(was_skipped)

    denom = max(len(rewards), 1)
    return PerceptionRewardResult(
        rewards=rewards,
        stats={
            "mean": sum(rewards) / denom,
            "skipped_rate": skipped / denom,
            "judge_parse_fail_rate": 0.0,
            "diagnostic_deplot_overlap_mean": sum(deplot_overlaps) / denom,
        },
    )


def score_image_teacher_perception_rewards(
    *,
    samples: Sequence[dict[str, Any]],
    responses: Sequence[str],
    teacher_model: Any,
    processor: Any,
    batch_size: int = 4,
    max_new_tokens: int = 8,
) -> PerceptionRewardResult:
    """Score visual grounding with a teacher that sees image + question + reasoning only."""
    if teacher_model is None or processor is None:
        return score_perception_rewards(samples=samples, responses=responses, source="image_teacher")

    from reward_utils.teacher_generate import TeacherGenerateRequest, teacher_generate_batched_chunks

    requests: list[TeacherGenerateRequest] = []
    request_indices: list[int] = []
    rewards = [0.0 for _ in responses]
    skipped = 0
    deplot_overlaps: list[float] = []
    for idx, (sample, response) in enumerate(zip(samples, responses)):
        deplot_overlaps.append(_diagnostic_deplot_overlap(sample, response))
        image = sample.get("image")
        if image is None:
            skipped += 1
            continue
        question = str(sample.get("question_wo_prompt") or sample.get("question") or sample.get("prompt") or "")
        requests.append(
            TeacherGenerateRequest(
                prompt_text=build_perception_judge_prompt(
                    source="image_teacher",
                    sample=sample,
                    question=question,
                    response=response,
                ),
                images=[image],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        )
        request_indices.append(idx)

    parse_fail = 0
    if requests:
        outputs, _ = teacher_generate_batched_chunks(
            teacher_model,
            processor,
            requests,
            chunk_size=batch_size,
            recorder=None,
            timing_kind="perception_reward",
        )
        for idx, output in zip(request_indices, outputs):
            score, ok = parse_perception_judge_score(output)
            rewards[idx] = score
            if not ok:
                parse_fail += 1
    denom = max(len(rewards), 1)
    return PerceptionRewardResult(
        rewards=rewards,
        stats={
            "mean": sum(rewards) / denom,
            "skipped_rate": skipped / denom,
            "judge_parse_fail_rate": parse_fail / max(len(request_indices), 1),
            "diagnostic_deplot_overlap_mean": sum(deplot_overlaps) / denom,
        },
    )
