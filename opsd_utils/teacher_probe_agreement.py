"""Agreement gate for no-gold teacher probe attempts."""
from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from typing import Any

from data_utils.chart.evaluator import eval_teacher_probe_chart


@dataclass(frozen=True)
class TeacherProbeAgreementDecision:
    agreement_accepted: bool
    verified_correct: bool
    reason_code: str
    selected_output: str
    selected_index: int
    selected_score: float
    normalized_answer: str
    parsed_answers: tuple[str, ...]
    normalized_answers: tuple[str, ...]
    parse_failed_count: int
    agreement_count: int


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[*_\[\]\(\)]", "", text).strip()
    while text and text[-1] in string.punctuation and text[-1] not in {"%", "/"}:
        text = text[:-1].strip()
    return re.sub(r"\s+", " ", text)


def _as_number(value: str) -> float | None:
    text = _clean_text(value).replace(",", "").rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def normalize_teacher_probe_agreement_answer(value: Any) -> str:
    """Normalize short teacher answers for reference-free agreement checks."""
    text = _clean_text(value).lower()
    if not text:
        return ""

    comma_form = re.sub(r"(?i)\s+\band\b\s+", ",", text)
    items = [_clean_text(part).lower() for part in comma_form.split(",") if part.strip()]
    if len(items) > 1:
        return ",".join(items)

    number = _as_number(text)
    if number is not None:
        return _format_number(number)
    return text


def decide_teacher_probe_agreement(
    *,
    outputs: list[str],
    reference: str | list[str],
    answer_flag: str = "answer:",
    max_relative_change: float = 0.05,
    selected_index: int = 0,
    min_agree: int | None = None,
) -> TeacherProbeAgreementDecision:
    if not outputs:
        return TeacherProbeAgreementDecision(
            agreement_accepted=False,
            verified_correct=False,
            reason_code="empty_outputs",
            selected_output="",
            selected_index=0,
            selected_score=0.0,
            normalized_answer="",
            parsed_answers=(),
            normalized_answers=(),
            parse_failed_count=0,
            agreement_count=0,
        )

    selected_index = max(0, min(int(selected_index), len(outputs) - 1))
    min_agree = len(outputs) if min_agree is None else max(1, int(min_agree))
    parsed_answers: list[str] = []
    normalized_answers: list[str] = []
    scores: list[float] = []
    parse_failed_count = 0
    for output in outputs:
        score, parsed = eval_teacher_probe_chart(
            output,
            reference,
            max_relative_change,
            answer_flag=answer_flag,
        )
        parsed_answers.append(parsed.answer)
        normalized_answers.append(normalize_teacher_probe_agreement_answer(parsed.answer))
        scores.append(float(score))
        if parsed.parse_failed or not parsed.answer:
            parse_failed_count += 1

    selected_output = str(outputs[selected_index] or "")
    selected_norm = normalized_answers[selected_index]
    selected_score = scores[selected_index]
    agreement_count = sum(1 for norm in normalized_answers if norm and norm == selected_norm)

    if parse_failed_count:
        reason_code = "parse_failed"
        agreement_accepted = False
    elif not selected_norm:
        reason_code = "parse_failed"
        agreement_accepted = False
    elif agreement_count < min_agree:
        reason_code = "answer_disagreement"
        agreement_accepted = False
    else:
        agreement_accepted = True
        reason_code = (
            "agreement_verified_correct"
            if selected_score > 0.0
            else "agreement_verified_wrong"
        )

    return TeacherProbeAgreementDecision(
        agreement_accepted=agreement_accepted,
        verified_correct=bool(agreement_accepted and selected_score > 0.0),
        reason_code=reason_code,
        selected_output=selected_output,
        selected_index=selected_index,
        selected_score=selected_score,
        normalized_answer=selected_norm if agreement_accepted else "",
        parsed_answers=tuple(parsed_answers),
        normalized_answers=tuple(normalized_answers),
        parse_failed_count=parse_failed_count,
        agreement_count=agreement_count,
    )
