"""Detect OPSD / completion patterns that indicate privileged-information leakage."""
from __future__ import annotations

import re
from typing import Any

_LEAKAGE_PHRASES = (
    "reference answer",
    "reference reasoning",
    "according to the reference",
    "according to the answer",
    "参考答案",
)

_ANSWER_IN_COMPLETION = re.compile(r"(?i)answer:\s*.+")


def completion_has_leakage_pattern(
    text: str,
    gold_answer: str | None = None,
    *,
    min_gold_substring_len: int = 4,
) -> bool:
    """Return True if completion text suggests privileged-info cheating."""
    if not text or not text.strip():
        return False
    lower = text.lower()
    for phrase in _LEAKAGE_PHRASES:
        if phrase in lower:
            return True
    if gold_answer:
        gold = gold_answer.strip()
        if len(gold) >= min_gold_substring_len and gold.lower() in lower:
            # Short numeric answers may false-positive; require Answer: prefix context
            if _ANSWER_IN_COMPLETION.search(text):
                return False
            if len(gold) >= 8:
                return True
    return False


def privileged_suffix_has_gold(suffix: str, sample: dict[str, Any]) -> bool:
    if not suffix.strip():
        return False
    if "[Reference Answer]" in suffix or "[Reference Reasoning]" in suffix:
        return True
    answer = (sample.get("answer") or "").strip()
    hint = (sample.get("hint") or "").strip()
    if answer and answer in suffix:
        return True
    if hint and len(hint) >= 8 and hint in suffix:
        return True
    return False
