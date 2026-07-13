"""ChartQA format reward guards against digit-spam and degenerate completions."""
from __future__ import annotations

import os
import re

from opsd_utils.diagnostics import _detect_char_repeat, is_degenerate_completion


def chart_min_thinking_length() -> int:
    return int(os.environ.get("DYME_FORMAT_MIN_THINKING", "8"))


def _answer_tail(response: str, answer_flag: str) -> str:
    parts = re.split(f"(?i){re.escape(answer_flag)}", response or "", maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def is_digit_spam_after_answer(response: str, answer_flag: str = "answer:") -> bool:
    """True when the answer segment is an implausibly long mostly-digit string."""
    tail = _answer_tail(response, answer_flag)
    if len(tail) < 12:
        return False
    digits = sum(ch.isdigit() for ch in tail)
    digit_ratio = digits / max(len(tail), 1)
    return digit_ratio > 0.75 and len(tail) >= 20


def should_zero_chart_format(
    response: str,
    answer_flag: str = "answer:",
    *,
    token_ids: list[int] | None = None,
    require_answer_flag_for_degen: bool = False,
) -> bool:
    """Return True when format reward should be forced to zero on ChartQA."""
    if is_digit_spam_after_answer(response, answer_flag):
        return True
    if _detect_char_repeat(response or ""):
        return True
    ids = token_ids if token_ids is not None else []
    if is_degenerate_completion(
        ids,
        response or "",
        answer_flag=answer_flag,
        require_answer_flag=require_answer_flag_for_degen,
    ):
        return True
    return False


def evaluate_format_reward(
    response: str,
    answer_flag: str,
    count_pattern: re.Pattern[str],
    *,
    min_thinking_length: int | None = None,
    task: str = "",
) -> float:
    """Shared format reward (1.0 / 0.0) for RewardCalculator and RewardCalculatorLocal."""
    min_len = 0 if min_thinking_length is None else min_thinking_length
    flag_lower = answer_flag.lower()
    if "chart" in (task or ""):
        min_len = max(min_len, chart_min_thinking_length())
        if should_zero_chart_format(
            response,
            flag_lower,
            require_answer_flag_for_degen=False,
        ):
            return 0.0

    answer_matches = count_pattern.findall(response)
    if len(answer_matches) != 1:
        return 0.0

    thinking = response.lower().split(flag_lower)[0]
    if "chart" in (task or ""):
        has_goal = len(re.findall(r"(?i)goal:", response or "")) == 1
        if has_goal and len(thinking.strip()) >= min_len:
            return 1.0
        if has_goal:
            return 0.5

    if len(thinking.strip()) < min_len:
        return 0.0

    return 1.0
