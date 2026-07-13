from __future__ import annotations

import re
from collections.abc import Sequence


_ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$")
_BAD_SUBSTRINGS = (
    "[Oracle]",
    "[Final Hard Rule]",
    "[Additional Information]",
    "[Verified Hint]",
    "[Reference Answer]",
    "[DePlot]",
    "[Visual Facts",
    "Reasoning style",
)
_BAD_LAST_LINE_RE = re.compile(r"(?i)^\s*(Goal|Observation|Reasoning|Conclusion)\s*:\s*$")


def score_eval_format_reward(response: str) -> float:
    """Reward outputs that look directly parseable by the ChartQA eval script."""
    text = response or ""
    if any(bad.lower() in text.lower() for bad in _BAD_SUBSTRINGS):
        return 0.0

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return 0.0
    if _BAD_LAST_LINE_RE.match(nonempty_lines[-1]):
        return 0.0

    matches = _ANSWER_RE.findall(text)
    if len(matches) != 1:
        return 0.0
    answer = matches[0].strip().strip(".")
    return 1.0 if answer else 0.0


def score_eval_format_rewards(responses: Sequence[str]) -> list[float]:
    return [score_eval_format_reward(response) for response in responses]
