from __future__ import annotations

from opsd_utils.diagnostics import summarize_template_behavior


BEHAVIOR_KEYS = (
    "full_cot_template",
    "partial_cot_template",
    "goal_without_answer",
    "empty_cot_skeleton",
    "malformed_answer_section",
)


def summarize_output_behavior_counts(texts: list[str]) -> dict[str, int]:
    counts = {"total": len(texts), **{key: 0 for key in BEHAVIOR_KEYS}}
    for text in texts:
        rates = summarize_template_behavior([text])
        for key in BEHAVIOR_KEYS:
            counts[key] += int(rates[f"{key}_rate"] > 0.5)
    return counts
