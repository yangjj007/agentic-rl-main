#!/usr/bin/env python3
"""Health check for clean no-gold ChartQA teacher-probe evidence."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.chart.deplot_pipeline import has_real_deplot, is_deplot_placeholder


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _answer_candidates(sample: dict[str, Any]) -> list[str]:
    raw = sample.get("answer", sample.get("label", ""))
    values = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for value in values:
        text = _as_text(value).strip()
        text = re.sub(r"(?i)^\s*answer\s*:\s*", "", text).strip()
        text = text.strip(" \t\r\n\"'`.,;:")
        if text:
            out.append(text)
    return out


def _contains_answer(text: str, answer: str) -> bool:
    text = _as_text(text).lower()
    answer = _as_text(answer).lower().strip()
    if not text or not answer:
        return False
    escaped = re.escape(answer)
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text))


def summarize_teacher_evidence_health(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    visual_fact_answer_substring = 0
    deplot_real = 0
    deplot_placeholder = 0
    deplot_missing = 0

    for sample in samples:
        vf_text = "\n".join(
            _as_text(sample.get(key, ""))
            for key in ("visual_fact", "visual_facts", "visual_fact_hint")
        )
        answers = _answer_candidates(sample)
        if answers and any(_contains_answer(vf_text, ans) for ans in answers):
            visual_fact_answer_substring += 1

        deplot = sample.get("visual_fact_deplot")
        if not _as_text(deplot).strip():
            deplot_missing += 1
        elif has_real_deplot(deplot):
            deplot_real += 1
        elif is_deplot_placeholder(deplot):
            deplot_placeholder += 1

    denom = max(total, 1)
    return {
        "total": total,
        "visual_fact_answer_substring": visual_fact_answer_substring,
        "visual_fact_answer_substring_rate": visual_fact_answer_substring / denom,
        "deplot_real": deplot_real,
        "deplot_real_rate": deplot_real / denom,
        "deplot_placeholder": deplot_placeholder,
        "deplot_placeholder_rate": deplot_placeholder / denom,
        "deplot_missing": deplot_missing,
        "deplot_missing_rate": deplot_missing / denom,
        "clean_evidence_present": deplot_real,
        "clean_evidence_present_rate": deplot_real / denom,
    }


def _print_text(stats: dict[str, Any]) -> None:
    print(f"total={stats['total']}")
    print(
        "visual_fact_answer_substring="
        f"{stats['visual_fact_answer_substring']} "
        f"rate={stats['visual_fact_answer_substring_rate']:.4f}"
    )
    print(
        f"deplot_real={stats['deplot_real']} "
        f"rate={stats['deplot_real_rate']:.4f}"
    )
    print(
        f"deplot_placeholder={stats['deplot_placeholder']} "
        f"rate={stats['deplot_placeholder_rate']:.4f}"
    )
    print(
        f"deplot_missing={stats['deplot_missing']} "
        f"rate={stats['deplot_missing_rate']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="ChartQA JSON file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("examples", []))
    if not isinstance(data, list):
        raise SystemExit("input must be a JSON list or a dict with data/examples")

    stats = summarize_teacher_evidence_health(data)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    else:
        _print_text(stats)


if __name__ == "__main__":
    main()
