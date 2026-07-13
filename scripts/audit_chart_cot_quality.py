#!/usr/bin/env python3
"""Offline G0 audit for deterministic ChartQA full-CoT quality signals."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_utils.chart_cot_verifier import (  # noqa: E402
    normalize_reasoning_template,
    parse_deplot_table,
    summarize_template_diversity,
    verify_chart_cot_trajectory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset must contain a JSON list: {path}")
    return [row for row in data if isinstance(row, dict)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _sample_rows(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), max_samples))
    return [rows[index] for index in indices]


def _dataset_response(row: dict[str, Any]) -> tuple[str, bool]:
    hint = str(row.get("hint") or row.get("visual_fact_hint") or "").strip()
    answer = str(row.get("answer") or "").strip()
    return f"{hint}\nAnswer: {answer}".strip(), True


def _candidate_response(row: dict[str, Any]) -> tuple[str, bool]:
    response = str(row.get("teacher_output") or row.get("response") or "")
    answer_correct = bool(row.get("teacher_correct", row.get("answer_correct", False)))
    return response, answer_correct


def _join_candidates_to_dataset(
    candidates: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_image = {
        str(row.get("image")): row
        for row in dataset_rows
        if str(row.get("image") or "").strip()
    }
    by_question = {
        str(row.get("question") or row.get("question_wo_prompt") or "").strip(): row
        for row in dataset_rows
        if str(row.get("question") or row.get("question_wo_prompt") or "").strip()
    }
    joined: list[dict[str, Any]] = []
    for candidate in candidates:
        image = str(candidate.get("image") or "").strip()
        question = str(candidate.get("question") or candidate.get("question_wo_prompt") or "").strip()
        dataset_row = by_image.get(image) if image else None
        if dataset_row is None and question:
            dataset_row = by_question.get(question)
        joined.append({**(dataset_row or {}), **candidate})
    return joined


def _row_result(row: dict[str, Any], response: str, answer_correct: bool, synthesized: bool) -> dict[str, Any]:
    deplot = row.get("visual_fact_deplot")
    verification = verify_chart_cot_trajectory(response, deplot, answer_correct=answer_correct)
    table = parse_deplot_table(deplot)
    template = normalize_reasoning_template(verification.parsed.reasoning, table)
    return {
        "question": str(row.get("question") or row.get("question_wo_prompt") or ""),
        "reference_answer": str(row.get("answer") or row.get("reference") or ""),
        "response": response,
        "synthesized_answer_line": synthesized,
        "quality": verification.quality,
        "reason_codes": list(verification.reason_codes),
        "structure_valid": verification.parsed.structure_valid,
        "deplot_available": verification.deplot_available,
        "grounded_claims": [asdict(claim) for claim in verification.grounded_claims],
        "reasoning_checks": [asdict(check) for check in verification.reasoning_checks],
        "conclusion_answer": asdict(verification.conclusion_answer),
        "reasoning_template": template,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualities = Counter(row["quality"] for row in rows)
    quality_counts = {quality: qualities.get(quality, 0) for quality in ("Q0", "Q1", "Q2", "Q3")}
    q3_templates = [row["reasoning_template"] for row in rows if row["quality"] == "Q3"]
    other_templates = [row["reasoning_template"] for row in rows if row["quality"] != "Q3"]
    return {
        "sample_count": len(rows),
        "quality_counts": quality_counts,
        "quality_rates": {
            quality: count / max(len(rows), 1) for quality, count in quality_counts.items()
        },
        "structure_valid_rate": sum(bool(row["structure_valid"]) for row in rows) / max(len(rows), 1),
        "deplot_available_rate": sum(bool(row["deplot_available"]) for row in rows) / max(len(rows), 1),
        "templates": {
            "q3": summarize_template_diversity(q3_templates),
            "non_q3": summarize_template_diversity(other_templates),
        },
    }


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    (out_dir / "chart_cot_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "chart_cot_quality_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "chart_cot_quality_conflicts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("quality", "reason_codes", "question", "reference_answer", "response"),
        )
        writer.writeheader()
        for row in rows:
            if row["quality"] != "Q0":
                continue
            writer.writerow(
                {
                    "quality": row["quality"],
                    "reason_codes": ",".join(row["reason_codes"]),
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "response": row["response"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_rows = _load_json(Path(args.dataset))
    if args.teacher_jsonl:
        source_rows = _join_candidates_to_dataset(
            _load_jsonl(Path(args.teacher_jsonl)),
            dataset_rows,
        )
        response_builder = _candidate_response
        synthesized = False
    else:
        source_rows = dataset_rows
        response_builder = _dataset_response
        synthesized = True
    sampled = _sample_rows(source_rows, args.max_samples, args.seed)
    results = []
    for row in sampled:
        response, answer_correct = response_builder(row)
        results.append(_row_result(row, response, answer_correct, synthesized))
    _write_outputs(Path(args.out_dir), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
