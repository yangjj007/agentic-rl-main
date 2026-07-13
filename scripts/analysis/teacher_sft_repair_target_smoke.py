#!/usr/bin/env python
"""Offline smoke audit for constrained teacher SFT repair targets."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opsd_utils.teacher_sft_repair import (
    build_teacher_sft_repair_target,
    constrain_teacher_sft_repair_target,
    teacher_sft_target_quality,
)


DEFAULT_CANDIDATE_GLOB = (
    "outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_4epoch/"
    "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair/teacher_probe_candidates/rank*.jsonl"
)
DEFAULT_DATASET = "data/chartqa/train_medium_vf_full.json"
DEFAULT_OUT_DIR = "outputs/test-fast/teacher-sft-repair-target-smoke/constrained_chartqa_hint"


def _clean_answer(value: Any) -> str:
    return re.sub(r"(?i)^\s*answer\s*:\s*", "", str(value or "").strip()).strip()


def _norm_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _basename(value: Any) -> str:
    return Path(str(value or "")).name


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class DatasetIndex:
    def __init__(self, records: list[dict[str, Any]]):
        self.exact: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.by_question_answer: dict[tuple[str, str], dict[str, Any]] = {}
        self.by_question_image: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            question = _norm_question(record.get("question") or record.get("question_wo_prompt"))
            image = _basename(record.get("image"))
            answer = _clean_answer(record.get("answer")).lower()
            if question and image and answer:
                self.exact.setdefault((question, image, answer), record)
            if question and answer:
                self.by_question_answer.setdefault((question, answer), record)
            if question and image:
                self.by_question_image.setdefault((question, image), record)

    def lookup(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        question = _norm_question(candidate.get("question"))
        image = _basename(candidate.get("image"))
        answer = _clean_answer(candidate.get("reference") or candidate.get("answer")).lower()
        if question and image and answer:
            found = self.exact.get((question, image, answer))
            if found is not None:
                return found
        if question and answer:
            found = self.by_question_answer.get((question, answer))
            if found is not None:
                return found
        if question and image:
            return self.by_question_image.get((question, image))
        return None


def _load_candidates(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(p) for p in glob.glob(pattern)):
        rows.extend(_iter_jsonl(path))
    return rows


def _sample_rows(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:max_samples]


def _metric_row(kind: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    n = len(rows)
    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key) is True) / max(n, 1)

    token_mean = sum(float(row.get("tokens") or 0) for row in rows) / max(n, 1)
    return {
        "kind": kind,
        "target_style": str(rows[0].get("target_style") or "") if rows else "",
        "n": str(n),
        "full_hint_format_rate": f"{rate('full_hint_format'):.4f}",
        "answer_last_line_rate": f"{rate('answer_last_line'):.4f}",
        "exact_reference_answer_line_rate": f"{rate('exact_reference_answer_line'):.4f}",
        "student_short_rate": f"{rate('student_short_format'):.4f}",
        "answer_only_rate": f"{rate('answer_only_format'):.4f}",
        "privileged_tag_rate": f"{rate('privileged_tag_present'):.4f}",
        "raw_clipped_rate": f"{rate('raw_clipped'):.4f}",
        "fallback_hint_rate": f"{rate('used_fallback_hint'):.4f}",
        "tokens_mean": f"{token_mean:.2f}",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "kind",
        "target_style",
        "n",
        "full_hint_format_rate",
        "answer_last_line_rate",
        "exact_reference_answer_line_rate",
        "student_short_rate",
        "answer_only_rate",
        "privileged_tag_rate",
        "raw_clipped_rate",
        "fallback_hint_rate",
        "tokens_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-glob", default=DEFAULT_CANDIDATE_GLOB)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-teacher-wrong", action="store_true")
    parser.add_argument("--target-styles", default="chartqa_hint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = _json_load(Path(args.dataset))
    if isinstance(dataset_records, dict):
        dataset_records = dataset_records.get("data") or dataset_records.get("records") or []
    index = DatasetIndex(list(dataset_records))
    candidates = _sample_rows(_load_candidates(args.candidate_glob), args.max_samples, args.seed)
    target_styles = [part.strip() for part in str(args.target_styles).split(",") if part.strip()]
    if not target_styles:
        target_styles = ["chartqa_hint"]

    records: list[dict[str, Any]] = []
    missing = 0
    for idx, candidate in enumerate(candidates):
        if not args.include_teacher_wrong and candidate.get("teacher_correct") is False:
            continue
        sample = index.lookup(candidate)
        if sample is None:
            missing += 1
            continue
        answer = _clean_answer(sample.get("answer") or candidate.get("reference"))
        raw_text = str(candidate.get("teacher_output") or "")
        constrained = constrain_teacher_sft_repair_target(
            raw_text,
            sample=sample,
            reference_answer=answer,
            sanitize_privileged=True,
        )
        target_records: list[tuple[str, str, dict[str, Any]]] = []
        if target_styles == ["chartqa_hint"]:
            target_records.append(
                (
                    "constrained_target",
                    constrained.text,
                    {
                        "target_style": "chartqa_hint",
                        "used_fallback_hint": constrained.used_fallback_hint,
                    },
                )
            )
        else:
            for style in target_styles:
                target = build_teacher_sft_repair_target(
                    raw_text,
                    sample=sample,
                    reference_answer=answer,
                    target_style=style,
                    sanitize_privileged=True,
                )
                target_records.append(
                    (
                        f"target_{style}",
                        target.text,
                        {
                            "target_style": style,
                            "used_fallback_hint": target.used_fallback_hint,
                            "student_short_format": target.student_short_format,
                            "answer_only_format": target.answer_only_format,
                        },
                    )
                )
        for kind, text, extra in (
            ("raw_teacher", raw_text, {"used_fallback_hint": False}),
            *target_records,
        ):
            quality = teacher_sft_target_quality(text, answer)
            quality.update(extra)
            records.append(
                {
                    "kind": kind,
                    "row_idx": idx,
                    "question": candidate.get("question") or sample.get("question"),
                    "image_basename": _basename(candidate.get("image") or sample.get("image")),
                    "reference": answer,
                    "text": text,
                    "tokens": len(text.split()),
                    **quality,
                }
            )

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_kind[record["kind"]].append(record)
    ordered_kinds = ["raw_teacher"] + [kind for kind in by_kind.keys() if kind != "raw_teacher"]
    summary_rows = [_metric_row(kind, by_kind[kind]) for kind in ordered_kinds]
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_jsonl(out_dir / "records.jsonl", records)
    metadata = {
        "candidate_rows": len(candidates),
        "record_rows": len(records),
        "missing_dataset_matches": missing,
        "include_teacher_wrong": bool(args.include_teacher_wrong),
        "target_styles": target_styles,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
