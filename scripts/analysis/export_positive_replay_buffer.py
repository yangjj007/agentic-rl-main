#!/usr/bin/env python
"""Export teacher-correct candidates as a clean positive replay buffer.

Candidate logs are used only as a verifier/filter signal. Student-visible
targets are rebuilt from the matched dataset row so truncated or privileged
teacher outputs do not become replay supervision.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opsd_utils.teacher_sft_repair import (
    build_teacher_sft_repair_target,
    teacher_sft_target_quality,
)


DEFAULT_CANDIDATE_GLOB = (
    "outputs/test-fast/pcd-no-visual/"
    "pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/"
    "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short/"
    "teacher_probe_candidates/rank*.jsonl"
)
DEFAULT_DATASET = "data/chartqa/train_medium_vf_full.json"
DEFAULT_OUT_DIR = "outputs/test-fast/positive-replay-buffer/student_hint_short"


def _clean_answer(value: Any) -> str:
    return re.sub(r"(?i)^\s*answer\s*:\s*", "", str(value or "").strip()).strip()


def _norm_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_answer(value)).strip().lower()


def _norm_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _basename(value: Any) -> str:
    return Path(str(value or "")).name


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _preview_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\\n", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class DatasetIndex:
    """Lookup ChartQA rows without trusting candidate source_idx."""

    def __init__(self, records: list[dict[str, Any]]):
        self.exact: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
        self.by_question_answer: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        self.by_question_image: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        for idx, record in enumerate(records):
            question = _norm_question(record.get("question") or record.get("question_wo_prompt"))
            image = _basename(record.get("image"))
            answer = _norm_answer(record.get("answer"))
            if question and image and answer:
                self.exact.setdefault((question, image, answer), (idx, record))
            if question and answer:
                self.by_question_answer.setdefault((question, answer), (idx, record))
            if question and image:
                self.by_question_image.setdefault((question, image), (idx, record))

    def lookup(self, candidate: dict[str, Any]) -> tuple[int, dict[str, Any], str] | None:
        question = _norm_question(candidate.get("question"))
        image = _basename(candidate.get("image"))
        answer = _norm_answer(candidate.get("reference") or candidate.get("answer"))
        if question and image and answer:
            found = self.exact.get((question, image, answer))
            if found is not None:
                return found[0], found[1], "exact"
        if question and answer:
            found = self.by_question_answer.get((question, answer))
            if found is not None:
                return found[0], found[1], "question_answer"
        if question and image:
            found = self.by_question_image.get((question, image))
            if found is not None:
                return found[0], found[1], "question_image"
        return None


def _dataset_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("data", "records", "train"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _candidate_scope(candidate: dict[str, Any]) -> str:
    if candidate.get("group_all_wrong") is True or candidate.get("is_all_wrong_probe_candidate") is True:
        return "all_wrong"
    if candidate.get("is_mixed_wrong_probe_candidate") is True:
        return "mixed_wrong"
    if candidate.get("group_has_correct") is True:
        return "mixed_wrong"
    return "unknown"


def _candidate_files(pattern: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(pattern))


def _load_candidates(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _candidate_files(pattern):
        rows.extend(_iter_jsonl(path))
    return rows


def _make_replay_row(
    *,
    candidate: dict[str, Any],
    sample: dict[str, Any],
    dataset_idx: int,
    match_method: str,
    target_style: str,
) -> dict[str, Any]:
    answer = _clean_answer(sample.get("answer") or candidate.get("reference"))
    raw_text = str(candidate.get("teacher_output") or "")
    target = build_teacher_sft_repair_target(
        raw_text,
        sample=sample,
        reference_answer=answer,
        target_style=target_style,
        sanitize_privileged=True,
    )
    quality = teacher_sft_target_quality(target.text, answer)
    scope = _candidate_scope(candidate)
    return {
        "dataset_idx": dataset_idx,
        "match_method": match_method,
        "question": sample.get("question") or candidate.get("question"),
        "image": sample.get("image") or candidate.get("image"),
        "image_basename": _basename(sample.get("image") or candidate.get("image")),
        "answer": answer,
        "target": target.text,
        "target_style": target_style,
        "target_quality": quality,
        "scope": scope,
        "global_step": candidate.get("global_step"),
        "generation_idx": candidate.get("generation_idx"),
        "source_idx": candidate.get("source_idx"),
        "rank": candidate.get("rank"),
        "route_reason": candidate.get("route_reason"),
        "final_route": candidate.get("final_route"),
        "group_reward_std": candidate.get("group_reward_std"),
        "provider_names": candidate.get("provider_names"),
        "student_correct": candidate.get("student_correct"),
        "teacher_correct": candidate.get("teacher_correct"),
        "teacher_output_preview": _preview_text(candidate.get("teacher_output")),
        "student_output_preview": _preview_text(candidate.get("student_output")),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("target_quality", {}).get(key) is True) / len(rows)


def _summary_row(counters: Counter, rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "candidate_rows": str(counters["candidate_rows"]),
        "teacher_correct_candidates": str(counters["teacher_correct_candidates"]),
        "filtered_teacher_wrong": str(counters["filtered_teacher_wrong"]),
        "filtered_parse_fail": str(counters["filtered_parse_fail"]),
        "filtered_missing_answer_flag": str(counters["filtered_missing_answer_flag"]),
        "missing_dataset_match": str(counters["missing_dataset_match"]),
        "deduplicated": str(counters["deduplicated"]),
        "emitted": str(len(rows)),
        "student_short_rate": f"{_rate(rows, 'student_short_format'):.4f}",
        "answer_only_rate": f"{_rate(rows, 'answer_only_format'):.4f}",
        "full_hint_format_rate": f"{_rate(rows, 'full_hint_format'):.4f}",
        "exact_reference_answer_line_rate": f"{_rate(rows, 'exact_reference_answer_line'):.4f}",
        "privileged_tag_rate": f"{_rate(rows, 'privileged_tag_present'):.4f}",
    }


def _write_summary_csv(path: Path, counters: Counter, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_rows",
        "teacher_correct_candidates",
        "filtered_teacher_wrong",
        "filtered_parse_fail",
        "filtered_missing_answer_flag",
        "missing_dataset_match",
        "deduplicated",
        "emitted",
        "student_short_rate",
        "answer_only_rate",
        "full_hint_format_rate",
        "exact_reference_answer_line_rate",
        "privileged_tag_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_summary_row(counters, rows))


def _write_by_scope_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope[str(row.get("scope") or "unknown")].append(row)
    fieldnames = [
        "scope",
        "emitted",
        "student_short_rate",
        "answer_only_rate",
        "full_hint_format_rate",
        "exact_reference_answer_line_rate",
        "privileged_tag_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scope in sorted(by_scope):
            scoped = by_scope[scope]
            writer.writerow(
                {
                    "scope": scope,
                    "emitted": str(len(scoped)),
                    "student_short_rate": f"{_rate(scoped, 'student_short_format'):.4f}",
                    "answer_only_rate": f"{_rate(scoped, 'answer_only_format'):.4f}",
                    "full_hint_format_rate": f"{_rate(scoped, 'full_hint_format'):.4f}",
                    "exact_reference_answer_line_rate": f"{_rate(scoped, 'exact_reference_answer_line'):.4f}",
                    "privileged_tag_rate": f"{_rate(scoped, 'privileged_tag_present'):.4f}",
                }
            )


def _target_to_sft_hint(target: str) -> str:
    kept: list[str] = []
    for line in str(target or "").splitlines():
        if re.match(r"(?i)^\s*answer\s*:", line):
            continue
        kept.append(line.rstrip())
    return "\n".join(line for line in kept if line.strip()).strip()


def _write_replay_train(path: Path, rows: list[dict[str, Any]]) -> None:
    records = [
        {
            "question": row["question"],
            "image": row["image"],
            "answer": row["answer"],
            "hint": _target_to_sft_hint(row["target"]),
            "target": row["target"],
            "target_style": row["target_style"],
            "source": "teacher_correct_positive_replay",
            "dataset_idx": row["dataset_idx"],
        }
        for row in rows
    ]
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def export_replay_buffer(
    *,
    candidate_glob: str,
    dataset_path: Path,
    target_style: str,
    min_step: int | None = None,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], Counter]:
    dataset = DatasetIndex(_dataset_records(_load_json(dataset_path)))
    counters: Counter = Counter()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for candidate in _load_candidates(candidate_glob):
        counters["candidate_rows"] += 1
        if candidate.get("teacher_correct") is True:
            counters["teacher_correct_candidates"] += 1
        else:
            counters["filtered_teacher_wrong"] += 1
            continue
        if candidate.get("parse_failed") is True:
            counters["filtered_parse_fail"] += 1
            continue
        if candidate.get("has_answer_flag") is False:
            counters["filtered_missing_answer_flag"] += 1
            continue
        if min_step is not None and int(candidate.get("global_step") or 0) < min_step:
            counters["filtered_min_step"] += 1
            continue

        found = dataset.lookup(candidate)
        if found is None:
            counters["missing_dataset_match"] += 1
            continue
        dataset_idx, sample, match_method = found
        answer = _clean_answer(sample.get("answer") or candidate.get("reference"))
        dedupe_key = (dataset_idx, answer)
        if dedupe_key in seen:
            counters["deduplicated"] += 1
            continue
        seen.add(dedupe_key)

        row = _make_replay_row(
            candidate=candidate,
            sample=sample,
            dataset_idx=dataset_idx,
            match_method=match_method,
            target_style=target_style,
        )
        rows.append(row)
        if max_records > 0 and len(rows) >= max_records:
            break

    return rows, counters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-glob", default=DEFAULT_CANDIDATE_GLOB)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--target-style",
        default="student_hint_short",
        choices=("student_hint_short", "student_short", "answer_only", "chartqa_hint"),
    )
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--preview-records", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, counters = export_replay_buffer(
        candidate_glob=args.candidate_glob,
        dataset_path=Path(args.dataset),
        target_style=args.target_style,
        min_step=args.min_step,
        max_records=max(0, int(args.max_records or 0)),
    )

    _write_jsonl(out_dir / "replay.jsonl", rows)
    _write_jsonl(out_dir / "preview.jsonl", rows[: max(0, int(args.preview_records or 0))])
    _write_replay_train(out_dir / "replay_train.json", rows)
    _write_summary_csv(out_dir / "summary.csv", counters, rows)
    _write_by_scope_csv(out_dir / "by_scope.csv", rows)
    metadata = {
        "candidate_glob": args.candidate_glob,
        "dataset": str(args.dataset),
        "target_style": args.target_style,
        "min_step": args.min_step,
        "max_records": args.max_records,
        "outputs": [
            "replay.jsonl",
            "replay_train.json",
            "summary.csv",
            "by_scope.csv",
            "preview.jsonl",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
