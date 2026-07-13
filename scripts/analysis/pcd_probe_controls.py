#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pcd_artifact_core import (
    collect_run_data,
    fmt_rate,
    mean_or_none,
    read_manifest,
    write_csv,
)


VALID_MODES = ("teacher_only", "completion_conditioned", "shuffled_completion")
CSV_FIELDS = [
    "control",
    "n",
    "teacher_correct_rate",
    "parse_fail_rate",
    "placeholder_rate",
    "generated_tokens_mean",
    "deplot_real_rate",
    "status",
]


def _mode_list(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in VALID_MODES]
    if unknown:
        raise ValueError(f"unknown probe-control mode(s): {', '.join(unknown)}")
    return modes


def _candidate_records(manifest_path: Path, variant: str, max_samples: int) -> list[dict[str, Any]]:
    manifest_rows = read_manifest(manifest_path)
    data = collect_run_data(manifest_rows)
    records = data.get(variant, {}).get("candidates", [])
    if max_samples > 0:
        return records[:max_samples]
    return records


def _student_output(record: dict[str, Any]) -> str:
    for key in ("student_output", "completion", "response", "generated_text", "student_completion"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _evidence(record: dict[str, Any]) -> str:
    for key in ("deplot_evidence", "visual_evidence", "evidence", "deplot_text"):
        value = record.get(key)
        if value:
            return str(value)
    privileged = record.get("privileged") if isinstance(record.get("privileged"), dict) else {}
    evidence = privileged.get("visual_fact_text") or privileged.get("deplot_text")
    return str(evidence or "")


def build_prompt(record: dict[str, Any], mode: str, *, shuffled_record: dict[str, Any] | None = None) -> str:
    question = str(record.get("question", ""))
    evidence = _evidence(record)
    lines = [
        "You are checking a chart-question answer.",
        f"Question: {question}",
    ]
    if evidence:
        lines.append(f"Visual evidence: {evidence}")
    if mode == "completion_conditioned":
        lines.append(f"Student completion: {_student_output(record)}")
    elif mode == "shuffled_completion":
        source = shuffled_record or record
        lines.append(f"Student completion: {_student_output(source)}")
    lines.append("Return the answer and keep reasoning concise.")
    return "\n".join(lines)


def _shuffled_for(records: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if len(records) < 2:
        return None
    return records[(index + 1) % len(records)]


def _deplot_real_rate(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    real = 0
    for record in records:
        privileged = record.get("privileged") if isinstance(record.get("privileged"), dict) else {}
        if privileged.get("visual_fact_deplot_status") == "real":
            real += 1
    return real / len(records)


def _bool_rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if record.get(key) is True) / len(records)


def _completion_conditioned_row(records: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [
        float(record.get("teacher_output_word_count", 0) or 0)
        for record in records
        if record.get("teacher_output_word_count") is not None
    ]
    return {
        "control": "completion_conditioned",
        "n": len(records),
        "teacher_correct_rate": fmt_rate(_bool_rate(records, "teacher_correct")),
        "parse_fail_rate": fmt_rate(_bool_rate(records, "parse_failed")),
        "placeholder_rate": fmt_rate(_bool_rate(records, "teacher_output_is_placeholder")),
        "generated_tokens_mean": fmt_rate(mean_or_none(tokens)),
        "deplot_real_rate": fmt_rate(_deplot_real_rate(records)),
        "status": "from_candidate_log" if records else "missing_candidate_log",
    }


def _missing_row(mode: str, records: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    status = "dry_run_prompt_only" if dry_run else "missing_offline_teacher_run"
    if not records:
        status = "missing_candidate_log"
    return {
        "control": mode,
        "n": len(records),
        "teacher_correct_rate": "",
        "parse_fail_rate": "",
        "placeholder_rate": "",
        "generated_tokens_mean": "",
        "deplot_real_rate": fmt_rate(_deplot_real_rate(records)),
        "status": status,
    }


def _write_prompt_preview(path: Path, records: list[dict[str, Any]], modes: list[str]) -> None:
    preview_path = path.with_name(path.stem + "_prompts.jsonl")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as f:
        for mode in modes:
            for index, record in enumerate(records):
                payload = {
                    "control": mode,
                    "index": index,
                    "question": record.get("question", ""),
                    "prompt": build_prompt(record, mode, shuffled_record=_shuffled_for(records, index)),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _write_booktabs(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Control",
        "n",
        "Recover ↑",
        "Parse Fail ↓",
        "Placeholder ↓",
        "Tokens",
        "DePlot Real ↑",
        "Status",
    ]
    body = []
    for row in rows:
        first = str(row["control"])
        if row["control"] == "completion_conditioned":
            first = "\\rowcolor{gray!10} " + first
        body.append(
            [
                first,
                row["n"],
                row["teacher_correct_rate"],
                row["parse_fail_rate"],
                row["placeholder_rate"],
                row["generated_tokens_mean"],
                row["deplot_real_rate"],
                row["status"],
            ]
        )
    lines = [
        "```latex",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        escaped = [_latex_escape(cell) for cell in row]
        if str(row[0]).startswith("\\rowcolor"):
            escaped[0] = str(row[0])
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_controls(
    *,
    manifest: Path,
    variant: str,
    modes: list[str],
    max_samples: int,
    out: Path,
    dry_run: bool,
) -> None:
    records = _candidate_records(manifest, variant, max_samples)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        if dry_run:
            rows.append(_missing_row(mode, records, dry_run=True))
        elif mode == "completion_conditioned":
            rows.append(_completion_conditioned_row(records))
        else:
            rows.append(_missing_row(mode, records, dry_run=False))
    write_csv(out, rows, CSV_FIELDS)
    _write_booktabs(out.with_suffix(".md"), rows)
    if dry_run:
        _write_prompt_preview(out, records, modes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PCD-OPD recoverability-control table rows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", default="deplot_no_vs_opd_pcd")
    parser.add_argument("--modes", default="teacher_only,completion_conditioned,shuffled_completion")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        modes = _mode_list(args.modes)
    except ValueError as exc:
        parser.error(str(exc))
    run_controls(
        manifest=args.manifest,
        variant=args.variant,
        modes=modes,
        max_samples=args.max_samples,
        out=args.out,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
