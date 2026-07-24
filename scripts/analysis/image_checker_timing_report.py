#!/usr/bin/env python
"""Summarize OPD image-checker smoke timing from logs and visual artifacts."""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Any


SUMMARY_KEYS = (
    "visual/ic_latency_ms",
    "visual/checker_latency_ms",
    "visual/refiner_latency_ms",
    "visual/ic_calls",
    "visual/checker_calls",
    "visual/refiner_calls",
    "visual/teacher_batch_calls",
    "visual/ic_batch_calls",
    "visual/checker_batch_calls",
    "visual/refiner_batch_calls",
    "visual/checker_high",
    "visual/checker_medium",
    "visual/checker_low",
    "visual/checker_parse_failure",
    "visual/checker_image_missing",
    "visual/checker_aux_evidence_used",
    "visual/fallback_checker",
    "visual/fallback_refiner",
)

_STEP_RE = re.compile(r"step_(\d+)")
_COMPLETION_MODE_COUNTS_RE = re.compile(r"completion_mode_counts=(\{.*?\})(?:\s*\||$)")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _ratio(num: float, den: float) -> float:
    den = _as_float(den)
    if den <= 0.0:
        return 0.0
    return round(_as_float(num) / den, 4)


def _parse_visual_batch_line(line: str) -> dict[str, float] | None:
    if "[VISUAL-BATCH]" not in line or "generate_summary" not in line:
        return None
    fields: dict[str, float] = {}
    for part in line.split(" | ")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key in SUMMARY_KEYS:
            fields[key] = _as_float(value.strip())
    return fields


def parse_log_files(log_files: Iterable[Path]) -> list[dict[str, float]]:
    batches: list[dict[str, float]] = []
    for path in log_files:
        if not path or not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parsed = _parse_visual_batch_line(line)
                if parsed is not None:
                    batches.append(parsed)
    return batches


def parse_trainer_completion_modes(log_files: Iterable[Path]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in log_files:
        if not path or not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _COMPLETION_MODE_COUNTS_RE.search(line)
                if not match:
                    continue
                raw = match.group(1)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        payload = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        continue
                if not isinstance(payload, dict):
                    continue
                for key, value in payload.items():
                    counts[str(key)] += int(_as_float(value))
    return dict(sorted(counts.items()))


def _step_from_path(path: Path) -> int:
    for parent in (path.parent, *path.parents):
        match = _STEP_RE.fullmatch(parent.name)
        if match:
            return int(match.group(1))
    return -1


def _score_label(score: Any) -> str:
    value = _as_float(score)
    if value >= 0.75:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def _artifact_files(output_dir: Path | None) -> list[Path]:
    if output_dir is None:
        return []
    root = output_dir / "visual_supervision"
    if not root.exists():
        return []
    return sorted(root.glob("step_*/rank*.jsonl"))


def parse_artifacts(output_dir: Path | None) -> dict[str, Any]:
    checker_labels: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    route_x_checker: Counter[str] = Counter()
    checker_by_sample: dict[tuple[int, int], str] = {}
    false_high = 0
    false_low = 0
    checker_rows = 0
    route_rows = 0

    for path in _artifact_files(output_dir):
        step = _step_from_path(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = str(row.get("kind", ""))
                sample_idx = int(_as_float(row.get("sample_idx"), -1))
                key = (step, sample_idx)
                if kind == "checker":
                    label = str(row.get("label") or _score_label(row.get("score"))).lower()
                    if label == "image_missing":
                        label = "low"
                    checker_by_sample[key] = label
                    checker_labels[label] += 1
                    checker_rows += 1
                elif kind == "route":
                    route = str(row.get("route") or "unknown").lower()
                    label = checker_by_sample.get(key) or _score_label(row.get("checker_score"))
                    routes[route] += 1
                    route_x_checker[f"{route}:{label}"] += 1
                    route_rows += 1
                    answer = _as_float(row.get("answer_reward"))
                    if label == "high" and answer <= 0.0:
                        false_high += 1
                    if label == "low" and answer > 0.0:
                        false_low += 1

    return {
        "artifact_checker_rows": checker_rows,
        "artifact_route_rows": route_rows,
        "checker_labels": dict(sorted(checker_labels.items())),
        "routes": dict(sorted(routes.items())),
        "route_x_checker": dict(sorted(route_x_checker.items())),
        "false_high_count": false_high,
        "false_low_count": false_low,
    }


def build_report(*, log_files: Iterable[Path], output_dir: Path | None = None) -> dict[str, Any]:
    log_file_list = [Path(path) for path in log_files]
    batches = parse_log_files(log_file_list)
    report: dict[str, Any] = {"log_batches": len(batches)}
    for key in SUMMARY_KEYS:
        report[key] = round(sum(_as_float(batch.get(key)) for batch in batches), 4)

    report["derived/checker_ms_per_call"] = _ratio(
        report["visual/checker_latency_ms"],
        report["visual/checker_calls"],
    )
    report["derived/checker_ms_per_batch"] = _ratio(
        report["visual/checker_latency_ms"],
        report["visual/checker_batch_calls"],
    )
    report["derived/ic_ms_per_call"] = _ratio(
        report["visual/ic_latency_ms"],
        report["visual/ic_calls"],
    )
    report["derived/refiner_ms_per_call"] = _ratio(
        report["visual/refiner_latency_ms"],
        report["visual/refiner_calls"],
    )
    report["trainer_completion_modes"] = parse_trainer_completion_modes(log_file_list)
    report.update(parse_artifacts(output_dir))
    return report


def _fmt_counter(mapping: dict[str, int]) -> str:
    if not mapping:
        return "none"
    return ",".join(f"{key}={value}" for key, value in sorted(mapping.items()))


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            "[IMAGE-CHECKER-TIMING] "
            f"log_batches={int(report.get('log_batches', 0))} "
            f"checker_latency_ms={report.get('visual/checker_latency_ms', 0.0)} "
            f"checker_calls={report.get('visual/checker_calls', 0.0)} "
            f"checker_batch_calls={report.get('visual/checker_batch_calls', 0.0)} "
            f"checker_ms_per_call={report.get('derived/checker_ms_per_call', 0.0)} "
            f"checker_ms_per_batch={report.get('derived/checker_ms_per_batch', 0.0)}"
        ),
        (
            "[IMAGE-CHECKER-TIMING] "
            f"ic_latency_ms={report.get('visual/ic_latency_ms', 0.0)} "
            f"ic_calls={report.get('visual/ic_calls', 0.0)} "
            f"ic_batch_calls={report.get('visual/ic_batch_calls', 0.0)} "
            f"ic_ms_per_call={report.get('derived/ic_ms_per_call', 0.0)} "
            f"refiner_latency_ms={report.get('visual/refiner_latency_ms', 0.0)} "
            f"refiner_calls={report.get('visual/refiner_calls', 0.0)} "
            f"refiner_batch_calls={report.get('visual/refiner_batch_calls', 0.0)} "
            f"refiner_ms_per_call={report.get('derived/refiner_ms_per_call', 0.0)}"
        ),
        (
            "[IMAGE-CHECKER-TIMING] "
            f"checker_labels={_fmt_counter(report.get('checker_labels', {}))} "
            f"routes={_fmt_counter(report.get('routes', {}))} "
            f"trainer_completion_modes={_fmt_counter(report.get('trainer_completion_modes', {}))} "
            f"route_x_checker={_fmt_counter(report.get('route_x_checker', {}))}"
        ),
        (
            "[IMAGE-CHECKER-TIMING] "
            f"parse_failure={report.get('visual/checker_parse_failure', 0.0)} "
            f"image_missing={report.get('visual/checker_image_missing', 0.0)} "
            f"aux_used={report.get('visual/checker_aux_evidence_used', 0.0)} "
            f"fallback_checker={report.get('visual/fallback_checker', 0.0)} "
            f"false_high_count={report.get('false_high_count', 0)} "
            f"false_low_count={report.get('false_low_count', 0)}"
        ),
    ]
    return "\n".join(lines)


def _discover_logs(log_dir: Path | None, pattern: str) -> list[Path]:
    if log_dir is None or not log_dir.exists():
        return []
    return sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", action="append", default=[], help="Training log file. Repeatable.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to scan when --log-file is absent.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output_dir containing visual_supervision.")
    parser.add_argument(
        "--pattern",
        default="train_opd_7b_dyme_probe_image_checker*.log",
        help="Log glob used with --log-dir.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    log_files = [Path(p) for p in args.log_file]
    if not log_files:
        log_files = _discover_logs(args.log_dir, args.pattern)
    report = build_report(log_files=log_files, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
