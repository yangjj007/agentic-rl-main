from __future__ import annotations

import ast
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "loss_mean": ("loss", "train/loss"),
    "opsd_loss_mean": ("loss/opsd", "opsd_loss"),
    "reward_mean": ("reward", "rewards/total/mean"),
    "accuracy_reward_mean": ("rewards/accuracy/mean", "accuracy_reward_mean"),
    "format_reward_mean": ("rewards/format/mean", "format_reward_mean"),
    "reward_std_mean": ("signal/reward_std_mean", "reward_std"),
    "group_all_wrong_rate": ("signal/group_all_wrong_rate",),
    "group_mixed_rate": ("signal/group_mixed_rate",),
    "reward_std_lt_0_01_rate": ("signal/reward_std_lt_0_01_rate",),
    "reward_std_lt_0_05_rate": ("signal/reward_std_lt_0_05_rate",),
    "reward_std_lt_0_10_rate": ("signal/reward_std_lt_0_10_rate",),
    "grpo_route_rate": (
        "global_signal/grpo_route_rate",
        "routing/grpo_route_rate",
        "routing/grpo_on_correct_rate",
    ),
    "opd_route_rate": (
        "global_signal/opd_route_rate",
        "routing/opd_route_rate",
        "routing/opd_teacher_call_rate",
    ),
    "sft_route_rate": (
        "global_signal/sft_route_rate",
        "routing/sft_route_rate",
        "routing/sft_replaced_ratio",
    ),
    "teacher_probe_candidate_rate": ("routing/teacher_probe_candidate_rate",),
    "teacher_correct_rate": ("routing/teacher_probe_correct_rate", "teacher_probe/teacher_correct_rate"),
    "opsd_effective_weight": ("loss/opsd_effective_weight",),
    "opsd_adaptive_multiplier": ("loss/opsd_adaptive_multiplier",),
    "grpo_zero_loss_rate": ("signal/grpo_zero_loss_rate",),
    "advantage_abs_mean": ("signal/advantage_abs_mean",),
    "completion_clipped_rate": ("completions/clipped_ratio", "completions/clipped_rate"),
    "completion_eos_rate": ("completions/eos_rate",),
    "degenerate_rate": ("completions/degenerate_rate", "degenerate_rate"),
}

COUNT_ALIASES: dict[str, tuple[str, ...]] = {
    "total_completion_count": ("routing/total_completion_count",),
    "wrong_completion_count": ("routing/wrong_completion_count",),
    "probe_candidate_count": ("routing/probe_candidate_count",),
    "teacher_correct_count": ("routing/teacher_correct_count",),
    "opd_route_count": ("routing/opd_route_count",),
    "sft_route_count": ("routing/sft_route_count",),
    "grpo_route_count": ("routing/grpo_route_count",),
}


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def fmt_rate(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.4f}"


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _resolve_path(raw: str, manifest_dir: Path) -> Path | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_path = Path(raw)
    if cwd_path.exists():
        return cwd_path
    return manifest_dir / path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_dir = path.parent
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("enabled", "1")).strip().lower() in {"0", "false", "no"}:
                continue
            resolved = dict(row)
            for key in ("run_dir", "train_log", "eval_log", "config_path"):
                value = _resolve_path(row.get(key, ""), manifest_dir)
                if value is not None:
                    resolved[key] = value
            glob_value = row.get("candidate_log_glob", "")
            resolved["candidate_log_glob"] = str(_resolve_path(glob_value, manifest_dir) or "") if glob_value else ""
            rows.append(resolved)
    return rows


def _parse_mapping_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            value = ast.literal_eval(text)
            return value if isinstance(value, dict) else None
        except (SyntaxError, ValueError):
            return None
    if text.startswith("[") and "{" in text:
        text = text[text.find("{") :]
        try:
            value = ast.literal_eval(text)
            return value if isinstance(value, dict) else None
        except (SyntaxError, ValueError):
            return None
    return None


def read_train_log(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_mapping_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _metric_value(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def count_value(row: dict[str, Any], key: str) -> float | None:
    return _metric_value(row, COUNT_ALIASES[key])


def bin_training_rows(rows: list[dict[str, Any]], *, bin_size: int = 10) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        step_raw = row.get("global_step", row.get("step", row.get("Step", index + 1)))
        step = int(safe_float(step_raw) or (index + 1))
        bucket = (step // bin_size) * bin_size
        buckets.setdefault(bucket, []).append(row)
    out: list[dict[str, Any]] = []
    for bucket in sorted(buckets):
        group = buckets[bucket]
        merged: dict[str, Any] = {"step_bin": bucket}
        for canonical, aliases in METRIC_ALIASES.items():
            values = [value for row in group if (value := _metric_value(row, aliases)) is not None]
            merged[canonical] = mean_or_none(values)
        for canonical, aliases in COUNT_ALIASES.items():
            values = [value for row in group if (value := _metric_value(row, aliases)) is not None]
            merged[canonical] = sum(values) if values else None
        out.append(merged)
    return out


def read_eval_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {}
    match = re.search(r"Current Global Mean Accuracy:\s*([0-9.]+)", text)
    if match:
        out["accuracy"] = float(match.group(1))
    match = re.search(r"Global samples processed:\s*([0-9]+)\s*/\s*([0-9]+)", text)
    if match:
        out["processed"] = int(match.group(1))
        out["total"] = int(match.group(2))
    match = re.search(r"Output type counts:\s*(\{.*?\})", text)
    if match:
        try:
            counts = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            counts = {}
        total = sum(int(value) for value in counts.values()) if counts else 0
        if total:
            out["full_cot_rate"] = int(counts.get("full_cot", 0)) / total
            out["answer_flag_rate"] = int(counts.get("answer_flag", 0)) / total
            out["other_rate"] = int(counts.get("other", 0)) / total
    return out


def read_candidate_logs(pattern: str) -> list[dict[str, Any]]:
    if not pattern:
        return []
    records: list[dict[str, Any]] = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def collect_run_data(manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        variant = row.get("variant", "")
        train_log = row.get("train_log") if isinstance(row.get("train_log"), Path) else None
        eval_log = row.get("eval_log") if isinstance(row.get("eval_log"), Path) else None
        data[variant] = {
            "manifest": row,
            "train": read_train_log(train_log),
            "eval": read_eval_log(eval_log),
            "candidates": read_candidate_logs(str(row.get("candidate_log_glob", ""))),
        }
    return data
