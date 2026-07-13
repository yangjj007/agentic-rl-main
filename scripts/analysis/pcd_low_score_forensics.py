#!/usr/bin/env python3
"""PCD low-score forensic summaries for existing no-visual runs."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RUNS = [
    (
        "pcd_bs1_aligned",
        Path("outputs/test-fast/pcd-no-visual/pcd_no_visual_aligned_4epoch/deplot_no_vs_opd_pcd"),
        Path("outputs/test-fast/logs/pcd_no_visual_pcd_no_visual_aligned_4epoch/deplot_no_vs_opd_pcd"),
        Path("outputs/test-fast/pcd-low-score-forensics/eval_sweep_aligned/summary.csv"),
    ),
    (
        "pcd_bs8_staged",
        Path("outputs/test-fast/pcd-no-visual/pcd_no_visual_staged/deplot_no_vs_opd_pcd"),
        Path("outputs/test-fast/logs/pcd_no_visual_pcd_no_visual_staged/deplot_no_vs_opd_pcd"),
        Path("outputs/test-fast/pcd-low-score-forensics/eval_sweep_staged/summary.csv"),
    ),
]


TRAIN_KEYS = [
    "reward",
    "rewards/accuracy/mean",
    "reward_std",
    "signal/grpo_zero_loss_rate",
    "signal/group_all_wrong_rate",
    "global_signal/grpo_route_rate",
    "routing/grpo_route_rate",
    "controller/grpo_route_rate",
    "controller/grpo_route_global_fraction",
    "controller/grpo_route_local_fallback_fraction",
    "routing/opd_route_rate",
    "routing/sft_route_rate",
    "routing/teacher_probe_candidate_rate",
    "routing/teacher_probe_correct_rate",
    "routing/teacher_probe_candidate_accuracy",
    "routing/teacher_probe_answer_flag_rate",
    "routing/teacher_probe_parse_fail_rate",
    "teacher_probe/generated_tokens_mean",
    "teacher_probe/generated_tokens_p95",
    "teacher_probe/clipped_rate",
    "completions/degenerate_rate",
    "completions/clipped_ratio",
    "completions/eos_rate",
    "loss/opsd",
    "loss/teacher_traj_fkl",
    "grad_norm",
]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def controller_grpo_route_signal(row: dict[str, Any]) -> tuple[float | None, str]:
    for key in ("global_signal/grpo_route_rate", "routing/grpo_route_rate"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value, key
    return None, ""


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_sort_key(label: str) -> tuple[int, int | str]:
    match = re.search(r"checkpoint-(\d+)", label)
    if match:
        return (0, int(match.group(1)))
    if "final" in label:
        return (1, 10**9)
    return (2, label)


def _normalize_checkpoint_label(raw_label: str) -> str:
    match = re.search(r"(checkpoint-\d+|final_checkpoint)", raw_label)
    if match:
        return match.group(1)
    if raw_label.startswith("eval_"):
        return raw_label.removeprefix("eval_").split("_2026", 1)[0]
    return raw_label


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _attempt_timestamp(source_label: str) -> str:
    matches = re.findall(r"(20\d{6})[_-]?(\d{6})", source_label)
    if not matches:
        return ""
    date, time = matches[-1]
    return f"{date}{time}"


def _eval_attempt_priority(row: dict[str, Any]) -> tuple[int, int, int, int, str, str]:
    processed = _safe_int(row.get("processed"))
    total = _safe_int(row.get("total"))
    threshold = min(total, 2496) if total > 0 else 2496
    exit_status = str(row.get("exit_status", "") or "").strip()
    errors = str(row.get("errors", "") or "").strip()
    source_label = str(row.get("source_label", "") or "")
    return (
        int(_safe_float(row.get("accuracy")) is not None),
        int(processed >= threshold),
        int(not errors and exit_status in ("", "0")),
        processed,
        _attempt_timestamp(source_label),
        source_label,
    )


def parse_eval_log(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    acc = re.findall(r"Current Global Mean Accuracy:\s*([0-9.]+)", text)
    processed = re.findall(r"Global samples processed:\s*(\d+)\s*/\s*(\d+)", text)
    output_types = re.findall(r"Output type counts:\s*(\{[^\n]*\})", text)
    status = re.findall(r"\[eval-exit\].*status=(\d+)", text)
    errors = [
        pattern
        for pattern in ("Traceback", "CUDA out of memory", "RuntimeError", "Failed to load dataset")
        if pattern in text
    ]
    row: dict[str, str] = {
        "accuracy": acc[-1] if acc else "",
        "processed": processed[-1][0] if processed else "",
        "total": processed[-1][1] if processed else "",
        "output_types": output_types[-1] if output_types else "",
        "exit_status": status[-1] if status else "",
        "errors": ";".join(errors),
        "log_path": str(path),
    }
    try:
        counts = ast.literal_eval(row["output_types"]) if row["output_types"] else {}
    except (SyntaxError, ValueError):
        counts = {}
    total = int(row["processed"] or 0)
    for key in ("other", "full_cot", "answer_flag"):
        value = int(counts.get(key, 0) or 0)
        row[f"{key}_rate"] = f"{value / total:.6f}" if total else ""
    return row


def collect_eval_rows(label: str, run_dir: Path, extra_summaries: list[Path] | None = None) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    summaries = [
        run_dir / "eval_chartqa" / "summary.csv",
        run_dir / "eval_sweep" / "summary.csv",
        run_dir / "eval_chartqa_checkpoints" / "summary.csv",
    ]
    summaries.extend(extra_summaries or [])
    for summary in summaries:
        if not summary.exists():
            continue
        with summary.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_label = row.get("label") or row.get("checkpoint") or ""
                checkpoint = _normalize_checkpoint_label(raw_label)
                out = {
                    "run": label,
                    "checkpoint": checkpoint,
                    "source_label": raw_label,
                    "accuracy": row.get("accuracy", ""),
                    "processed": row.get("processed", ""),
                    "total": row.get("total", ""),
                    "exit_status": row.get("exit_status", ""),
                    "errors": row.get("errors", ""),
                    "output_types": row.get("output_types", ""),
                    "other_rate": row.get("other_rate", ""),
                    "full_cot_rate": row.get("full_cot_rate", ""),
                    "answer_flag_rate": row.get("answer_flag_rate", ""),
                    "log_path": row.get("log_path", ""),
                }
                if out["log_path"]:
                    log_path = Path(out["log_path"])
                    if not log_path.is_absolute():
                        log_path = Path.cwd() / log_path
                    if log_path.exists() and not (out["other_rate"] and out["full_cot_rate"]):
                        parsed = parse_eval_log(log_path)
                        out.update({k: v for k, v in parsed.items() if v})
                candidates.setdefault(checkpoint, []).append(out)
    for log in sorted((run_dir / "eval_chartqa").glob("eval_*.log")) + sorted(
        (run_dir / "eval_sweep" / "logs").glob("*.log")
    ):
        parsed = parse_eval_log(log)
        raw = log.stem
        checkpoint = _normalize_checkpoint_label(raw)
        candidates.setdefault(checkpoint, []).append(
            {
                "run": label,
                "checkpoint": checkpoint,
                "source_label": raw,
                **parsed,
            }
        )
    rows = [max(attempts, key=_eval_attempt_priority) for attempts in candidates.values()]
    return sorted(rows, key=lambda row: _checkpoint_sort_key(str(row.get("checkpoint", ""))))


def _token_len(text: str) -> int:
    return len(str(text or "").split())


def collect_candidate_funnel(label: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    records = []
    for path in sorted((run_dir / "teacher_probe_candidates").glob("rank*.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(record)
    scopes = {
        "all": records,
        "all_wrong": [r for r in records if r.get("is_all_wrong_probe_candidate") or r.get("group_all_wrong")],
        "mixed_wrong": [r for r in records if r.get("is_mixed_wrong_probe_candidate")],
    }
    for scope, scope_records in scopes.items():
        n = len(scope_records)
        if not n:
            rows.append({"run": label, "scope": scope, "candidates": 0})
            continue
        teacher_correct = sum(1 for r in scope_records if bool(r.get("teacher_correct")))
        final_opd = sum(1 for r in scope_records if str(r.get("final_route", "")).lower() == "opd")
        final_sft = sum(1 for r in scope_records if str(r.get("final_route", "")).lower().startswith("sft"))
        parse_fail = sum(1 for r in scope_records if bool(r.get("parse_failed")))
        answer_flag = sum(1 for r in scope_records if bool(r.get("has_answer_flag")))
        gold_like = sum(1 for r in scope_records if bool(r.get("teacher_probe_gold_suffix")))
        token_lens = [_token_len(r.get("teacher_output", "")) for r in scope_records]
        rows.append(
            {
                "run": label,
                "scope": scope,
                "candidates": n,
                "teacher_correct": teacher_correct,
                "teacher_correct_rate": teacher_correct / n,
                "final_opd": final_opd,
                "final_opd_rate": final_opd / n,
                "final_sft": final_sft,
                "final_sft_rate": final_sft / n,
                "parse_fail": parse_fail,
                "parse_fail_rate": parse_fail / n,
                "answer_flag": answer_flag,
                "answer_flag_rate": answer_flag / n,
                "teacher_gold_suffix_like": gold_like,
                "teacher_tokens_mean": mean(token_lens) if token_lens else 0.0,
                "teacher_tokens_p95": sorted(token_lens)[min(len(token_lens) - 1, int(0.95 * (len(token_lens) - 1)))],
            }
        )
    return rows


def _metric_rows(log_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            row = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict) and any(
            key in row
            for key in (
                "reward",
                "rewards/accuracy/mean",
                "routing/opd_route_rate",
                "signal/grpo_zero_loss_rate",
            )
        ):
            rows.append(row)
    return rows


def collect_training_windows(label: str, log_dir: Path) -> list[dict[str, Any]]:
    logs = sorted(log_dir.glob("train_opd_7b_dyme_probe_*.log"))
    completed = []
    for path in logs:
        rows = _metric_rows(path)
        if rows and rows[-1].get("epoch", 0) and float(rows[-1].get("epoch", 0)) >= 3.9:
            completed.append((path, rows))
    if not completed and logs:
        completed = [(logs[-1], _metric_rows(logs[-1]))]
    if not completed:
        return []
    log_path, rows = completed[-1]
    for metric_row in rows:
        value, source = controller_grpo_route_signal(metric_row)
        metric_row["controller/grpo_route_rate"] = value
        metric_row["controller/grpo_route_global_fraction"] = float(
            source == "global_signal/grpo_route_rate"
        )
        metric_row["controller/grpo_route_local_fallback_fraction"] = float(
            source == "routing/grpo_route_rate"
        )
    windows = [
        ("first_50", rows[:50]),
        ("mid_50", rows[max(0, len(rows) // 2 - 25) : max(0, len(rows) // 2 - 25) + 50]),
        ("last_50", rows[-50:]),
        ("all", rows),
    ]
    out = []
    for window_name, window_rows in windows:
        row: dict[str, Any] = {
            "run": label,
            "window": window_name,
            "n": len(window_rows),
            "train_log": str(log_path),
        }
        for key in TRAIN_KEYS:
            values = [_safe_float(r.get(key)) for r in window_rows]
            values = [v for v in values if v is not None]
            row[key] = mean(values) if values else ""
        out.append(row)
    return out


def classify_failure(pred: str, gold: str) -> str:
    text = (pred or "").lower()
    gold_text = (gold or "").lower()
    if "visual facts" in text or "deplot" in text:
        return "deplot_or_visual_fact_leak"
    if re.search(r"\d+\.\d{6,}", text):
        return "long_decimal"
    if any(word in text or word in gold_text for word in ("ratio", "average", "difference", "sum", "%")):
        return "numeric_reasoning"
    if text.count("answer:") != 1:
        return "answer_format"
    return "other_wrong"


def collect_eval_failures(label: str, run_dir: Path, limit_per_type: int = 12) -> list[dict[str, Any]]:
    logs = sorted((run_dir / "eval_chartqa").glob("eval_final_checkpoint_*.log"))
    if not logs:
        logs = sorted((run_dir / "eval_sweep" / "logs").glob("final_checkpoint.log"))
    if not logs:
        return []
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if "######" not in line:
            continue
        parts = line.split("######")
        if len(parts) < 3:
            continue
        pred = parts[0].strip()
        gold = parts[1].strip()
        score_text = parts[2].strip().lower()
        if score_text.startswith("true"):
            continue
        category = classify_failure(pred, gold)
        if counts[category] >= limit_per_type:
            continue
        counts[category] += 1
        rows.append(
            {
                "run": label,
                "category": category,
                "prediction_preview": pred[:240].replace("\n", "\\n"),
                "gold": gold[:120],
                "eval_log": str(logs[-1]),
            }
        )
    return rows


def load_run_config(label: str, run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.json"
    if not path.exists():
        return {"run": label}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    opsd = cfg.get("opsd", {})
    probe = opsd.get("teacher_probe", {})
    visual = opsd.get("visual_supervision", {})
    loss = opsd.get("loss", {})
    return {
        "run": label,
        "num_train_epochs": cfg.get("training", {}).get("dyme_args", {}).get("num_train_epochs"),
        "probe_batch_size": probe.get("batch_size"),
        "probe_all_wrong_after_step": probe.get("probe_all_wrong_after_step"),
        "probe_max_per_batch": probe.get("max_per_batch"),
        "loss_type": loss.get("loss_type"),
        "opsd_weight": loss.get("opsd_weight"),
        "variance_adaptive": loss.get("variance_adaptive"),
        "visual_enabled": visual.get("enabled"),
        "visual_prefetch_ic": visual.get("prefetch_ic"),
        "train_dataset": cfg.get("dataset", {}).get("train_dataset"),
    }


def build_report(
    out_dir: Path,
    config_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    funnel_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    anchor_dirs: list[Path],
) -> None:
    best_by_run = {}
    for row in eval_rows:
        acc = _safe_float(row.get("accuracy"))
        if acc is None:
            continue
        current = best_by_run.get(row["run"])
        if current is None or acc > float(current["accuracy"]):
            best_by_run[row["run"]] = row
    lines = ["# PCD Low-Score Forensics", ""]
    lines.append("## Checkpoint Sweep")
    if best_by_run:
        for run, row in sorted(best_by_run.items()):
            lines.append(
                f"- `{run}` best `{row.get('checkpoint')}` accuracy={float(row['accuracy']):.4f}; "
                f"final rows are in `checkpoint_accuracy.csv`."
            )
    else:
        lines.append("- No completed checkpoint eval rows found yet.")
    lines.extend(["", "## Key Configs"])
    for row in config_rows:
        lines.append(
            f"- `{row['run']}`: probe_batch_size={row.get('probe_batch_size')}, "
            f"all_wrong_after_step={row.get('probe_all_wrong_after_step')}, "
            f"visual_prefetch_ic={row.get('visual_prefetch_ic')}, loss={row.get('loss_type')}."
        )
    lines.extend(["", "## Rescue Funnel"])
    for row in funnel_rows:
        if row.get("scope") not in ("all_wrong", "mixed_wrong"):
            continue
        n = int(row.get("candidates") or 0)
        if not n:
            continue
        lines.append(
            f"- `{row['run']}` `{row['scope']}`: n={n}, "
            f"teacher_correct={float(row.get('teacher_correct_rate', 0)):.3f}, "
            f"OPD={float(row.get('final_opd_rate', 0)):.3f}, "
            f"parse_fail={float(row.get('parse_fail_rate', 0)):.3f}."
        )
    lines.extend(["", "## Training Signal"])
    for row in train_rows:
        if row.get("window") != "last_50":
            continue
        lines.append(
            f"- `{row['run']}` last_50: acc_reward={float(row.get('rewards/accuracy/mean') or 0):.4f}, "
            f"grpo_zero={float(row.get('signal/grpo_zero_loss_rate') or 0):.3f}, "
            f"all_wrong={float(row.get('signal/group_all_wrong_rate') or 0):.3f}, "
            f"opd_route={float(row.get('routing/opd_route_rate') or 0):.3f}."
        )
    lines.extend(["", "## Anchor Status"])
    if anchor_dirs:
        for path in anchor_dirs:
            lines.append(f"- Found candidate no-PCD anchor: `{path}`")
    else:
        lines.append("- No same-protocol `deplot_no_vs_opd` 4epoch anchor found under `outputs/test-fast`.")
        lines.append("- To fill this gap, run: `bash scripts/test/run_pcd_forensics_anchor.sh`.")
    lines.extend(["", "## Failure Samples"])
    by_category = Counter(row["category"] for row in failure_rows)
    if by_category:
        for category, count in by_category.most_common():
            lines.append(f"- {category}: {count} sampled rows in `eval_failure_samples.csv`.")
    else:
        lines.append("- No final eval failure samples parsed.")
    lines.append("")
    out_dir.joinpath("pcd_low_score_forensics.md").write_text("\n".join(lines), encoding="utf-8")


def make_checkpoint_accuracy_plot(out_dir: Path, eval_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out_dir.joinpath("checkpoint_accuracy_plot.skipped.txt").write_text(
            f"matplotlib unavailable: {exc}\n",
            encoding="utf-8",
        )
        return
    series: dict[str, list[tuple[int, float, str]]] = {}
    for row in eval_rows:
        acc = _safe_float(row.get("accuracy"))
        if acc is None:
            continue
        checkpoint = str(row.get("checkpoint") or "")
        match = re.search(r"checkpoint-(\d+)", checkpoint)
        step = int(match.group(1)) if match else 588 if checkpoint == "final_checkpoint" else 10**9
        series.setdefault(str(row.get("run")), []).append((step, acc, checkpoint))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, points in sorted(series.items()):
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=run)
        for x, y, checkpoint in points:
            if checkpoint == "final_checkpoint":
                ax.annotate("final", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_title("PCD Checkpoint Accuracy Sweep")
    ax.set_xlabel("global step")
    ax.set_ylabel("ChartQA accuracy")
    ax.set_ylim(0.35, max(0.62, max(y for points in series.values() for _, y, _ in points) + 0.03))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_accuracy_curve.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/test-fast/pcd-low-score-forensics")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL:RUN_DIR:LOG_DIR[:EVAL_SUMMARY]",
        help="Add/override a run tuple. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs = []
    if args.run:
        for spec in args.run:
            parts = spec.split(":")
            if len(parts) not in (3, 4):
                raise SystemExit(f"bad --run spec: {spec}")
            extra = Path(parts[3]) if len(parts) == 4 else None
            runs.append((parts[0], Path(parts[1]), Path(parts[2]), extra))
    else:
        runs = DEFAULT_RUNS

    config_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for item in runs:
        label, run_dir, log_dir = item[:3]
        extra_summary = item[3] if len(item) > 3 else None
        config_rows.append(load_run_config(label, run_dir))
        eval_rows.extend(collect_eval_rows(label, run_dir, [extra_summary] if extra_summary else []))
        funnel_rows.extend(collect_candidate_funnel(label, run_dir))
        train_rows.extend(collect_training_windows(label, log_dir))
        failure_rows.extend(collect_eval_failures(label, run_dir))

    anchor_dirs = sorted(Path("outputs/test-fast").glob("**/deplot_no_vs_opd"))

    _write_csv(
        out_dir / "run_config_summary.csv",
        config_rows,
        [
            "run",
            "num_train_epochs",
            "probe_batch_size",
            "probe_all_wrong_after_step",
            "probe_max_per_batch",
            "loss_type",
            "opsd_weight",
            "variance_adaptive",
            "visual_enabled",
            "visual_prefetch_ic",
            "train_dataset",
        ],
    )
    _write_csv(
        out_dir / "checkpoint_accuracy.csv",
        eval_rows,
        [
            "run",
            "checkpoint",
            "source_label",
            "accuracy",
            "processed",
            "total",
            "exit_status",
            "errors",
            "other_rate",
            "full_cot_rate",
            "answer_flag_rate",
            "output_types",
            "log_path",
        ],
    )
    _write_csv(
        out_dir / "route_funnel.csv",
        funnel_rows,
        [
            "run",
            "scope",
            "candidates",
            "teacher_correct",
            "teacher_correct_rate",
            "final_opd",
            "final_opd_rate",
            "final_sft",
            "final_sft_rate",
            "parse_fail",
            "parse_fail_rate",
            "answer_flag",
            "answer_flag_rate",
            "teacher_gold_suffix_like",
            "teacher_tokens_mean",
            "teacher_tokens_p95",
        ],
    )
    _write_csv(out_dir / "training_windows.csv", train_rows, ["run", "window", "n", *TRAIN_KEYS, "train_log"])
    _write_csv(
        out_dir / "eval_failure_samples.csv",
        failure_rows,
        ["run", "category", "prediction_preview", "gold", "eval_log"],
    )
    make_checkpoint_accuracy_plot(out_dir, eval_rows)
    build_report(out_dir, config_rows, eval_rows, funnel_rows, train_rows, failure_rows, anchor_dirs)
    print(f"Wrote forensic report under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
