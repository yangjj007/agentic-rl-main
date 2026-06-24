#!/usr/bin/env python3
"""Summarize and plot OPD/DyME direct training metrics from logs."""
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
from collections import Counter
from pathlib import Path


DEFAULT_DIRECT_METRICS: tuple[tuple[str, str], ...] = (
    ("rewards/accuracy/mean", "Accuracy reward"),
    ("rewards/format/mean", "Format reward"),
    ("reward", "Total reward"),
    ("loss", "Training loss"),
    ("routing/sft_replaced_ratio", "SFT route ratio"),
    ("routing/grpo_on_correct_rate", "GRPO route ratio"),
    ("routing/opd_teacher_call_rate", "OPD route ratio"),
)


def parse_metric_dicts(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in re.finditer(r"\{[^\n]*\}", text):
        try:
            row = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict) and "loss" in row:
            rows.append(row)
    return rows


def avg(rows: list[dict], key: str, window: int | None = None) -> float:
    vals = [
        float(r[key])
        for r in (rows[-window:] if window else rows)
        if key in r and r[key] is not None
    ]
    return sum(vals) / len(vals) if vals else 0.0


def parse_metric_spec(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_DIRECT_METRICS)
    metrics: list[tuple[str, str]] = []
    for item in raw.split(","):
        key = item.strip()
        if key:
            metrics.append((key, key))
    if not metrics:
        raise ValueError("--metrics must contain at least one metric key")
    return metrics


def parse_compare_item(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"--compare item must be LABEL=PATH, got: {item}")
    label, path = item.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"--compare item must be LABEL=PATH, got: {item}")
    return label, Path(path)


def binned_series(rows: list[dict], metrics: list[tuple[str, str]], interval: int) -> list[dict]:
    if interval < 1:
        raise ValueError("--step-interval must be >= 1")
    series: list[dict] = []
    for start_idx in range(0, len(rows), interval):
        chunk = rows[start_idx : start_idx + interval]
        if not chunk:
            continue
        entry: dict[str, float | int] = {
            "step_start": start_idx + 1,
            "step_end": start_idx + len(chunk),
            "num_rows": len(chunk),
        }
        for key, _label in metrics:
            vals = [float(row[key]) for row in chunk if key in row and row[key] is not None]
            entry[key] = sum(vals) / len(vals) if vals else math.nan
        series.append(entry)
    return series


def write_compare_csv(
    output_path: Path,
    all_series: dict[str, list[dict]],
    metrics: list[tuple[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "step_start", "step_end", "num_rows"] + [key for key, _ in metrics]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, series in all_series.items():
            for row in series:
                out = {field: row.get(field, "") for field in fieldnames}
                out["method"] = method
                writer.writerow(out)


def plot_compare(
    output_path: Path,
    all_series: dict[str, list[dict]],
    metrics: list[tuple[str, str]],
    interval: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.2 * nrows, 4.0)), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (key, label) in zip(axes_flat, metrics):
        for method, series in all_series.items():
            x = [int(row["step_end"]) for row in series if not math.isnan(float(row.get(key, math.nan)))]
            y = [float(row[key]) for row in series if not math.isnan(float(row.get(key, math.nan)))]
            if x and y:
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.4, label=method)
        ax.set_title(label)
        ax.set_xlabel(f"Training step (mean per {interval} steps)")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(labels), 4),
            frameon=False,
        )
    fig.suptitle("Direct training and routing metrics from logs", y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_compare(args: argparse.Namespace) -> int:
    metrics = parse_metric_spec(args.metrics)
    compare_items = [parse_compare_item(item) for item in args.compare]
    all_series: dict[str, list[dict]] = {}
    row_counts: dict[str, int] = {}

    for label, path in compare_items:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = parse_metric_dicts(text)
        row_counts[label] = len(rows)
        all_series[label] = binned_series(rows, metrics, args.step_interval)

    if args.csv_out:
        write_compare_csv(Path(args.csv_out), all_series, metrics)
        print(f"Wrote CSV: {args.csv_out}")
    if args.plot_out:
        plot_compare(Path(args.plot_out), all_series, metrics, args.step_interval)
        print(f"Wrote plot: {args.plot_out}")

    print("# Direct Metric Compare")
    for label, count in row_counts.items():
        print(f"- {label}: parsed metric rows={count}, bins={len(all_series[label])}")
    print(f"- step interval: {args.step_interval}")
    print("- metrics:")
    for key, label in metrics:
        print(f"  - {key} ({label})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple logs with LABEL=PATH items and emit 10-step series/plots.",
    )
    parser.add_argument("--step-interval", type=int, default=10)
    parser.add_argument("--metrics", help="Comma-separated metric keys for --compare mode.")
    parser.add_argument("--csv-out", help="CSV path for --compare mode.")
    parser.add_argument("--plot-out", help="PNG/PDF path for --compare mode.")
    args = parser.parse_args()

    if args.compare:
        return run_compare(args)
    if not args.log_path:
        parser.error("log_path is required unless --compare is used")

    text = Path(args.log_path).read_text(encoding="utf-8", errors="replace")
    rows = parse_metric_dicts(text)
    alerts = Counter(re.findall(r"\[ALERT\] (\w+)", text))
    modes = Counter(re.findall(r"selected_mode='?([A-Z_]+)'?", text))

    print("# OPD/RLSD Routing Analysis\n")
    print(f"- metric rows: {len(rows)}")
    if rows:
        last = rows[-1]
        print(f"- last epoch: {last.get('epoch', 'N/A')}")
        print(f"- last accuracy: {last.get('rewards/accuracy/mean', 'N/A')}")
        print(f"- last format: {last.get('rewards/format/mean', 'N/A')}")
        print(f"- last total reward: {last.get('reward', 'N/A')}")
        print(f"- last loss: {last.get('loss', 'N/A')}")
        print(f"- last OPD teacher call: {last.get('routing/opd_teacher_call_rate', 'N/A')}")
        print(f"- last SFT replaced: {last.get('routing/sft_replaced_ratio', 'N/A')}")
        print(f"- last GRPO-on-correct: {last.get('routing/grpo_on_correct_rate', 'N/A')}")

        print(f"\n## Rolling Average (last {args.window})")
        for key in [
            "rewards/accuracy/mean",
            "rewards/format/mean",
            "reward",
            "loss",
            "routing/sft_replaced_ratio",
            "routing/grpo_on_correct_rate",
            "routing/opd_teacher_call_rate",
            "routing/teacher_probe_candidate_rate",
            "routing/teacher_probe_correct_rate",
            "routing/teacher_probe_wrong_rate",
            "signal/grpo_zero_loss_rate",
            "loss/opsd",
            "loss/teacher_traj_fkl",
        ]:
            print(f"- {key}: {avg(rows, key, args.window):.4f}")

    print("\n## Mode Router Counts")
    for key, value in modes.most_common():
        print(f"- {key}: {value}")

    print("\n## Alerts")
    for key, value in alerts.most_common(10):
        print(f"- {key}: {value}")

    guang_repeat = len(re.findall(r"光{6,}", text))
    goalie_lines = len(re.findall(r"Goalie", text))
    teacher_probe_summaries = len(re.findall(r"teacher answer probe finished", text))
    print("\n## Output Pattern Counts")
    print(f"- guang-repeat lines: {guang_repeat}")
    print(f"- Goalie lines: {goalie_lines}")
    print(f"- teacher probe summaries: {teacher_probe_summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
