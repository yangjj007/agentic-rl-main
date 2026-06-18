#!/usr/bin/env python3
"""Summarize OPD/RLSD routing, teacher-probe, and degeneration metrics from logs."""
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from pathlib import Path


def parse_metric_dicts(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            rows.append(ast.literal_eval(match.group()))
        except (SyntaxError, ValueError):
            continue
    return rows


def avg(rows: list[dict], key: str, window: int | None = None) -> float:
    vals = [float(r.get(key, 0.0) or 0.0) for r in (rows[-window:] if window else rows)]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

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
        print(f"- last degenerate: {last.get('completions/degenerate_rate', 'N/A')}")
        print(f"- last OPD teacher call: {last.get('routing/opd_teacher_call_rate', 'N/A')}")
        print(f"- last SFT replaced: {last.get('routing/sft_replaced_ratio', 'N/A')}")
        print(f"- last teacher probe correct: {last.get('routing/teacher_probe_correct_rate', 'N/A')}")

        print(f"\n## Rolling Average (last {args.window})")
        for key in [
            "rewards/accuracy/mean",
            "rewards/format/mean",
            "completions/degenerate_rate",
            "routing/sft_replaced_ratio",
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
