#!/usr/bin/env python3
"""Compare key TriMode training metrics between baseline and new logs."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_metrics(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    metrics = []
    for m in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            metrics.append(ast.literal_eval(m.group()))
        except (SyntaxError, ValueError):
            pass
    return metrics


def alert_counts(path: str) -> dict[str, int]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = defaultdict(int)
    for code in re.findall(r"\[global_step=\d+\]\[ALERT\] (\w+)", text):
        counts[code] += 1
    return dict(counts)


def opsd_mask_mean(path: str) -> float:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    probes = re.findall(r"opsd_mask_true=(\d+) \| opsd_mask_false=(\d+)", text)
    ratios = [int(t) / (int(t) + int(f)) for t, f in probes if int(t) + int(f) > 0]
    return sum(ratios) / len(ratios) if ratios else 0.0


def routing_field_mean(path: str, field: str) -> float:
    """Mean of routing/* fields from trainer log dicts (RLSD health metrics)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    key = f"routing/{field}"
    values = []
    for m in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            row = ast.literal_eval(m.group())
        except (SyntaxError, ValueError):
            continue
        if key in row:
            values.append(float(row[key]))
    return sum(values) / len(values) if values else 0.0


def metric_at(metrics: list[dict], idx: int, key: str, default=0.0):
    if idx >= len(metrics):
        return default
    return metrics[idx].get(key, default)


def summarize(label: str, path: str) -> dict:
    metrics = parse_metrics(path)
    alerts = alert_counts(path)
    return {
        "label": label,
        "path": path,
        "steps": len(metrics),
        "step1_clip": metric_at(metrics, 1, "completions/clipped_ratio"),
        "step1_eos": 1.0 - metric_at(metrics, 1, "completions/clipped_ratio"),  # proxy if eos not in metrics
        "logit_collapse": alerts.get("LOGIT_MODE_COLLAPSE", 0),
        "gen_clip_collapse": alerts.get("GEN_CLIP_COLLAPSE", 0),
        "rl_zero": alerts.get("RL_ZERO_SIGNAL", 0),
        "opsd_mask_mean": opsd_mask_mean(path),
        "opsd_on_correct_rate": routing_field_mean(path, "opsd_on_correct_rate"),
        "privileged_suffix_has_gold_rate": routing_field_mean(path, "privileged_suffix_has_gold_rate"),
        "leakage_pattern_rate": routing_field_mean(path, "leakage_pattern_rate"),
        "late_format": metric_at(metrics, min(200, len(metrics) - 1), "rewards/format/mean"),
        "late_mean_len": metric_at(metrics, min(200, len(metrics) - 1), "completions/mean_length"),
        "late_acc": metric_at(metrics, min(200, len(metrics) - 1), "rewards/accuracy/mean"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two TriMode training logs")
    parser.add_argument("baseline", help="Baseline log (e.g. pre-antidegen run)")
    parser.add_argument("candidate", nargs="?", help="New log to compare (optional)")
    args = parser.parse_args()

    base = summarize("baseline", args.baseline)
    print(f"# TriMode log comparison\n")
    print(f"| metric | {base['label']} |")
    print(f"|--------|----------|")
    print(f"| steps | {base['steps']} |")
    print(f"| step1 clip | {base['step1_clip']:.3f} |")
    print(f"| LOGIT_MODE_COLLAPSE | {base['logit_collapse']} |")
    print(f"| GEN_CLIP_COLLAPSE | {base['gen_clip_collapse']} |")
    print(f"| opsd_mask mean | {base['opsd_mask_mean']:.3f} |")
    print(f"| opsd_on_correct_rate | {base['opsd_on_correct_rate']:.4f} |")
    print(f"| privileged_suffix_has_gold_rate | {base['privileged_suffix_has_gold_rate']:.4f} |")
    print(f"| leakage_pattern_rate | {base['leakage_pattern_rate']:.4f} |")
    print(f"| step~200 format | {base['late_format']:.3f} |")
    print(f"| step~200 acc | {base['late_acc']:.3f} |")
    print(f"| step~200 mean_len | {base['late_mean_len']:.1f} |")

    if not args.candidate:
        print("\n(Tip: pass a second log path to print delta columns.)")
        return 0

    cand = summarize("candidate", args.candidate)
    print(f"\n| metric | {base['label']} | {cand['label']} | delta |")
    print(f"|--------|----------|-----------|-------|")
    for key in (
        "steps",
        "step1_clip",
        "logit_collapse",
        "gen_clip_collapse",
        "opsd_mask_mean",
        "opsd_on_correct_rate",
        "privileged_suffix_has_gold_rate",
        "leakage_pattern_rate",
        "late_format",
        "late_acc",
        "late_mean_len",
    ):
        b, c = base[key], cand[key]
        if isinstance(b, float):
            delta = c - b
            print(f"| {key} | {b:.3f} | {c:.3f} | {delta:+.3f} |")
        else:
            delta = c - b
            print(f"| {key} | {b} | {c} | {delta:+d} |")

    # Success criteria from antidegen plan
    print("\n## Antidegen success checks (candidate vs baseline)")
    checks = [
        ("step1 clip < 1.0", cand["step1_clip"] < 1.0),
        ("LOGIT_MODE_COLLAPSE down >30%", cand["logit_collapse"] < base["logit_collapse"] * 0.7),
        ("opsd_mask mean > 8%", cand["opsd_mask_mean"] > 0.08),
        ("opsd_mask improved", cand["opsd_mask_mean"] > base["opsd_mask_mean"]),
        ("opsd_on_correct_rate == 0 (RLSD)", cand["opsd_on_correct_rate"] < 0.01),
        ("no privileged gold suffix (RLSD)", cand["privileged_suffix_has_gold_rate"] < 0.01),
        ("no leakage patterns", cand["leakage_pattern_rate"] < 0.01),
    ]
    for name, ok in checks:
        print(f"- [{'x' if ok else ' '}] {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
