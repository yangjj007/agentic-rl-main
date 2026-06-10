#!/usr/bin/env python3
"""Offline degeneration attribution report from TriMode training logs."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_kv_line(line: str) -> dict:
    out: dict = {}
    for m in re.finditer(r"(\w+)=([^|]+?)(?=\s*\||\s*$)", line):
        k, v = m.group(1).strip(), m.group(2).strip()
        try:
            if v.startswith("[") or v.startswith("{"):
                out[k] = json.loads(v.replace("'", '"'))
            elif re.match(r"^-?\d+\.\d+([eE][+-]?\d+)?$", v):
                out[k] = float(v)
            elif re.match(r"^-?\d+$", v):
                out[k] = int(v)
            else:
                out[k] = v
        except (json.JSONDecodeError, ValueError):
            out[k] = v
    return out


def load_log(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_metrics(text: str) -> list[dict]:
    metrics = []
    for m in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            metrics.append(ast.literal_eval(m.group()))
        except (SyntaxError, ValueError):
            pass
    return metrics


def parse_health_generate(text: str) -> list[tuple[int, dict]]:
    rows = []
    pat = re.compile(
        r"\[OPSD-HEALTH\][^\n]*\[global_step=(\d+)\]\[generate\] batch health \| (.+)"
    )
    for m in pat.finditer(text):
        rows.append((int(m.group(1)), parse_kv_line(m.group(2))))
    return rows


def parse_health_step(text: str) -> list[tuple[int, dict]]:
    rows = []
    pat = re.compile(
        r"\[OPSD-HEALTH\][^\n]*\[global_step=(\d+)\]\[step\] step summary \| (.+)"
    )
    for m in pat.finditer(text):
        rows.append((int(m.group(1)), parse_kv_line(m.group(2))))
    return rows


def parse_health_alerts(text: str) -> list[tuple[int, str, dict]]:
    rows = []
    pat = re.compile(
        r"\[OPSD-HEALTH\][^\n]*\[global_step=(\d+)\]\[ALERT\] (\w+) \| (.+)"
    )
    for m in pat.finditer(text):
        rows.append((int(m.group(1)), m.group(2), parse_kv_line(m.group(3))))
    return rows


def parse_detail_health(text: str) -> list[tuple[int, str, dict]]:
    rows = []
    pat = re.compile(
        r"\[OPSD-DETAIL\][^\n]*\[step=(\d+)\]\[every=\d+\]\[health\] ([^|]+) \| (.+)"
    )
    for m in pat.finditer(text):
        rows.append((int(m.group(1)), m.group(2).strip(), parse_kv_line(m.group(3))))
    return rows


def detect_step1_collapse(metrics: list[dict]) -> str | None:
    if len(metrics) < 2:
        return None
    d0, d1 = metrics[0], metrics[1]
    clip0 = d0.get("completions/clipped_ratio", 0)
    clip1 = d1.get("completions/clipped_ratio", 0)
    if clip0 < 0.2 and clip1 > 0.8:
        return (
            f"step 1 collapse: clipped {clip0:.2f} -> {clip1:.2f}; "
            f"format {d0.get('rewards/format/mean', 'NA')} -> {d1.get('rewards/format/mean', 'NA')}"
        )
    return None


def summarize_alerts(alerts: list[tuple[int, str, dict]]) -> Counter:
    return Counter(code for _, code, _ in alerts)


def build_report(text: str, baseline_text: str | None = None) -> str:
    metrics = parse_metrics(text)
    health_gen = parse_health_generate(text)
    health_step = parse_health_step(text)
    alerts = parse_health_alerts(text)
    detail = parse_detail_health(text)

    lines = ["# TriMode Degeneration Report", ""]
    lines.append(f"- Metric steps parsed: {len(metrics)}")
    lines.append(f"- Health generate lines: {len(health_gen)}")
    lines.append(f"- Health step summaries: {len(health_step)}")
    lines.append(f"- Health alerts: {len(alerts)}")
    lines.append(f"- Detail health bundles: {len(detail)}")
    lines.append("")

    collapse = detect_step1_collapse(metrics)
    if collapse:
        lines.append("## Step 1 collapse")
        lines.append(f"- {collapse}")
        lines.append("")

    if alerts:
        lines.append("## Alert summary")
        for code, count in summarize_alerts(alerts).most_common():
            first_step = min(s for s, c, _ in alerts if c == code)
            lines.append(f"- `{code}`: {count}x (first at step {first_step})")
        lines.append("")

    if health_gen:
        lines.append("## Generation health timeline (sampled)")
        for step, fields in health_gen:
            if step <= 20 or step % 25 == 0:
                lines.append(
                    f"- step {step}: degenerate={fields.get('degenerate_rate', 'NA')} "
                    f"clip={fields.get('clipped_rate', 'NA')} eos={fields.get('eos_rate', 'NA')} "
                    f"alerts={fields.get('alerts', 'none')}"
                )
        lines.append("")

    qi_count = len(re.findall(r"其其其", text))
    lines.append("## CJK repeat in log")
    lines.append(f"- `其其其` occurrences: {qi_count}")
    lines.append("")

    if baseline_text:
        base_metrics = parse_metrics(baseline_text)
        base_alerts = len(parse_health_alerts(baseline_text))
        lines.append("## Baseline comparison")
        lines.append(f"- Baseline metric steps: {len(base_metrics)}")
        lines.append(f"- Baseline alerts: {base_alerts} vs current: {len(alerts)}")
        if metrics and base_metrics:
            m_cur = metrics[min(10, len(metrics) - 1)]
            m_base = base_metrics[min(10, len(base_metrics) - 1)]
            lines.append(
                f"- At ~step 10: current clip={m_cur.get('completions/clipped_ratio', 'NA')} "
                f"vs baseline {m_base.get('completions/clipped_ratio', 'NA')}"
            )
        lines.append("")

    hints: list[str] = []
    for _, msg, fields in detail:
        if "correlation" in msg and "root_cause_hints" in fields:
            h = fields["root_cause_hints"]
            if isinstance(h, list):
                hints.extend(str(x) for x in h if x != "none")
    if hints:
        lines.append("## Root cause hints (from DETAIL health)")
        for h in dict.fromkeys(hints):
            lines.append(f"- {h}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Degeneration report from training log")
    parser.add_argument("log", nargs="?", default=None, help="Training log path")
    parser.add_argument("--baseline", default=None, help="Optional baseline log for comparison")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    log_path = args.log or (sys.argv[1] if len(sys.argv) > 1 else "train_trimode.log")
    text = load_log(log_path)
    baseline_text = load_log(args.baseline) if args.baseline else None

    report = build_report(text, baseline_text)
    if args.json:
        payload = {
            "log": log_path,
            "metrics_count": len(parse_metrics(text)),
            "alerts": parse_health_alerts(text),
            "report_md": report,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
