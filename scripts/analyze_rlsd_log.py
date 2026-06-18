#!/usr/bin/env python3
"""Analyze RLSD training log health."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path


def parse_metrics(text: str) -> list[dict]:
    metrics = []
    for m in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            metrics.append(ast.literal_eval(m.group()))
        except (SyntaxError, ValueError):
            pass
    return metrics


def alert_counts(text: str) -> Counter:
    return Counter(re.findall(r"\[ALERT\] (\w+)", text))


def health_snapshots(text: str) -> list[str]:
    return re.findall(r"\[health\] step snapshot \| ([^\n]+)", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    args = parser.parse_args()
    text = Path(args.log_path).read_text(encoding="utf-8", errors="replace")

    alerts = alert_counts(text)
    metrics = parse_metrics(text)
    health = health_snapshots(text)

    print("# RLSD 训练健康分析\n")
    print("## 进度")
    prog = re.findall(r"(\d+)%\|[^|]+\| (\d+)/(\d+)", text)
    if prog:
        pct, cur, total = prog[-1]
        print(f"- 当前: step {cur}/{total} ({pct}%), epoch≈{metrics[-1].get('epoch', '?') if metrics else '?'}")
    print(f"- 已记录 metrics 行数: {len(metrics)}")

    print("\n## 防泄露指标 (RLSD 核心验收)")
    if metrics:
        last = metrics[-1]
        checks = [
            ("opsd_on_correct_rate == 0", last.get("routing/opsd_on_correct_rate", -1) == 0.0),
            ("privileged_suffix_has_gold == 0", last.get("teacher/privileged_suffix_has_gold_rate", -1) == 0.0),
            ("leakage skipped only", last.get("routing/opsd_skipped_leakage", 0) == 0.0),
        ]
        for name, ok in checks:
            print(f"- [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"- routing/opsd_on_correct_rate: {last.get('routing/opsd_on_correct_rate', 'N/A')}")
        print(f"- teacher/privileged_suffix_has_gold_rate: {last.get('teacher/privileged_suffix_has_gold_rate', 'N/A')}")
        print(f"- routing/opd_teacher_call_rate: {last.get('routing/opd_teacher_call_rate', 'N/A')}")

    print("\n## 学习信号 (是否真正在学)")
    if metrics:
        for idx in [0, 50, 100, 200, 500, 1000, len(metrics) - 1]:
            if idx >= len(metrics):
                continue
            m = metrics[idx]
            print(
                f"- step {idx:4d}: acc={m.get('rewards/accuracy/mean', 0):.3f} "
                f"fmt={m.get('rewards/format/mean', 0):.3f} "
                f"len={m.get('completions/mean_length', 0):.1f} "
                f"degen={m.get('completions/degenerate_rate', 0):.3f} "
                f"grpo_zero={m.get('signal/grpo_zero_loss_rate', 0):.3f} "
                f"loss={m.get('loss', 0):.4f}"
            )

    print("\n## ALERT 分布 (Top 10)")
    for k, v in alerts.most_common(10):
        print(f"- {k}: {v}")

    print("\n## 最新 batch 生成样例 (step 1099 附近)")
    samples = re.findall(r"decode_skip_special='([^']{1,80})'", text[-8000:])
    if samples:
        print(f"- 典型输出: {samples[:4]}")

    print("\n## 综合判断")
    if metrics:
        last = metrics[-1]
        degen = last.get("completions/degenerate_rate", 0)
        acc = last.get("rewards/accuracy/mean", 0)
        fmt = last.get("rewards/format/mean", 0)
        grpo_zero = last.get("signal/grpo_zero_loss_rate", 1)
        anti_leak_ok = (
            last.get("routing/opsd_on_correct_rate", 1) == 0
            and last.get("teacher/privileged_suffix_has_gold_rate", 1) == 0
        )
        learning_ok = acc > 0.05 and fmt > 0.3 and degen < 0.5 and grpo_zero < 0.9

        print(f"- 防泄露路由: {'健康' if anti_leak_ok else '异常'}")
        print(f"- 学习效果: {'不健康 (退化/零信号)' if not learning_ok else '尚可'}")
        if degen >= 0.9:
            print("  → 模型已塌缩为极短重复输出 (如 ' Sund\\n')，GRPO 无法介入")
        if grpo_zero >= 0.99:
            print("  → GRPO 全程零梯度，仅靠 self-OPSD 在更新")
        if last.get("routing/sft_replaced_ratio", 0) == 0:
            print("  → 在线 SFT 冷启动未触发 (可能组内总有部分样本非全错)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
