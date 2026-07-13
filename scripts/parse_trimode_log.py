#!/usr/bin/env python3
"""Quick parser for trimode training logs."""
import ast
import re
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "train_trimode_4gpu_20260610_125224.log"
text = open(path, encoding="utf-8", errors="replace").read()

metrics = []
for m in re.finditer(r"\{'loss':[^\n]+\}", text):
    try:
        metrics.append(ast.literal_eval(m.group()))
    except (SyntaxError, ValueError):
        pass

print(f"Total metric lines: {len(metrics)}")
print("\n=== Metrics timeline ===")
for i, d in enumerate(metrics):
    if i % 25 == 0 or i >= len(metrics) - 3:
        print(
            f"  idx={i:4d} format={d.get('rewards/format/mean', 0):.3f} "
            f"acc={d.get('rewards/accuracy/mean', 0):.3f} "
            f"clip={d.get('completions/clipped_ratio', 0):.3f} "
            f"degen={d.get('completions/degenerate_rate', 'NA')} "
            f"len={d.get('completions/mean_length', 0):.1f} "
            f"opsd={d.get('loss/opsd', 0):.4f} "
            f"alerts={d.get('health/alert_count', 'NA')}"
        )

health_gen = re.findall(
    r"\[global_step=(\d+)\]\[generate\] batch health \| .*?degenerate_rate=([\d.]+).*?"
    r"clipped_rate=([\d.]+).*?eos_rate=([\d.]+)",
    text,
)
print(f"\n=== OPSD-HEALTH generate lines (n={len(health_gen)}) ===")
for gs, deg, clip, eos in health_gen:
    gs = int(gs)
    if gs in (0, 1, 2, 5, 10, 14, 20, 50) or gs % 50 == 0:
        print(f"  step {gs:3d}: degen={float(deg):.2f} clip={float(clip):.2f} eos={float(eos):.2f}")

health_alerts = re.findall(
    r"\[global_step=(\d+)\]\[ALERT\] (\w+)",
    text,
)
print(f"\n=== OPSD-HEALTH alerts (n={len(health_alerts)}) ===")
alert_counts = defaultdict(int)
for step, code in health_alerts:
    alert_counts[code] += 1
for code, cnt in sorted(alert_counts.items(), key=lambda x: -x[1]):
    print(f"  {code}: {cnt}x")

probe = re.findall(
    r"global_step=(\d+).*opsd_mask_true=(\d+) \| opsd_mask_false=(\d+)", text
)
ratios = [int(t) / 32 for _, t, _ in probe] if probe else [0]
print(f"\n=== OPSD mask (rank0 probes, n={len(ratios)}) ===")
if ratios:
    print(f"  mean={sum(ratios)/len(ratios):.3f}  min={min(ratios):.3f}  max={max(ratios):.3f}")
    print(f"  zero_mask_steps={sum(1 for r in ratios if r == 0)}/{len(ratios)}")

gen = re.findall(
    r"global_step=(\d+).*effective_tokens_mean=([\d.]+).*"
    r"repeat_loop_count=(\d+).*eos_terminated_rate=([\d.]+)",
    text,
)
print(f"\n=== Generation stats (n={len(gen)}) ===")
for gs, et, rl, eos in gen:
    gs = int(gs)
    if gs in (0, 1, 2, 5, 10, 14, 20, 50, 100, 150, 180, 190, 195, 196) or gs % 50 == 0:
        print(
            f"  step {gs:3d}: eff={float(et):5.0f} repeat={rl:2s} eos={float(eos):.2f}"
        )

for i, d in enumerate(metrics):
    if d.get("rewards/format/mean", 1) < 0.3 and i > 3:
        print(f"\n=== Format collapse first at idx {i} ===")
        print(f"  format={d.get('rewards/format/mean')} clip={d.get('completions/clipped_ratio')}")
        break

if len(metrics) >= 2:
    d0, d1 = metrics[0], metrics[1]
    if d0.get("completions/clipped_ratio", 0) < 0.2 and d1.get("completions/clipped_ratio", 0) > 0.8:
        print("\n=== Step 1 clip collapse detected ===")
        print(f"  step0 clip={d0.get('completions/clipped_ratio')} -> step1 clip={d1.get('completions/clipped_ratio')}")

nz = [i for i, d in enumerate(metrics) if d.get("loss/opsd", 0) > 1e-6]
print(f"\n=== loss/opsd > 0: {len(nz)}/{len(metrics)} steps ===")
if nz:
    print(f"  first idx={nz[0]}, last idx={nz[-1]}")
    print(f"  sample values: {[round(metrics[i]['loss/opsd'], 4) for i in nz[:8]]}")

qi_steps = re.findall(r"global_step=(\d+).*decode_skip_special='[^']*其其其", text)
print(f"\n=== Steps with CJK repeat in probe decode: {len(qi_steps)} ===")
if qi_steps:
    print(f"  first={qi_steps[0]} last={qi_steps[-1]}")

print("\n(Tip: run scripts/degeneration_report.py for full markdown report)")
