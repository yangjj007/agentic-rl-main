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
            f"len={d.get('completions/mean_length', 0):.1f} "
            f"opsd={d.get('loss/opsd', 0):.4f} "
            f"mask={d.get('opsd/mask_ratio', 'NA')}"
        )

probe = re.findall(
    r"global_step=(\d+).*opsd_mask_true=(\d+) \| opsd_mask_false=(\d+)", text
)
ratios = [int(t) / 32 for _, t, _ in probe]
print(f"\n=== OPSD mask (rank0 probes, n={len(ratios)}) ===")
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
    if gs in (0, 1, 2, 5, 10, 20, 50, 100, 150, 180, 190, 195, 196) or gs % 50 == 0:
        print(
            f"  step {gs:3d}: eff={float(et):5.0f} repeat={rl:2s} eos={float(eos):.2f}"
        )

# find format collapse
for i, d in enumerate(metrics):
    if d.get("rewards/format/mean", 1) < 0.3 and i > 3:
        print(f"\n=== Format collapse first at idx {i} ===")
        print(f"  format={d.get('rewards/format/mean')} clip={d.get('completions/clipped_ratio')}")
        break

# opsd loss nonzero
nz = [i for i, d in enumerate(metrics) if d.get("loss/opsd", 0) > 1e-6]
print(f"\n=== loss/opsd > 0: {len(nz)}/{len(metrics)} steps ===")
if nz:
    print(f"  first idx={nz[0]}, last idx={nz[-1]}")
    print(f"  sample values: {[round(metrics[i]['loss/opsd'], 4) for i in nz[:8]]}")

# step 50 detail blocks
for step in (0, 50, 100, 150):
    pat = rf"\[step={step}\]\[every=50\]\[routing\] mode routing summary \| ([^\n]+)"
    m = re.search(pat, text)
    if m:
        print(f"\n=== OPSD-DETAIL step {step} routing ===")
        print(" ", m.group(1)[:200])

# 其其 count over time - sample generate decodes
qi_steps = re.findall(r"global_step=(\d+).*decode_skip_special='[^']*其其其", text)
print(f"\n=== Steps with CJK repeat in probe decode: {len(qi_steps)} ===")
if qi_steps:
    print(f"  first={qi_steps[0]} last={qi_steps[-1]}")
