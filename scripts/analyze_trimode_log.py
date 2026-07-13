"""Analyze TriMode training log for routing, rewards, generation quality."""
import json
import re
import sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "train_trimode_4gpu_20260609_175823.log"

re_step = re.compile(r"\[step=(\d+)\]\[every=10\]")
re_routing = re.compile(r"\[routing\] mode routing summary \| (.+)")
re_reward = re.compile(r"\[reward\] aggregate reward stats \| (.+)")
re_gen = re.compile(r"\[generation\] completion mask summary \| (.+)")
re_probe = re.compile(
    r"global_step=(\d+) \| .*?"
    r"effective_tokens_mean=([\d.]+) \| .*?"
    r"paren_then_eos_count=(\d+) \| .*?"
    r"one_token_count=(\d+) \| .*?"
    r"eos_terminated_rate=([\d.]+)"
)
re_loss = re.compile(r"\[loss\] GRPO / OPSD loss breakdown \| (.+)")
re_per_sample = re.compile(
    r"\[reward\] per_sample\[(\d+)\] \| group=(\d+) \| format=([\d.]+) \| acc=([\d.]+)"
)
re_completion = re.compile(
    r"\[generation\] sample\[(\d+)\] \| group=(\d+) \| effective_tokens=(\d+) \| has_eos=(\w+) \| text='([^']*)'"
)


def parse_kv(s: str) -> dict:
    d = {}
    for m in re.finditer(r"(\w+)=([^|]+?)(?=\s*\||\s*$)", s):
        k, v = m.group(1), m.group(2).strip()
        try:
            if v.startswith("{") or v.startswith("["):
                d[k] = json.loads(v.replace("'", '"'))
            elif re.match(r"^-?\d+\.\d+([eE][+-]?\d+)?$", v) or re.match(r"^-?\d+\.\d+$", v):
                d[k] = float(v)
            elif re.match(r"^-?\d+$", v):
                d[k] = int(v)
            else:
                d[k] = v
        except (json.JSONDecodeError, ValueError):
            d[k] = v
    return d


def main():
    routing = {}
    rewards = {}
    gens = {}
    losses_by_step = defaultdict(list)
    probes = []
    per_sample_rewards = defaultdict(list)
    sample_completions = defaultdict(list)

    first_ts = last_ts = None
    grpo_routes = sft_routes = 0
    total_debug_routes = 0

    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
            if m:
                if first_ts is None:
                    first_ts = m.group(1)
                last_ts = m.group(1)

            if "completion_mode_counts=" in line and "rank=0/4" in line:
                cm = re.search(r'completion_mode_counts=(\{[^}]+\})', line)
                if cm:
                    total_debug_routes += 1
                    c = json.loads(cm.group(1).replace("'", '"'))
                    grpo_routes += c.get("GRPO", 0)
                    sft_routes += c.get("SFT", 0)

            if "[OPSD-PROBE]" in line and "rank=0/4" in line and "[generate] raw generate summary" in line:
                pm = re_probe.search(line)
                if pm:
                    probes.append(
                        (
                            int(pm.group(1)),
                            float(pm.group(2)),
                            int(pm.group(3)),
                            int(pm.group(4)),
                            float(pm.group(5)),
                        )
                    )
                continue

            if "[OPSD-DETAIL]" not in line or "rank=0/4" not in line:
                continue

            sm = re_step.search(line)
            if not sm:
                continue
            step = int(sm.group(1))

            if "[routing]" in line:
                rm = re_routing.search(line)
                if rm:
                    routing[step] = parse_kv(rm.group(1))
            elif "[reward] aggregate" in line:
                rm = re_reward.search(line)
                if rm:
                    rewards[step] = parse_kv(rm.group(1))
            elif "[reward] per_sample" in line:
                rm = re_per_sample.search(line)
                if rm:
                    per_sample_rewards[step].append(
                        (int(rm.group(1)), float(rm.group(3)), float(rm.group(4)))
                    )
            elif "[generation] completion mask" in line:
                rm = re_gen.search(line)
                if rm:
                    gens[step] = parse_kv(rm.group(1))
            elif "[generation] sample[" in line:
                rm = re_completion.search(line)
                if rm and len(sample_completions[step]) < 4:
                    sample_completions[step].append(rm.group(5)[:120])
            elif "[loss] GRPO / OPSD loss breakdown" in line:
                rm = re_loss.search(line)
                if rm:
                    losses_by_step[step].append(parse_kv(rm.group(1)))

    print("=== TRAINING OVERVIEW ===")
    print(f"Time range: {first_ts} -> {last_ts}")
    print(f"OPSD-DETAIL routing snapshots (rank0): {len(routing)}")
    if routing:
        print(f"Global steps: {min(routing)} - {max(routing)}")
    print(f"OPSD-PROBE generate summaries (rank0): {len(probes)}")
    if probes:
        print(f"Probe global steps: {min(p[0] for p in probes)} - {max(p[0] for p in probes)}")

    print("\n=== ROUTING RATIOS (rank0 DETAIL, every 10 steps) ===")
    print(f"{'step':>6} {'OPSD':>7} {'GRPO':>7} {'SFT':>7} {'has_correct':>16} {'mask_ratio':>10}")
    steps_sorted = sorted(routing.keys())
    sample_idx = list(range(0, len(steps_sorted), max(1, len(steps_sorted) // 12)))
    if steps_sorted and steps_sorted[-1] not in [steps_sorted[i] for i in sample_idx]:
        sample_idx.append(len(steps_sorted) - 1)

    tot_opsd = tot_grpo = tot_sft = 0
    any_grpo_steps = []
    any_sft_steps = []
    for step in steps_sorted:
        r = routing[step]
        counts = r.get("completion_mode_counts", {})
        if isinstance(counts, str):
            counts = json.loads(counts.replace("'", '"'))
        total = sum(counts.values()) or 32
        opsd = counts.get("OPSD", 0)
        grpo = counts.get("GRPO", 0)
        sft = counts.get("SFT", 0)
        tot_opsd += opsd
        tot_grpo += grpo
        tot_sft += sft
        if grpo > 0:
            any_grpo_steps.append(step)
        if sft > 0:
            any_sft_steps.append(step)

    for i in sample_idx:
        step = steps_sorted[i]
        r = routing[step]
        counts = r.get("completion_mode_counts", {})
        if isinstance(counts, str):
            counts = json.loads(counts.replace("'", '"'))
        total = sum(counts.values()) or 32
        print(
            f"{step:>6} {counts.get('OPSD', 0) / total * 100:>6.1f}%"
            f" {counts.get('GRPO', 0) / total * 100:>6.1f}%"
            f" {counts.get('SFT', 0) / total * 100:>6.1f}%"
            f" {str(r.get('has_correct', '?')):>16}"
            f" {r.get('opsd_mask_ratio', 0) * 100:>9.1f}%"
        )

    tot = tot_opsd + tot_grpo + tot_sft
    print(
        f"\nAggregate ({len(routing)} snapshots x 32 samples):"
        f" OPSD={tot_opsd / tot * 100:.1f}%"
        f" GRPO={tot_grpo / tot * 100:.1f}%"
        f" SFT={tot_sft / tot * 100:.1f}%"
    )
    print(f"Steps with any GRPO on rank0: {len(any_grpo_steps)} ({any_grpo_steps[:5]}...)")
    print(f"Steps with any SFT on rank0: {len(any_sft_steps)}")

    # all-rank debug routing (rank0 only in DETAIL but DEBUG has all ranks)
    opsd_all = total_debug_routes * 32 - grpo_routes - sft_routes
    tot_all = total_debug_routes * 32
    if tot_all:
        print(
            f"\nAll-rank DEBUG routing lines (rank0 only in file grep):"
            f" GRPO samples={grpo_routes} SFT={sft_routes}"
            f" (from rank=0 completion_mode_counts lines: {total_debug_routes})"
        )

    print("\n=== REWARD / FORMAT LEARNING (rank0) ===")
    print(f"{'step':>6} {'fmt_zero':>9} {'acc_zero':>9} {'acc_sum':>8} {'fmt_sum':>8} {'w_mean':>8}")
    reward_steps = sorted(rewards.keys())
    r_sample = list(range(0, len(reward_steps), max(1, len(reward_steps) // 12)))
    if reward_steps and reward_steps[-1] not in [reward_steps[i] for i in r_sample]:
        r_sample.append(len(reward_steps) - 1)
    for i in r_sample:
        step = reward_steps[i]
        r = rewards[step]
        print(
            f"{step:>6} {r.get('format_zero_rate', 0) * 100:>8.1f}%"
            f" {r.get('acc_zero_rate', 0) * 100:>8.1f}%"
            f" {r.get('acc_sum', 0):>8.2f}"
            f" {r.get('format_sum', 0):>8.2f}"
            f" {r.get('weighted_mean', 0):>8.4f}"
        )
    if rewards:
        s0, sL = rewards[min(rewards)], rewards[max(rewards)]
        print(f"\nStep {min(rewards)} -> {max(rewards)} delta:")
        print(f"  format_zero_rate: {s0.get('format_zero_rate', 0) * 100:.1f}% -> {sL.get('format_zero_rate', 0) * 100:.1f}%")
        print(f"  acc_zero_rate:    {s0.get('acc_zero_rate', 0) * 100:.1f}% -> {sL.get('acc_zero_rate', 0) * 100:.1f}%")
        print(f"  acc_sum:          {s0.get('acc_sum', 0):.2f} -> {sL.get('acc_sum', 0):.2f}")
        print(f"  format_sum:       {s0.get('format_sum', 0):.2f} -> {sL.get('format_sum', 0):.2f}")

    # format reward rate from per-sample
    print("\n=== FORMAT REWARD RATE (per-sample, rank0) ===")
    for step in [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 850]:
        if step in per_sample_rewards:
            samples = per_sample_rewards[step]
            fmt_ok = sum(1 for _, f, _ in samples if f > 0)
            acc_ok = sum(1 for _, _, a in samples if a > 0)
            print(f"step {step:>4}: format_reward>0: {fmt_ok}/{len(samples)} ({fmt_ok/len(samples)*100:.1f}%)  acc>0: {acc_ok}/{len(samples)} ({acc_ok/len(samples)*100:.1f}%)")

    print("\n=== SAMPLE COMPLETIONS (first 4 per step, rank0) ===")
    for step in [0, 10, 100, 300, 500, max(sample_completions) if sample_completions else 0]:
        if step in sample_completions:
            print(f"\n--- step {step} ---")
            for i, t in enumerate(sample_completions[step][:4]):
                print(f"  [{i}] {t}")

    print("\n=== GENERATION LENGTH (rank0 DETAIL) ===")
    print(f"{'step':>6} {'eff_tok':>10} {'eos_term':>9} {'at_max':>8}")
    gen_steps = sorted(gens.keys())
    g_sample = list(range(0, len(gen_steps), max(1, len(gen_steps) // 12)))
    if gen_steps and gen_steps[-1] not in [gen_steps[i] for i in g_sample]:
        g_sample.append(len(gen_steps) - 1)
    for i in g_sample:
        step = gen_steps[i]
        g = gens[step]
        print(
            f"{step:>6} {g.get('effective_tokens_mean', 0):>10.1f}"
            f" {g.get('eos_terminated_rate', 0) * 100:>8.1f}%"
            f" {g.get('at_max_length_rate', 0) * 100:>7.1f}%"
        )

    print("\n=== GENERATION DEGENERATION (rank0 PROBE) ===")
    if probes:
        print(f"{'gstep':>6} {'eff_tok':>9} {'paren_eos':>10} {'one_tok':>8} {'eos_rate':>8}")
        pick = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 850]
        probe_by_gs = {p[0]: p for p in probes}
        for gs in pick:
            if gs in probe_by_gs:
                p = probe_by_gs[gs]
                print(f"{p[0]:>6} {p[1]:>9.1f} {p[2]:>10} {p[3]:>8} {p[4]:>7.2f}")
        print(f"\nTotal paren_then_eos: {sum(p[2] for p in probes)} / {len(probes)} regenerates")
        print(f"Total one_token: {sum(p[3] for p in probes)}")
        early = [p for p in probes if p[0] <= 10]
        late = [p for p in probes if p[0] >= 800]
        if early:
            print(f"Early (gs<=10) avg eff_tok={sum(p[1] for p in early)/len(early):.1f}")
        if late:
            print(f"Late (gs>=800) avg eff_tok={sum(p[1] for p in late)/len(late):.1f}, eos_rate={sum(p[4] for p in late)/len(late):.2f}")

    print("\n=== LOSS (rank0, last micro-batch per step) ===")
    loss_keys = ["grpo_loss_scalar", "opsd_loss_scalar", "opsd_loss", "opsd_active_samples", "opsd_samples"]
    loss_steps = sorted(losses_by_step.keys())
    l_sample = list(range(0, len(loss_steps), max(1, len(loss_steps) // 12)))
    if loss_steps and loss_steps[-1] not in [loss_steps[i] for i in l_sample]:
        l_sample.append(len(loss_steps) - 1)
    print(f"{'step':>6} {'grpo':>10} {'opsd':>10}")
    for i in l_sample:
        step = loss_steps[i]
        d = losses_by_step[step][-1]
        opsd_l = d.get("opsd_loss_scalar", d.get("opsd_loss", 0))
        print(f"{step:>6} {d.get('grpo_loss_scalar', 0):>10.4f} {opsd_l:>10.4f}")


if __name__ == "__main__":
    main()
