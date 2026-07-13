# No-Full-Hint Hard-SFT OPD Design

Date: 2026-07-13
Status: Approved by user on 2026-07-13

## 1. Goal

Create a causal OPD experiment that disables every training path which applies
the full ChartQA dataset hint as a hard token target, while preserving
verifier-routed OPD, GRPO, effective sampling, the realized-global-GRPO
controller, and the model's freedom to produce full chain-of-thought.

The change must also make this invariant observable during training so a future
run cannot be mislabeled as no-hard-imitation while legacy online SFT remains
active.

## 2. Evidence and Failure

The stopped run
`oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946` correctly kept
both teacher-specific hard paths at zero:

```text
loss/teacher_traj_effective_weight = 0
routing/teacher_sft_repair_rate = 0
```

However, `DyMETrainer` still built legacy online-SFT targets as
`hint + answer`. ChartQA dataset hints contain the complete
`Goal/Observation/Reasoning/Conclusion` trajectory, and the fast profile
reserved four SFT slots per all-wrong group during warmup. The observed SFT
route rate remained approximately 0.49--0.53. The run therefore removed
teacher-generated trajectories but retained full-hint hard supervision.

The step 51--60 recovery gate failed with accuracy 0.0070, GRPO route 0.0156,
and degenerate rate 0.9156. This is not sufficient evidence that OPD itself
caused the collapse because the legacy hard-target path was still active.

## 3. Chosen Intervention

Add an explicit no-full-hint-hard-SFT experiment variant with these invariants:

```text
teacher trajectory disabled
teacher-SFT repair disabled
legacy online-SFT slots disabled
legacy all-wrong online-SFT routing disabled
full-hint hard-target completion count = 0
```

Teacher-verified wrong completions continue to use OPD. Correct completions use
GRPO. Unverified, teacher-incorrect, or budget-overflow completions use the
existing skip/GRPO fallback policy; they must not silently fall back to the
dataset full hint.

The intervention does not:

- penalize `Goal`, `Observation`, `Reasoning`, or `Conclusion` in the reward;
- ban full-CoT generation;
- remove structured reasoning instructions from evaluation;
- apply heading-selective OPD;
- change the OPD divergence or controller signal.

Heading-selective OPD remains a later ablation only if the clean causal run
still shows strong style drift.

## 4. Configuration Interface

Expose two explicit environment-backed gate controls:

```text
DYME_DISABLE_ONLINE_SFT_SLOTS=1
DYME_ONLINE_SFT_ON_ALL_WRONG=0
```

The resolved OPD gate must contain:

```python
{
    "disable_online_sft_slots": True,
    "online_sft_on_all_wrong": False,
}
```

The experiment runner must set both values for the new variant. Legacy
variants retain current defaults for reproducibility.

## 5. Routing Invariants

At completion assembly time, record why each hard replacement occurred. The
new variant must satisfy all of the following on every logged step:

```text
routing/legacy_online_sft_rate = 0
routing/full_hint_hard_target_rate = 0
routing/teacher_sft_repair_rate = 0
loss/teacher_traj_effective_weight = 0
```

`routing/sft_route_rate` alone is insufficient because it merges multiple
sources. The new metrics must distinguish legacy dataset-hint SFT from
teacher-SFT repair and other route labels.

If either legacy online-SFT or full-hint hard-target rate becomes non-zero, the
long-run monitor exits with a mechanism-violation status and stops the run.

## 6. Template Monitoring Correction

Teacher-probe candidate JSONL stores previews with physical newlines escaped as
`\\n`. The external monitor must normalize `\\r\\n` and `\\n` back to line
breaks before applying section regexes. This normalization is restricted to
diagnostic parsing and does not alter model outputs or training data.

The corrected candidate metrics must report at least:

```text
candidate_full_cot_template_rate
candidate_partial_cot_template_rate
candidate_goal_without_answer_rate
candidate_canonical_answer_rate
candidate_malformed_answer_rate
```

A high partial-heading rate is a warning. A mechanism invariant violation is an
immediate stop. A sustained full-template collapse with malformed answers is a
stop condition. Structured but substantive and correctly answered full-CoT is
not considered a failure.

## 7. Variant and Runtime Gates

The new variant is matched to the stopped run except for removal of all
full-hint hard SFT. It retains:

- oracle hint and DePlot evidence for the diagnostic upper-bound setting;
- verifier-routed OPD;
- effective sampling;
- realized-global-GRPO continuous controller;
- OPD initial/final weight and route-cap endpoints;
- generation and optimizer hyperparameters;
- 8-GPU, 4epoch budget;
- automatic 8-GPU ChartQA final evaluation with batch size 1.

Before the full run, an 8-GPU smoke must cover:

1. all-wrong, teacher-correct completions route to OPD without legacy SFT;
2. mixed groups retain GRPO on correct completions;
3. teacher-incorrect or unavailable completions follow fallback without a full
   dataset hint;
4. hard-target invariants remain zero on all ranks;
5. candidate monitor correctly parses escaped multiline previews;
6. no OOM, NCCL divergence, or hanging rank.

The full run uses matched-window gates at steps 20, 40, 60, and 100. Early
partial style drift alone does not stop training. Mechanism violations stop
immediately. At step 60, persistent degenerate rate above 0.60 together with
accuracy below 0.02 and GRPO route below 0.02 stops the run.

The external health checker records `total_rows`, writes each reached gate once as
`gate_20.json`, `gate_40.json`, `gate_60.json`, or `gate_100.json`, and returns a
dedicated recovery-failure exit code only when all three step-60 conditions hold.
The long-run watch session treats mechanism violation, malformed template collapse,
and this joint recovery failure as stop conditions; candidate-only partial drift remains
warning-only.

## 8. Runtime Resilience

The clean causal run must not lose an epoch of evidence to an unrelated device failure.
For this variant only, the default save policy is:

```text
save_strategy = steps
save_steps = 50
save_total_limit = 3
```

Legacy variants retain epoch-boundary saving. A resilient wrapper waits until all eight
GPUs satisfy configurable memory, temperature, and utilization thresholds for consecutive samples.
Every ready sample additionally requires the global
`nvidia-smi --query-compute-apps` list to be empty. A low-utilization resident process below
the memory threshold is not an idle GPU; this prevents an eight-rank attempt from sharing a
device with an existing training process.
It retries only recognized CUDA/NCCL transient failures. If a checkpoint exists, the next
attempt uses automatic resume; otherwise the partial output is archived before a clean
restart. Configuration, data, seed, routing, and supervision settings remain unchanged.

## 9. Verification

TDD must cover:

- environment parsing for both new gate values;
- runner dry-run exports for the new variant;
- no legacy SFT slot replacement when disabled;
- no all-wrong online-SFT route when disabled;
- exact source-specific routing metrics;
- monitor normalization of escaped candidate previews;
- monitor mechanism-violation exit code;
- legacy variant compatibility;
- distributed smoke scenarios and clean exit.
- step-checkpoint environment parsing and variant-scoped runner defaults;
- GPU temperature/memory waiting, compute-process-empty gating, transient resume, and
  non-transient no-retry behavior.

No production behavior changes are permitted before the corresponding test has
failed for the expected missing behavior.

## 10. Paper Interpretation

This is a causal cleanup experiment, not a new paper contribution. Its purpose
is to isolate OPD's effectiveness and complementarity from full-trajectory hard
supervision. The paper's core novelty remains the systematic introduction and
evaluation of OPD for sub-1B VLM verifiable reasoning. Routing, monitoring, and
the controller are enabling mechanisms that make the OPD comparison valid.
