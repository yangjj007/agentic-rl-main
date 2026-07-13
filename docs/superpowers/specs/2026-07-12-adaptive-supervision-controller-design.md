# Adaptive Supervision Controller Design

Date: 2026-07-12

## 1. Goal

Replace the independent step/progress switches for effective sampling, OPD
route capping, OPD loss decay, and teacher-trajectory decay with one
scale-independent controller driven by observed RL learnability.

The same controller must work without retuning when training changes from four
epochs to ten epochs. It must remain deterministic across distributed ranks and
must preserve the existing ChartQA full-CoT quality gate.

## 2. Problem

The current training path contains several overlapping schedules:

- effective sampling has its own activation boundary;
- OPD route capping has another activation boundary;
- OPD loss has a linear decay interval;
- teacher trajectory loss has a separate decay interval;
- a dynamic-trigger monitor computes related signals but does not control
  training.

Changing fixed steps to normalized progress avoids one failure mode but still
ties behavior to a particular learning curve. It also allows logically related
actions to become inconsistent, such as capping OPD before GRPO has usable
signal or retaining full teacher loss after RL has become effective.

## 3. Controller Signal

The controller observes two globally aggregated optimizer-step signals:

- `mixed_rate`: fraction of prompt groups containing both correct and wrong
  completions;
- `zero_loss_rate`: fraction of prompt groups whose GRPO advantages are all
  effectively zero.

It maintains conservative exponential moving averages initialized as:

```text
mixed_ema = 0
zero_loss_ema = 1
```

For update coefficient `alpha`:

```text
mixed_ema = alpha * mixed_rate + (1 - alpha) * mixed_ema
zero_loss_ema = alpha * zero_loss_rate + (1 - alpha) * zero_loss_ema
readiness = mixed_ema * (1 - zero_loss_ema)
mastery = max(previous_mastery, readiness)
```

Conservative initialization removes the need for a step-count warmup. A noisy
first batch cannot immediately reduce supervision. Monotonic mastery prevents
short-term metric regressions from re-enabling teacher supervision.

## 4. Continuous Control

Let `target_readiness` be the readiness at which RL process supervision is
considered mature. Convert mastery to a smooth transition:

```text
x = clamp(mastery / target_readiness, 0, 1)
transition = x * x * (3 - 2 * x)
supervision = 1 - transition
```

All related actions derive from this one supervision value:

```text
opsd_weight = opsd_final + (opsd_initial - opsd_final) * supervision
teacher_traj_weight = teacher_final +
                      (teacher_initial - teacher_final) * supervision
opd_max_per_prompt = ceil(opd_cap_final +
                          (opd_cap_initial - opd_cap_final) * supervision)
```

Default values for the new full-CoT experiment are:

```text
alpha = 0.10
target_readiness = 0.20
opsd_initial = 1.50
opsd_final = 0.50
teacher_initial = 0.50
teacher_final = 0.00
opd_cap_initial = num_generations
opd_cap_final = 2
```

`target_readiness=0.20` has a semantic interpretation close to mixed rate
`0.30` with zero-loss rate `0.30`, rather than an epoch-relative timestamp.

## 5. Effective Sampling

When the controller is enabled, dynamic effective sampling is active from the
start. This does not require a trigger:

- before mixed examples exist, all-wrong and unknown rows retain their normal
  sampling weights;
- when mixed examples appear, their existing higher sampling weight takes
  effect automatically;
- no separate activation schedule is needed.

The sampler continues to receive per-example outcome updates. Its legacy
step/progress activation remains available only for variants that do not enable
the new controller.

## 6. Distributed Consistency

Controller input must be computed from global counts, not rank0 health logs.
For each generated batch, every rank contributes:

- prompt-group count;
- mixed-group count;
- zero-advantage-group count.

Counts are summed across ranks. Each rank then computes identical global rates
and calls the deterministic controller once for the same optimizer step. The
controller ignores duplicate updates for a step so gradient accumulation and
buffer reuse cannot advance it multiple times.

The resulting immutable state snapshot is used by:

- OPD route capping during routing;
- OPD loss weighting during `compute_loss`;
- teacher trajectory weighting during `compute_loss`;
- adaptive-supervision metrics.

## 7. Simplification and Compatibility

Create `opsd_utils/adaptive_supervision.py` as the only owner of readiness,
mastery, interpolation, and action derivation.

For the new adaptive variant:

- disable independent OPD weight decay;
- disable independent teacher trajectory decay;
- disable scheduled OPD route-cap activation;
- replace the diagnostic-only dynamic-trigger monitor;
- keep the existing route-cap filtering implementation, but supply the
  controller-derived cap and mark it active from the controller state;
- keep the full-CoT Q3 quality gate unchanged.

Legacy variants retain their current schedules. This permits exact reproduction
of prior experiments while making the new path substantially smaller.

## 8. Metrics

Log one coherent metric group:

```text
adaptive/enabled
adaptive/mixed_rate
adaptive/zero_loss_rate
adaptive/mixed_ema
adaptive/zero_loss_ema
adaptive/readiness
adaptive/mastery
adaptive/supervision
adaptive/opsd_weight
adaptive/teacher_traj_weight
adaptive/opd_max_per_prompt
adaptive/update_count
```

The old `phase/dynamic_*` metrics are not emitted by the adaptive variant.

## 9. Failure Behavior

- Missing or non-finite rates are treated conservatively as `mixed=0` and
  `zero_loss=1`.
- `target_readiness` must be positive.
- weights and caps are clamped to their configured endpoint ranges.
- controller state is monotonic in mastery and cannot increase supervision.
- controller configuration is saved in the resolved run config.

## 10. Verification

Unit tests must cover:

- conservative initialization;
- no readiness under all-wrong/zero-signal batches;
- smooth supervision reduction as mixed signal improves;
- monotonic mastery after a metric regression;
- resistance to a single early spike through EMA initialization;
- endpoint weights and caps;
- duplicate-step idempotence;
- invalid/non-finite inputs;
- scale independence under identical signal sequences labelled as four-epoch
  and ten-epoch runs.

Integration tests must prove that the adaptive variant disables all legacy
schedules and that trainer routing/loss paths consume the same controller
snapshot. A short distributed smoke run must log finite adaptive metrics and
complete without rank divergence or OOM.
