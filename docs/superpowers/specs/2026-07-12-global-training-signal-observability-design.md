# Global Training Signal Observability Design

## Problem

The adaptive supervision run mixed globally reduced controller metrics with
rank-local routing and health metrics. At step 65, rank 0 reported nearly all
wrong groups and almost no GRPO routing while the controller reported nonzero
global mixed groups and continued reducing supervision. The logs could not
distinguish a real global failure from rank-local sampling variance.

The controller also labels zero advantage from the weighted total reward as
`zero_loss`. Format or thinking reward variance can make that signal nonzero
even when ChartQA accuracy is identical within the group. This is useful
diagnostic information but is not equivalent to task-effective variance.

## Design

Introduce one pure count aggregation helper that converts local per-rank counts
to globally reduced rates. The trainer will use it once per generated step to
publish a coherent `global_signal/*` snapshot.

The snapshot contains:

- prompt, mixed, all-wrong, and all-correct rates based on ChartQA accuracy;
- total-reward zero-advantage rate;
- task-accuracy zero-advantage rate;
- disagreement rate between the two zero-advantage definitions;
- GRPO, OPD, SFT, and skip route rates after final routing;
- mean accuracy reward, clipping, EOS, and degeneration rates.

The existing adaptive controller remains unchanged for this diagnostic run.
Its current signal is logged explicitly as `total_reward_zero_loss`, removing
the ambiguous `zero_loss` interpretation from the new global snapshot. Legacy
metric names remain available for compatibility.

## Synchronization

Each rank emits integer counts and additive sums. Accelerate `reduce(sum)` is
the only cross-rank operation. Rates are derived from the reduced denominators,
so every rank receives the same snapshot. No averages of local rates are used.

## Early Health Gate

The monitoring workflow may stop a run only when a global rolling window shows
all three conditions:

- task all-wrong rate above `0.90`;
- final GRPO route rate below `0.05`;
- mean accuracy reward below `0.02`.

Rank-local health alerts remain diagnostics and cannot independently stop a
run.

## Compatibility

The change adds metrics and does not change losses, routing, sampling, or the
adaptive controller. After a 20-30 step smoke establishes the true global
signal relationships, a separate approved change may switch the controller to
task-effective readiness.

## Verification

- Pure tests prove count aggregation and zero-signal disagreement semantics.
- Trainer tests prove one global reduction and metric publication.
- An 8-GPU smoke verifies identical finite global snapshots and no collective
  mismatch.
- The next full run is evaluated automatically on ChartQA after saving the
  final checkpoint.
