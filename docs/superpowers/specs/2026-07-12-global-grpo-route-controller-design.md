# Global GRPO Route Controller Design

## Motivation

The 30-step global diagnostic showed that total-reward variance is not a
reliable proxy for task-effective learning. During steps 11-20, the task
accuracy zero rate was `1.0`, the total-reward zero rate was `0.0`, and the
global final GRPO route rate was `0.0`. Format and thinking rewards therefore
made the old controller reduce supervision despite no task-effective GRPO
route.

## Signal

Use one globally synchronized scalar:

`signal_rate = global final GRPO completion route count / global completion count`

The final route is measured after teacher repair, OPD capping, route guards,
and effective filtering. It is the fraction of generated completions that will
actually retain GRPO training behavior.

## Controller

- Initialize `signal_ema=0` and full supervision.
- Update once after final routing and global count reduction.
- `signal_ema = alpha * signal_rate + (1-alpha) * signal_ema`.
- `mastery = max(previous_mastery, signal_ema)`.
- Normalize by target route rate `0.30` and use the existing smoothstep curve.
- Derive OPD weight `1.5 -> 0.5`, teacher weight `0.5 -> 0`, and OPD cap `8 -> 2`
  from the same snapshot.
- The snapshot measured at step N controls actions at step N+1, avoiding a
  routing/control circular dependency.

## Compatibility

The original mixed/zero controller mode remains available for historical
variants. The adaptive full-CoT variant selects `global_grpo_route`. Existing
diagnostic metrics remain unchanged, and direct-signal metrics are added as
`adaptive/signal_rate` and `adaptive/signal_ema`.

## Verification

- Pure tests cover direct EMA, endpoints, monotonicity, and duplicate steps.
- Trainer tests prove pre-route updates are skipped in direct mode and the
  globally reduced route snapshot updates the controller exactly once.
- A 4-step 8-GPU smoke verifies collective safety.
- A 30-step comparison verifies supervision remains near full while global
  GRPO route is near zero.
