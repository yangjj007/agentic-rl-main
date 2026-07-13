# Condition-Aware OPD Overflow Routing Design

## Goal

Create the next 4-epoch ChartQA training variant that preserves structured
`Goal/Observation/Reasoning/Conclusion` reasoning while preventing the late
OPD route cap from converting most recoverable wrong completions into full-hint
SFT targets.

## Evidence

The completed final checkpoint scored `0.5724`. Its last-50 routing metrics were:

- `routing/sft_route_rate=0.555`
- `routing/opd_route_cap_rate=0.430`
- `routing/teacher_sft_repair_rate=0.06625`
- `routing/grpo_route_rate=0.230625`

The route cap currently sends every overflow OPD completion to `MODE_SFT`.
Those rows are trained on the normal full `hint + answer` target, so the cap is
responsible for most late SFT traffic. The short teacher repair itself is small
and remains useful.

## Considered Approaches

### 1. Disable the OPD cap

This removes cap-created SFT, but leaves mixed wrong completions on OPD and does
not help GRPO become the dominant late signal.

### 2. Apply the existing effective-group filter after the cap

This can remove all-wrong rows, but mixed overflow rows have already become SFT.
It therefore does not correct the central mixed-group routing error.

### 3. Condition-aware overflow routing (selected)

Keep at most two OPD completions per prompt after the configured training phase.
For additional OPD rows:

- if the prompt group is mixed, route the row to GRPO so its original negative
  group-relative advantage is retained;
- if the prompt group is all-wrong, route the row to an explicit skip mode so
  it contributes zero policy, OPD, and SFT gradient;
- remove any teacher trajectory attached to an overflow row.

This preserves one `student_hint_short` SFT repair slot in all-wrong groups,
because teacher-SFT repair routing runs before the OPD cap. It also preserves up
to two OPD rows per prompt.

## Configuration

### Normalized training schedule

Absolute boundaries such as `147`, `294`, and `441` are artifacts of one
4-epoch run with 588 optimizer steps. They must not define algorithm behavior.

Add a shared phase schedule with two modes:

- `step`: legacy absolute-step behavior for reproducing old experiments;
- `progress`: normalized behavior for new experiments, where
  `progress = global_step / max_training_steps`.

The new variant uses `progress` with these boundaries, matching the relative
phases of the completed 4-epoch run:

- teacher-trajectory decay: `0.25 -> 0.50`;
- effective sampling activation: `0.50`;
- condition-aware OPD route-cap activation: `0.50`;
- OPD weight decay: `0.50 -> 0.75`.

`Trainer.state.max_steps` is the preferred training horizon. Existing
`resolve_max_training_steps` fallback logic handles explicit `max_steps` and
epoch-based runs. The same progress resolver must be used by loss schedules,
the dynamic sampler, and route-cap activation so their phases cannot drift.

Expose the schedule through:

- `DYME_PHASE_SCHEDULE_MODE=step|progress`;
- `DYME_TEACHER_TRAJ_DECAY_START_PROGRESS`;
- `DYME_TEACHER_TRAJ_DECAY_END_PROGRESS`;
- `DYME_OPSD_DECAY_START_PROGRESS`;
- `DYME_OPSD_DECAY_END_PROGRESS`;
- `DYME_EFFECTIVE_SAMPLING_START_PROGRESS`;
- `DYME_OPSD_ROUTE_CAP_START_PROGRESS`.

Legacy step variables remain supported and are used only in `step` mode.

### Overflow policy

Add an OPD route-cap overflow policy with two supported values:

- `sft`: legacy behavior and global default;
- `mixed_grpo_all_wrong_skip`: new condition-aware behavior.

Expose it through `DYME_OPSD_OVERFLOW_ROUTE` and pass it into the existing
`loss.route_cap.overflow_route` config field.

Register a new runner variant:

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow`

The variant inherits the current settings for oracle providers,
`student_hint_short` repair, teacher-trajectory decay, OPD weight decay/cap, and
effective sampling. It leaves additive eval-format reward disabled and does not
enable positive or rollout replay.

### Dynamic-trigger observability

The user-preferred future policy is to activate late phases from training
signals rather than elapsed progress. This run does not let those signals alter
routing yet; it records enough state to select and validate a dynamic policy.

Maintain an exponential moving average for globally consistent prompt-group
signals:

- mixed-rate EMA, default alpha `0.1`;
- GRPO zero-loss EMA, default alpha `0.1`.

Record two independent shadow conditions against configurable defaults:

- EMA alpha `0.10`;
- minimum progress `0.20`;
- patience `20` consecutive optimizer steps;
- sampling-needed: mixed-rate EMA at most `0.20` and zero-loss EMA at least
  `0.70`;
- RL-ready: mixed-rate EMA at least `0.30` and zero-loss EMA at most `0.30`.

The sampling-needed condition identifies sparse/collapsed GRPO signal and could
later activate effective sampling. The RL-ready condition identifies stable
mixed groups and could later activate OPD decay plus condition-aware GRPO
transition. Keeping them separate prevents a circular controller that waits for
effective sampling's result before allowing effective sampling to start.

These values are diagnostic thresholds, not active routing gates. Expose them
through `DYME_DYNAMIC_TRIGGER_*` environment variables so logs from the next
run can be replayed against alternative thresholds without retraining. Latch
the first step and progress where each condition reaches its patience target.

## Routing API

Add `MODE_SKIP` as a fourth internal routing mode. `DyMETrainer` already has a
zero-advantage fallback branch for modes that are not GRPO, OPD, or SFT; make
the skip behavior explicit and count it in routing metrics.

Extend `apply_opd_route_cap` to receive `group_has_correct`. Preserve the current
return shape, but extend `OpdRouteCapStats` with:

- `rerouted_grpo`
- `skipped`

The old `sft` policy must remain byte-for-byte compatible in routing behavior.

## Metrics

Add these late-routing metrics:

- `routing/opd_route_cap_grpo_rate`
- `routing/opd_route_cap_skip_rate`
- `routing/skip_route_rate`

Add normalized schedule metrics:

- `phase/training_progress`
- `phase/max_training_steps`
- `phase/teacher_traj_decay_active`
- `phase/effective_sampling_active`
- `phase/opd_decay_active`
- `phase/opd_route_cap_active`

Add dynamic-trigger observation metrics:

- `phase/dynamic_mixed_rate_ema`
- `phase/dynamic_zero_loss_rate_ema`
- `phase/dynamic_sampling_needed_now`
- `phase/dynamic_sampling_needed_streak`
- `phase/dynamic_sampling_would_trigger`
- `phase/dynamic_sampling_trigger_progress`
- `phase/dynamic_rl_ready_now`
- `phase/dynamic_rl_ready_streak`
- `phase/dynamic_rl_would_trigger`
- `phase/dynamic_rl_trigger_progress`

Keep all existing route-cap metrics so the new run remains comparable with the
completed experiment.

## Tests

Use test-first development for:

1. normalized phase boundaries scale with total training steps;
2. legacy absolute-step schedules remain unchanged;
3. mixed-group OPD overflow becomes GRPO;
4. all-wrong OPD overflow becomes skip;
5. teacher trajectories are removed for both overflow destinations;
6. legacy `sft` overflow remains unchanged;
7. dynamic-trigger EMA metrics update without controlling the active schedule;
8. sampling-needed and RL-ready streaks latch independently;
9. the new runner variant exports progress scheduling, OPD decay, effective
   sampling, the new
   overflow policy, `student_hint_short`, and disabled eval-format reward;
10. config environment overrides reach the shared phase schedule and
   `loss.route_cap.overflow_route`.

Run the focused routing/config/runner tests, followed by the complete related
test modules and a dry-run of the new variant.

## Success Criteria

The smoke test must finish without routing/config errors, log nonzero
condition-aware cap metrics after progress reaches `0.50`, and emit both shadow
dynamic-trigger metric families without allowing them to alter the active
progress schedule. The 4-epoch run should aim for:

- last-50 `routing/grpo_route_rate > 0.30`;
- last-50 `routing/sft_route_rate < 0.30`;
- last-50 `signal/grpo_zero_loss_rate < 0.15`;
- zero privileged-tag leakage;
- final ChartQA accuracy above `0.5872`, with `0.60+` as the target.

Structured complete CoT is allowed at any rate. Evaluation should separately
track empty-shell templates and answer-line validity instead of treating all
full CoT as pollution.
