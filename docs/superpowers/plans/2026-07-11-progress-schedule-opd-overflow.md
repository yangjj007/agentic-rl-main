# Progress Schedule and OPD Overflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normalized training-phase scheduling, diagnostic shadow triggers, and condition-aware OPD overflow routing for the next ChartQA run.

**Architecture:** A focused `phase_schedule` module resolves legacy step and normalized progress boundaries. A stateful `DynamicTriggerMonitor` observes mixed/zero-loss EMAs without controlling training. Routing adds an explicit skip mode and a new overflow policy, while the trainer, config, and runner wire these pieces together behind a new variant.

**Tech Stack:** Python, PyTorch/Transformers Trainer, Bash runner configuration, pytest.

---

### Task 1: Shared normalized phase schedule

**Files:**
- Create: `opsd_utils/phase_schedule.py`
- Create: `tests/test_phase_schedule.py`
- Modify: `opsd_utils/teacher_traj_schedule.py`
- Test: `tests/test_teacher_traj_schedule.py`

- [ ] Write failing tests proving `0.25/0.50/0.75` resolve to `3/5/8` for 10 steps and `147/294/441` for 588 steps, while legacy step mode remains unchanged.
- [ ] Run `pytest -q tests/test_phase_schedule.py tests/test_teacher_traj_schedule.py` and verify failures are caused by the missing progress API.
- [ ] Implement `training_progress`, `resolve_schedule_step`, `schedule_active`, and progress-aware linear weight interpolation.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Shadow dynamic-trigger monitor

**Files:**
- Create: `opsd_utils/dynamic_trigger_monitor.py`
- Create: `tests/test_dynamic_trigger_monitor.py`

- [ ] Write failing tests for EMA updates, independent sampling-needed/RL-ready streaks, minimum progress, patience, and latched trigger progress.
- [ ] Run `pytest -q tests/test_dynamic_trigger_monitor.py` and verify failure from the missing monitor.
- [ ] Implement `DynamicTriggerConfig`, `DynamicTriggerState`, and `DynamicTriggerMonitor.update()` returning numeric log metrics.
- [ ] Re-run the monitor tests and verify they pass.

### Task 3: Condition-aware route-cap behavior

**Files:**
- Modify: `opsd_utils/constants.py`
- Modify: `opsd_utils/__init__.py`
- Modify: `opsd_utils/signal_aware_routing.py`
- Modify: `tests/test_signal_aware_routing.py`

- [ ] Write failing tests showing mixed overflow becomes `MODE_GRPO`, all-wrong overflow becomes `MODE_SKIP`, attached teacher trajectories are removed, progress activation scales with the horizon, and legacy `sft` behavior is preserved.
- [ ] Run `pytest -q tests/test_signal_aware_routing.py` and verify the new cases fail for the expected missing mode/policy.
- [ ] Add `MODE_SKIP`, progress-aware cap configuration, `group_has_correct`, and cap statistics for GRPO reroutes/skips.
- [ ] Re-run routing tests and verify they pass.

### Task 4: Config and sampler integration

**Files:**
- Modify: `config/config_opd_7b_dyme_probe.py`
- Modify: `trainer/DyMETrainer.py`
- Modify: `tests/test_config_antidegen.py`
- Modify: `tests/test_dynamic_signal_sampler.py`

- [ ] Write failing tests for progress-mode environment overrides and sampler activation at normalized progress.
- [ ] Run the focused config/sampler tests and verify expected failures.
- [ ] Add phase schedule, overflow route, and dynamic-trigger environment configuration; make `DynamicSignalRepeatSampler.set_step()` horizon-aware.
- [ ] Re-run the focused tests and verify they pass.

### Task 5: Trainer routing and metrics integration

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Modify: `tests/test_signal_aware_routing.py`
- Modify: `tests/test_dynamic_trigger_monitor.py`

- [ ] Extend tests to cover skip rows receiving zero advantage and diagnostic monitor metrics remaining observational.
- [ ] Run the tests and verify the trainer-facing expectations fail before integration.
- [ ] Instantiate the shadow monitor, pass normalized schedule context to sampler/cap/loss schedules, count skip routes, and log phase, cap, and trigger metrics.
- [ ] Re-run routing, monitor, schedule, and trainer-adjacent tests.

### Task 6: New 4-epoch runner variant

**Files:**
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_pcd_no_visual_runner.py`
- Modify: `docs/pcd_oracle_exploration_notes.md`

- [ ] Write a failing dry-run test requiring the new variant to export progress scheduling, `mixed_grpo_all_wrong_skip`, OPD/trajectory decay, effective sampling, `student_hint_short`, and disabled eval-format/replay.
- [ ] Run the runner test and verify it fails because the variant is unregistered.
- [ ] Register the variant, export all new environment variables, and document smoke/4epoch commands plus metrics.
- [ ] Re-run the runner tests and execute a shell dry-run.

### Task 7: Verification

**Files:**
- Verify all files above.

- [ ] Run `pytest -q tests/test_phase_schedule.py tests/test_teacher_traj_schedule.py tests/test_dynamic_trigger_monitor.py tests/test_signal_aware_routing.py tests/test_dynamic_signal_sampler.py tests/test_config_antidegen.py tests/test_pcd_no_visual_runner.py`.
- [ ] Run `bash scripts/test/run_pcd_no_visual_4epoch.sh --dry-run --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow`.
- [ ] Scan the dry-run for absolute `147/294/441` phase control in the new variant and confirm progress mode is authoritative.

The repository's `.git` directory is empty, so commit steps are unavailable in this workspace. Each passing test boundary serves as the implementation checkpoint.
