# Adaptive Effective Sampling Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every adaptive-supervision experiment declare and execute effective sampling from step zero while preserving legacy step/progress schedules.

**Architecture:** The experiment runner will resolve adaptive variants to an explicit zero activation boundary, so dry-run output and saved configuration match the approved controller semantics. `DyMETrainer` remains the runtime authority through `DynamicSignalRepeatSampler(always_active=True)` whenever the adaptive controller exists, with a direct trainer test preventing regressions.

**Tech Stack:** Bash experiment runner, Python, PyTorch sampler, pytest.

---

### Task 1: Lock Runner Resolution With a Failing Test

**Files:**
- Modify: `tests/test_pcd_no_visual_runner.py`

- [x] **Step 1: Extend the adaptive variant assertions**

Add exact assertions to each adaptive runner test:

```python
assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0" in out
assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0" in out
```

- [x] **Step 2: Run the focused runner tests and verify RED**

Run:

```bash
PYTHONPATH=. /home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/test_pcd_no_visual_runner.py::test_full_cot_adaptive_supervision_variant_removes_legacy_schedules \
  tests/test_pcd_no_visual_runner.py::test_opd_no_hard_imitation_variant_disables_trajectory_and_sft_repair \
  tests/test_pcd_no_visual_runner.py::test_opd_no_full_hint_hard_sft_variant_exports_all_invariants -q
```

Expected: all three fail because the dry-run still exports the legacy `294` and `0.50` boundaries.

### Task 2: Resolve Adaptive Runner Boundaries to Step Zero

**Files:**
- Modify: `scripts/test/run_pcd_no_visual.sh`

- [x] **Step 1: Add one shared adaptive normalization block**

After variant-specific settings, normalize effective sampling only when adaptive supervision is enabled:

```bash
if [[ "${ADAPTIVE_SUPERVISION}" == "1" ]]; then
  EFFECTIVE_SAMPLING_AFTER_STEP=0
  EFFECTIVE_SAMPLING_START_PROGRESS=0.0
fi
```

This deliberately overrides stale caller values for adaptive runs because the approved design removes the legacy activation schedule entirely.

- [x] **Step 2: Run the focused runner tests and verify GREEN**

Run the Task 1 pytest command. Expected: all three pass.

### Task 3: Prove Trainer Runtime Activation Is Schedule-Independent

**Files:**
- Modify: `tests/test_adaptive_supervision_trainer.py`

- [x] **Step 1: Write a trainer sampler test**

Construct a trainer with an enabled adaptive controller and an effective-sampling config that still contains `after_step=294`. Build the sampler and assert:

```python
assert sampler.always_active is True
sampler.set_step(0, max_steps=588)
assert sampler.enabled_for_step is True
```

- [x] **Step 2: Run the new test and verify its current runtime behavior**

Run:

```bash
PYTHONPATH=. /home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_adaptive_supervision_trainer.py -q
```

Expected: pass, documenting the existing trainer safeguard. If construction exposes a missing trainer fixture field, fix only the test fixture because no new production behavior is required.

### Task 4: Regression, Dry Run, and Queued-Run Hot Reload

**Files:**
- Modify: `docs/superpowers/plans/2026-07-13-adaptive-effective-sampling-start.md`
- Append: `outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/launch.info`

- [x] **Step 1: Run the relevant regression suite**

Run:

```bash
PYTHONPATH=. /home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/test_adaptive_supervision.py \
  tests/test_adaptive_supervision_trainer.py \
  tests/test_config_antidegen.py \
  tests/test_pcd_no_visual_runner.py \
  tests/test_pcd_resilient_runner.py -q
```

Expected: all tests pass.

- [x] **Step 2: Audit the exact queued variant with dry-run**

Run the four-epoch runner in dry-run mode and verify adaptive supervision is enabled, effective sampling is enabled with both boundaries at zero, all independent loss decays are disabled, and the global GRPO route is the readiness source.

- [x] **Step 3: Hot reload only while the GPU gate is still waiting**

Confirm the state file still says `waiting_for_gpu_gate` and no checkpoint or training log exists. Restart the train/watch/forensics tmux commands only if their already-running shell captured code or environment that would prevent the new runner resolution from being used; otherwise leave the waiting process intact because it invokes the runner after the gate opens.

- [x] **Step 4: Record the correction**

Append the test result, dry-run evidence, and whether a tmux restart was needed to `launch.info`. Mark this plan complete after the queued run is demonstrably configured with step-zero effective sampling.

## Verification Evidence

- RED: all three adaptive runner tests failed because dry-run exported `DYME_EFFECTIVE_SAMPLING_AFTER_STEP=294`.
- GREEN: the same three tests passed after adaptive-only boundary normalization.
- Runtime safeguard: `test_adaptive_effective_sampler_is_active_from_step_zero_despite_legacy_boundary` passed and proves `always_active=True` bypasses a stale legacy boundary.
- Regression: 70 adaptive/config/runner/resilient tests passed.
- Exact queued-variant dry-run: `effective sampling: enabled=1 after=0`; exported `DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0`, `DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route`, and all independent decay/monitor flags disabled.
- Queue state at verification: `waiting_for_gpu_gate`, output directory absent, zero checkpoints. No tmux restart was needed because `run_pcd_no_visual_resilient.sh` launches the main runner only after the GPU gate passes.
