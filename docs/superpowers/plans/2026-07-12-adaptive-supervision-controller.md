# Adaptive Supervision Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace epoch-dependent OPD and teacher schedules with one globally synchronized continuous supervision controller.

**Architecture:** A pure controller module converts global mixed/zero-loss rates into a monotonic mastery value and one immutable action snapshot. `DyMETrainer` updates it once per generated optimizer step and reuses the snapshot for sampling, route capping, OPD loss, and teacher-trajectory loss; legacy variants keep their old schedules.

**Tech Stack:** Python, PyTorch distributed collectives through Accelerate, pytest, Bash experiment runner.

---

### Task 1: Pure Adaptive Controller

**Files:**
- Create: `opsd_utils/adaptive_supervision.py`
- Create: `tests/test_adaptive_supervision.py`

- [x] **Step 1: Write failing tests for initialization, interpolation, endpoints, monotonicity, duplicate steps, and invalid values**

Define tests against `AdaptiveSupervisionConfig`, `AdaptiveSupervisionController`, and `AdaptiveSupervisionState`. Assert full initial supervision, smooth intermediate values, exact final endpoints, no mastery regression, and idempotent repeated step updates.

- [x] **Step 2: Run the controller tests and verify RED**

Run: `python -m pytest tests/test_adaptive_supervision.py -q`

Expected: collection failure because `opsd_utils.adaptive_supervision` does not exist.

- [x] **Step 3: Implement the minimal pure controller**

Implement frozen config/state dataclasses, finite input sanitization, conservative EMA initialization, smoothstep interpolation, monotonic mastery, derived weights/cap, and duplicate-step idempotence. The module must not import torch or trainer code.

- [x] **Step 4: Run controller tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 2: Global Signal Aggregation and Trainer Snapshot

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Create: `tests/test_adaptive_supervision_trainer.py`

- [x] **Step 1: Write failing trainer tests**

Test construction from `opsd_config["adaptive_supervision"]`, global count-to-rate conversion, identical action snapshots from identical global counts, and fallback to legacy behavior when disabled.

- [x] **Step 2: Run trainer integration tests and verify RED**

Run: `python -m pytest tests/test_adaptive_supervision_trainer.py -q`

Expected: failures for missing trainer controller helpers.

- [x] **Step 3: Add controller lifecycle and globally reduced signal update**

Instantiate the controller in `DyMETrainer.__init__`. Add a focused helper that reduces prompt, mixed, and zero-advantage group counts across ranks, updates once per generated optimizer step, stores the immutable snapshot, and appends `adaptive/*` metrics.

- [x] **Step 4: Run trainer integration tests and verify GREEN**

Run the Task 2 pytest command. Expected: all tests pass.

### Task 3: Replace Independent Actions in the Adaptive Path

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Modify: `opsd_utils/opd_route_cap.py`
- Modify: `trainer/sampler.py`
- Modify: `tests/test_adaptive_supervision_trainer.py`
- Modify: `tests/test_opd_route_cap.py`

- [x] **Step 1: Write failing tests for shared snapshot consumption**

Assert that sampler activation is unconditional for the adaptive path, route cap equals the snapshot cap, OPD effective weight equals the snapshot OPD weight, and teacher trajectory effective weight equals the same snapshot's teacher value.

- [x] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_adaptive_supervision_trainer.py tests/test_opd_route_cap.py -q`

Expected: adaptive overrides are not yet consumed.

- [x] **Step 3: Wire the snapshot into all four action points**

When adaptive control is enabled, bypass sampler schedule checks, pass a controller-derived active cap to route filtering, bypass OPD linear decay, and bypass teacher-trajectory linear decay. Do not alter legacy execution when disabled.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 pytest command. Expected: all tests pass.

### Task 4: Configuration and Reproducible Variant

**Files:**
- Modify: `config/config_opd_7b_dyme_probe.yaml`
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_config_antidegen.py`
- Modify: `tests/test_pcd_no_visual_runner.py`

- [x] **Step 1: Write failing config and runner tests**

Add assertions for environment overrides, resolved adaptive config, and a new `deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision` variant that enables adaptive control while disabling all four legacy schedules.

- [x] **Step 2: Run config tests and verify RED**

Run: `python -m pytest tests/test_config_antidegen.py tests/test_pcd_no_visual_runner.py -q`

Expected: missing adaptive configuration and variant failures.

- [x] **Step 3: Add config fields and runner variant**

Expose environment variables for enablement, EMA alpha, readiness target, endpoint weights, and endpoint caps. The new variant must preserve full-CoT Q3 gating and effective-sampling weights while explicitly disabling progress/step decay and the old dynamic-trigger monitor.

- [x] **Step 4: Run config tests and verify GREEN**

Run the Task 4 pytest command. Expected: all tests pass.

### Task 5: Deterministic Scenario Simulator

**Files:**
- Create: `scripts/simulate_adaptive_supervision.py`
- Create: `tests/test_adaptive_supervision_simulator.py`

- [x] **Step 1: Write failing simulator tests**

Cover built-in scenarios `cold_start`, `gradual_learning`, `regression`, and `single_spike`. Assert JSON summaries are deterministic and regression never increases supervision.

- [x] **Step 2: Run simulator tests and verify RED**

Run: `python -m pytest tests/test_adaptive_supervision_simulator.py -q`

Expected: simulator module is missing.

- [x] **Step 3: Implement the simulator**

Provide built-in signal sequences plus optional JSONL input. Print per-step state as JSONL and a final summary so smoke debugging does not require GPUs.

- [x] **Step 4: Run all scenarios and verify invariants**

Run: `python scripts/simulate_adaptive_supervision.py --all`

Expected: finite metrics, monotonic mastery, non-increasing supervision, and correct endpoint bounds.

### Task 6: Regression Suite and Runtime Smoke

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-adaptive-supervision-controller.md`

- [x] **Step 1: Run the complete focused regression suite**

Run all adaptive, routing, quality-gate, config, runner, and existing schedule tests. Expected: all pass.

- [x] **Step 2: Dry-run the new experiment variant**

Run the runner's dry-run mode and inspect exported environment plus resolved config. Expected: adaptive enabled; legacy weight decay, route-cap schedule, and dynamic monitor disabled.

- [x] **Step 3: Run a short 8-GPU smoke**

Run 4-8 optimizer steps with the new variant. Expected: all ranks enter training, `adaptive/*` metrics are finite, effective sampling is active, controller actions agree with the logged snapshot, and no OOM/rank divergence occurs.

- [x] **Step 4: Analyze smoke output and fix defects with new failing tests**

For every runtime defect, add a reproducing failing test before the implementation fix, rerun the focused tests, and repeat the smoke only when needed.

- [x] **Step 5: Mark the plan complete only after runtime evidence is recorded**

Update checkboxes and summarize the final smoke run path, log path, controller trajectory, and any remaining training-quality uncertainty.

## Runtime Evidence

- Four-step 8-GPU smoke output: `outputs/test-fast/pcd-no-visual/adaptive_supervision_smoke_20260712_1515/deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision`
- Four-step log: `outputs/test-fast/logs/pcd_no_visual_adaptive_supervision_smoke_20260712_1515/deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision/train_opd_7b_dyme_probe_20260712_153345.log`
- At step 4, global readiness reached `0.0156254`; the shared snapshot produced supervision `0.982642`, OPD weight `1.482642`, teacher weight `0.491321`, and cap `8`.
- One-step clean-exit output: `outputs/test-fast/pcd-no-visual/adaptive_supervision_exit_smoke_20260712_1550/deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision`
- One-step log: `outputs/test-fast/logs/pcd_no_visual_adaptive_supervision_exit_smoke_20260712_1550/deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision/train_opd_7b_dyme_probe_20260712_154548.log`
- The clean-exit smoke saved `final_checkpoint` and returned runner status `0`.
- Remaining uncertainty: short smoke runs validate synchronization and wiring, but not whether the learned signal reaches the target readiness early enough or improves final ChartQA accuracy. That requires the full training/evaluation run.
