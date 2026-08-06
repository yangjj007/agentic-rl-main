# Global Training Signal Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add globally reduced task, route, and output-health metrics that distinguish task-effective GRPO signal from total-reward variance.

**Architecture:** A pure aggregation module defines count fields and derives rates from globally summed tensors. `DyMETrainer` collects local counts before and after routing, performs one synchronized reduction per step, and logs one coherent `global_signal/*` snapshot without changing training behavior.

**Tech Stack:** Python, PyTorch, Accelerate distributed reduction, pytest.

---

### Task 1: Pure Global Signal Snapshot

**Files:**
- Create: `opsd_utils/global_training_signal.py`
- Create: `tests/test_global_training_signal.py`

- [ ] **Step 1: Write failing tests** for rate denominators, task/total zero disagreement, empty-count fallback, and finite output.
- [ ] **Step 2: Run RED:** `PYTHONPATH=. python -m pytest -q tests/test_global_training_signal.py` must fail because the module is missing.
- [ ] **Step 3: Implement frozen count and snapshot dataclasses** plus `snapshot_from_counts` using additive counts only.
- [ ] **Step 4: Run GREEN:** the Task 1 command must pass.

### Task 2: Trainer Collection and Reduction

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Create: `tests/test_global_training_signal_trainer.py`

- [ ] **Step 1: Write failing tests** for one reduce call, identical global snapshots, task-zero semantics, and `global_signal/*` metric publication.
- [ ] **Step 2: Run RED:** `PYTHONPATH=. python -m pytest -q tests/test_global_training_signal_trainer.py` must fail on missing trainer helpers.
- [ ] **Step 3: Collect pre-route task counts** from accuracy reward groups and total-reward advantages.
- [ ] **Step 4: Collect post-route counts** from final route masks plus completion clipping, EOS, degeneration, and accuracy sums.
- [ ] **Step 5: Reduce once and publish** the immutable `global_signal/*` snapshot.
- [ ] **Step 6: Run GREEN:** the Task 2 command and existing adaptive trainer tests must pass.

### Task 3: Runtime Diagnostics

**Files:**
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_pcd_no_visual_runner.py`

- [ ] **Step 1: Write a failing runner test** requiring the adaptive variant to enable global-signal logging.
- [ ] **Step 2: Run RED** and confirm the missing environment/config assertion.
- [ ] **Step 3: Add the environment toggle and resolved config snapshot field.**
- [ ] **Step 4: Run GREEN** for runner and config tests.

### Task 4: Verification and Restart

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-global-training-signal-observability.md`

- [ ] **Step 1: Run focused regression tests** for adaptive control, routing, config, and global signal metrics.
- [ ] **Step 2: Run a 4-step 8-GPU smoke** and verify finite identical global snapshots.
- [ ] **Step 3: Run a 20-30 step health smoke** and compare global versus rank-local metrics.
- [ ] **Step 4: Start the next 4-epoch tmux run** only if the global early-health gate passes.
- [ ] **Step 5: Attach automatic 8-GPU final evaluation** and continue the >0.60 loop.
