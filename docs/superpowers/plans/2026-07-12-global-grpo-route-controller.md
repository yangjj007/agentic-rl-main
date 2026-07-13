# Global GRPO Route Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive adaptive supervision from the globally reduced final GRPO completion route rate.

**Architecture:** Extend the pure adaptive controller with a direct-signal update that reuses its monotonic smoothstep action derivation. The trainer selects the signal source from config, skips the legacy pre-route update in direct mode, and applies the globally reduced route snapshot after final routing for the next step.

**Tech Stack:** Python, PyTorch, Accelerate, pytest, Bash experiment runner.

---

### Task 1: Direct Signal Controller

**Files:**
- Modify: `opsd_utils/adaptive_supervision.py`
- Modify: `tests/test_adaptive_supervision.py`

- [ ] Write failing tests for direct EMA, conservative initialization, monotonic mastery, endpoint actions, and duplicate-step idempotence.
- [ ] Run RED and confirm missing `update_signal` behavior.
- [ ] Implement `update_signal(step, signal_rate)` using direct EMA and the existing action derivation.
- [ ] Run GREEN for pure controller tests.

### Task 2: Trainer Signal Source

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Modify: `tests/test_adaptive_supervision_trainer.py`
- Modify: `tests/test_global_training_signal_trainer.py`

- [ ] Write failing tests for `global_grpo_route` source selection.
- [ ] Verify legacy pre-route update is skipped in direct mode.
- [ ] Verify a global snapshot updates the direct controller once and logs signal metrics.
- [ ] Implement trainer source selection and post-route update.
- [ ] Run GREEN for focused trainer tests.

### Task 3: Config and Variant

**Files:**
- Modify: `config/config_opd_7b_dyme_probe.py`
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_config_antidegen.py`
- Modify: `tests/test_pcd_no_visual_runner.py`

- [ ] Write failing tests for `DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route` and target `0.30`.
- [ ] Add config parsing and adaptive variant exports.
- [ ] Run GREEN for config and runner tests.

### Task 4: Distributed Verification and Restart

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-global-grpo-route-controller.md`

- [ ] Run the focused regression suite.
- [ ] Run a 4-step 8-GPU smoke and verify direct signal metrics.
- [ ] Run a 30-step comparison and verify supervision remains high when GRPO route remains low.
- [ ] Start the full 4-epoch tmux run with automatic monitoring and final evaluation.
