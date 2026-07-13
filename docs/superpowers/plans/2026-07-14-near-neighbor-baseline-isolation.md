# Near-Neighbor Baseline Semantic Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misleading VOLD/SSOPD executable labels with one honestly named mixed-group hard-replay diagnostic and block unimplemented matched baselines.

**Architecture:** Keep the existing hard-replay algorithm unchanged while renaming its public contract from runner through trainer metrics. Remove VOLD from the executable matrix until a true two-stage implementation exists, and use tests to prove both the new contract and the absence of silent legacy aliases.

**Tech Stack:** Bash runners, Python/PyTorch trainer configuration, pytest, Markdown experiment ledger.

---

### Task 1: Lock Runner Semantics With Failing Tests

**Files:**
- Modify: `tests/test_chartqa_10epoch_ablation_matrix.py`
- Modify: `tests/test_pcd_no_visual_runner.py`

- [ ] Replace expected matrix labels with `mixed_group_shortest_correct_hard_replay` and assert `vold_cold_start`/`ssopd_mixed_group` are absent.
- [ ] Require the new PCD variant to export `DYME_MIXED_GROUP_HARD_REPLAY=1` and no legacy SSOPD variable.
- [ ] Require requests for retired labels to fail with an explicit semantic-isolation error.
- [ ] Run the focused tests and confirm they fail because the old labels still exist.

### Task 2: Rename the Executable Contract

**Files:**
- Modify: `scripts/test/run_chartqa_10epoch_ablation_matrix.sh`
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `config/config_opd_7b_dyme_probe.py`

- [ ] Replace the old matrix and PCD variant names with the honest hard-replay name.
- [ ] Export and consume `DYME_MIXED_GROUP_HARD_REPLAY`; reject `DYME_SSOPD_MIXED_GROUP` when set.
- [ ] Remove VOLD from the executable matrix and return a clear error for retired labels.
- [ ] Run the runner tests and confirm they pass.

### Task 3: Rename Internal Algorithm And Metrics

**Files:**
- Modify: `opsd_utils/self_distill.py`
- Modify: `trainer/DyMETrainer.py`
- Modify: `opsd_utils/health_monitor.py`
- Modify: `tests/test_mixed_group_hard_replay.py`
- Modify: `tests/test_health_monitor.py`

- [ ] First update tests to require `MixedGroupHardReplayPlan`, `build_mixed_group_hard_replay_plan`, and honest routing metric names.
- [ ] Run tests and confirm import/metric failures.
- [ ] Rename the implementation and trainer variables without changing target-selection behavior.
- [ ] Run unit tests and confirm green.

### Task 4: Synchronize Experiment Documentation

**Files:**
- Modify: `docs/chartqa_10epoch_ablation_migration.md`
- Modify: `docs/opd_experiment_plan.md`
- Modify: `docs/paper_reconstruction/experiment_ledger.md`
- Modify: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Modify: `docs/paper_reconstruction/experiment_plot_plan.md`

- [ ] Rename the runnable diagnostic everywhere it is presented as a current experiment.
- [ ] Keep VOLD and SSOPD only in literature/comparator requirements, clearly marked unimplemented.
- [ ] Record that the hard-replay path is not eligible for matched-neighbor claims.

### Task 5: Regression And Frozen-Run Audit

**Files:**
- Verify: `tests/test_chartqa_10epoch_ablation_matrix.py`
- Verify: `tests/test_pcd_no_visual_runner.py`
- Verify: `tests/test_mixed_group_hard_replay.py`
- Verify: `tests/test_health_monitor.py`
- Verify: `scripts/analysis/check_frozen_run_env.py`

- [ ] Run focused pytest for runner, config, trainer helper, and health metrics.
- [ ] Search production code for retired labels and ensure only explicit rejection text remains.
- [ ] Dry-run the frozen Oracle variant and run its environment checker to prove unchanged semantics.
- [ ] Append the verified result to the long-run `launch.info` without changing the queued command.
