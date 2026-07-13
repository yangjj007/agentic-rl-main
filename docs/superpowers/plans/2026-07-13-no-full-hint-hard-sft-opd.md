# No-Full-Hint Hard-SFT OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a matched OPD variant in which teacher trajectory, teacher repair, legacy online-SFT slots, and every full ChartQA hint hard target are verifiably absent.

**Architecture:** Environment-backed gate controls disable ordinary online SFT and select a condition-aware non-SFT fallback for teacher-probe failures. Trainer assembly records source-specific hard-target counts, while the external monitor normalizes escaped candidate previews and stops on any invariant violation. Legacy variants retain their current behavior.

**Tech Stack:** Python, PyTorch, Accelerate, pytest, Bash, tmux, 8-GPU torchrun.

---

### Task 1: Correct Escaped Candidate Monitoring

**Files:**
- Modify: `scripts/analysis/check_opd_template_health.py`
- Modify: `tests/test_opd_template_health_check.py`

- [x] **Step 1: Write failing monitor tests**

Add a candidate preview containing escaped `\\n` separators and assert that the monitor recognizes all five sections, canonical `Answer:`, malformed answers, and full-template rate. Add a metric row with non-zero legacy hard-target exposure and assert mechanism-violation exit code 2.

- [x] **Step 2: Verify RED**

Run: `/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_opd_template_health_check.py -q`

Expected: escaped multiline and new invariant assertions fail.

- [x] **Step 3: Implement normalized parsing and invariant checks**

Normalize diagnostic text with:

```python
def normalize_candidate_preview(text: str) -> str:
    return str(text or "").replace("\\r\\n", "\n").replace("\\n", "\n")
```

Use `summarize_template_behavior` plus a canonical Answer regex to emit full,
partial, Goal-without-Answer, canonical Answer, and malformed Answer rates. Treat
non-zero `routing/legacy_online_sft_rate` or
`routing/full_hint_hard_target_rate` as mechanism violations.

- [x] **Step 4: Verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 2: Expose No-Hard-SFT Configuration

**Files:**
- Modify: `config/config_opd_7b_dyme_probe.py`
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_config_antidegen.py`
- Modify: `tests/test_pcd_no_visual_runner.py`

- [x] **Step 1: Write failing config and runner tests**

Assert that `DYME_DISABLE_ONLINE_SFT_SLOTS=1`,
`DYME_ONLINE_SFT_ON_ALL_WRONG=0`, and
`DYME_TEACHER_PROBE_FAILURE_ROUTE=mixed_grpo_all_wrong_skip` resolve to the OPD
gate and are exported by a new reproducible variant.

- [x] **Step 2: Verify RED**

Run: `/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_config_antidegen.py tests/test_pcd_no_visual_runner.py -q`

Expected: missing environment overrides and variant assertions fail.

- [x] **Step 3: Add minimal environment-backed config and variant**

Read the new variables with existing `env_bool`/`env_str` helpers. Add variant
`deplot_no_vs_opd_pcd_oracle_hint_opd_no_full_hint_hard_sft_adaptive_supervision`
matched to the stopped run, but export all three no-hard-SFT controls.

- [x] **Step 4: Verify GREEN**

Run the Task 2 pytest command. Expected: all tests pass and legacy defaults remain unchanged.

### Task 3: Condition-Aware Teacher-Probe Failure Routing

**Files:**
- Modify: `opsd_utils/signal_aware_routing.py`
- Modify: `trainer/DyMETrainer.py`
- Create: `tests/test_teacher_probe_failure_routing.py`

- [x] **Step 1: Write failing pure routing tests**

Define a helper API:

```python
teacher_probe_failure_mode(
    *, group_has_correct: bool, route: str
) -> int
```

Assert `mixed_grpo_all_wrong_skip` returns `MODE_GRPO` for mixed groups and
`MODE_SKIP` for all-wrong groups; `sft` preserves legacy `MODE_SFT`; invalid
routes raise `ValueError`.

- [x] **Step 2: Verify RED**

Run: `/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_teacher_probe_failure_routing.py -q`

Expected: helper import fails.

- [x] **Step 3: Implement helper and wire all failure sites**

Use the helper for no-evidence, probe-budget overflow, and teacher-wrong cases.
Pass group correctness already available to `_apply_teacher_probe_routing`.
Candidate records must log final route names `grpo_*` or `skip_*`, never SFT,
under the new route.

- [x] **Step 4: Verify GREEN**

Run the Task 3 pytest command. Expected: all tests pass.

### Task 4: Source-Specific Hard-Target Metrics

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Modify: `opsd_utils/health_monitor.py`
- Modify: `tests/test_health_monitor.py`
- Create: `tests/test_online_sft_source_metrics.py`

- [x] **Step 1: Write failing metric tests**

Exercise route-source classification and assert separate counts/rates for
legacy slot SFT, routed fallback SFT, forced SFT, aggregate legacy online SFT,
and full-hint hard targets. Assert the no-hard-SFT configuration yields exact
zeros.

- [x] **Step 2: Verify RED**

Run: `/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_online_sft_source_metrics.py tests/test_health_monitor.py -q`

Expected: new metric keys are missing.

- [x] **Step 3: Add minimal source classification and health mappings**

At the `hint + answer` replacement branch, classify the trigger before
replacement and append `routing/legacy_online_sft_rate` and
`routing/full_hint_hard_target_rate`. Map both into health output and add a
hard-imitation alert if either is non-zero while the no-hard-SFT gate is active.

- [x] **Step 4: Verify GREEN**

Run the Task 4 pytest command. Expected: all tests pass.

### Task 5: Focused Regression and Dry Run

**Files:**
- Modify: `docs/superpowers/plans/2026-07-13-no-full-hint-hard-sft-opd.md`

- [x] **Step 1: Run focused regression suite**

Run all routing, runner, config, monitoring, diagnostics, controller, and
effective-sampling tests. Expected: all pass.

- [x] **Step 2: Dry-run the new variant**

Run the runner with `--dry-run`. Expected exports:

```text
DYME_TEACHER_TRAJECTORY=0
DYME_DISABLE_ONLINE_SFT_SLOTS=1
DYME_ONLINE_SFT_ON_ALL_WRONG=0
DYME_TEACHER_PROBE_FAILURE_ROUTE=mixed_grpo_all_wrong_skip
DYME_ADAPTIVE_SUPERVISION=1
```

- [x] **Step 3: Record resolved configuration evidence**

Import the generated config under the dry-run environment and assert the
resolved gate contains the exact booleans and fallback route.

### Task 6: Multi-Scenario 8-GPU Smoke

**Files:**
- Modify only if a smoke defect receives a failing regression test first.

- [x] **Step 1: Launch a short 8-GPU smoke**

Run 4--8 optimizer steps with the new variant in tmux. Expected: no OOM/NCCL
failure and clean checkpoint exit.

- [x] **Step 2: Validate runtime invariants**

Inspect every step for exact-zero teacher trajectory, teacher repair, legacy
online SFT, and full-hint hard-target rates. Confirm OPD appears on verified
wrong completions and GRPO appears when correct completions exist.

- [x] **Step 3: Simulate failure branches**

Use focused unit/smoke fixtures for teacher-wrong, no-evidence, and budget
overflow. Confirm mixed groups route GRPO and all-wrong groups skip.

- [x] **Step 4: Fix defects via RED-GREEN cycles**

For each runtime defect, add a failing test, observe RED, implement the minimal
fix, rerun focused tests, and repeat smoke only after GREEN.

### Task 7: Full 4epoch Run and Automatic Eval

**Files:**
- Update: `docs/paper_reconstruction/experiment_ledger.md`
- Update: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Update: `docs/opd_main_training_eval_results.md`

- [x] **Step 1: Launch full run in tmux**

Use 8 GPUs, four epochs, unique run/output/log/state paths, and automatic final
8-GPU ChartQA eval with `DYME_EVAL_BATCH_SIZE=1`.

- [x] **Step 2: Launch independent monitor**

Check process health, OOM/NCCL/traceback, hard-target invariants, route rates,
zero-loss, all-wrong, accuracy, degeneration, clip/EOS, and corrected template
behavior at scheduled windows.

- [ ] **Step 3: Apply health gates**

Stop immediately on mechanism violation. Apply the matched step-60 gate from
the spec. Otherwise continue through final checkpoint and eval.

- [ ] **Step 4: Parse final result and continue the experiment loop**

Only a valid summary with processed at least 2496/2500 can guide iteration;
prefer 2500/2500 for the paper. If accuracy is not above 0.60, preserve
forensics, choose the next evidence-based intervention, and repeat TDD/smoke/full
run. If above 0.60, synchronize all paper artifacts and completion evidence.

## Plan Self-Review

- Every design invariant maps to Tasks 1--4.
- Legacy compatibility is explicitly tested in Tasks 2 and 3.
- Full training is gated on unit, dry-run, resolved-config, and distributed
  runtime evidence.
- No heading penalty or CoT ban is introduced.
- No placeholders or undefined implementation decisions remain.
- This workspace has an empty `.git` directory, so commit steps are omitted;
  artifacts and test output provide the audit trail.

## Execution Record

- The first distributed smoke exposed forced malformed-completion replacement as
  a second hard-target bypass. Its reproducing trainer test failed before the
  bypass fix; the final smoke
  `no_full_hint_hard_sft_smoke_20260713_145520` completed with all hard-target
  invariants at zero.
- The focused implementation and resilience regression was rerun after spec
  approval on 2026-07-13: `107 passed`.
- Runner dry-run resolves teacher trajectory off, online-SFT slots off,
  all-wrong online SFT off, condition-aware non-SFT fallback, adaptive effective
  sampling from step zero, and step checkpoints every 50 steps.
- The full resilient run is queued in tmux as
  `dyme_no_full_hint_resilient_181613`; its independent watcher and post-eval
  forensics sessions are alive. It remains at `waiting_for_gpu_gate` while
  external compute processes occupy part of the eight-GPU node.
- Replaying the exact watcher against the interrupted clean run through step 86
  returns `rc=0` and `status=ok`: full-template, empty-skeleton, malformed-Answer,
  and all hard-target maxima are zero. This confirms that partial student style
  drift alone does not terminate the queued run.
- A later evidence audit found that the watcher recovery gate and low-score
  forensic replay still read rank-local `routing/grpo_route_rate`, although the
  approved controller uses `global_signal/grpo_route_rate`. Test-first fixes now
  use global-first, explicit local fallback for legacy logs, and report source
  fractions. The exact step-86 watcher replay remains `rc=0/status=ok` with
  `grpo_route_global_fraction=1.0`; the combined monitor/forensic regression is
  `45 passed`. The queued watcher reloads the script on every poll, so no tmux
  restart is required before training begins.
- A final frozen prelaunch regression at `2026-07-13 22:18 CST` covers the
  runner, no-full-hint hard-target gates, route-source accounting, global route
  snapshots, adaptive controller, health monitor, low-score forensics, paper
  artifact parser, and resilient launcher: `95 passed`. No training parameter
  changed after this check.
- Runtime configuration is now independently audited after the real
  `run_env.json` appears. `scripts/analysis/check_frozen_run_env.py` checks 25
  frozen fields, including 4 epochs, 50-step checkpoints, all hard-SFT gates,
  effective sampling from step zero, and the global-GRPO controller. The
  `dyme_no_full_hint_resilient_181613_preflight` tmux stops the training session
  on any mismatch. A historical clean-run replay correctly reports the five
  old save/sampling fields as violations, and the expanded regression is
  `98 passed`.
