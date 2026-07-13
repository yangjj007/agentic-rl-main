# Full DyME Matched Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately selectable Full DyME comparator that differs from the matched Pure DyME comparator only by enabling Visual Checker, Visual Refiner, and IC prefetch.

**Architecture:** Keep `config_dyme_matched.py` as the single source of truth for the matched optimization and decoding budget. Build the Full variant by deep-copying that configuration and replacing only its `visual_supervision` block, while one runner selects the variant-specific config, output stem, and explicit visual environment flags.

**Tech Stack:** Python configuration modules, Bash runner, Accelerate, pytest.

---

### Task 1: Specify Full DyME parity behavior

**Files:**
- Modify: `tests/test_dyme_matched_runner.py`

- [x] **Step 1: Write failing configuration test**

Assert that Full DyME keeps OPD disabled, enables checker/refiner/prefetch, and retains matched learning rate and generation count.

- [x] **Step 2: Write failing runner test**

Invoke `run_dyme_matched_4epoch.sh --variant full --dry-run` and require a distinct Full output directory/config plus the same 4epoch, save, and 8-GPU evaluation budget.

- [x] **Step 3: Verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_dyme_matched_runner.py
```

Expected: two existing Pure tests pass; the Full config test fails because the module is absent and the Full runner test fails because `--variant` is unsupported.

### Task 2: Implement Full DyME configuration

**Files:**
- Create: `scripts/test/config/config_dyme_full_matched.py`
- Test: `tests/test_dyme_matched_runner.py`

- [x] **Step 1: Deep-copy the matched Pure DyME configuration**

Load the sibling `config_dyme_matched` module from its own directory so direct file-based imports and normal runner imports both work.

- [x] **Step 2: Replace only Visual Supervision**

Set `CONFIG["opsd"]["visual_supervision"]` from `build_visual_supervision_config()` and leave `CONFIG["opsd"]["enabled"]` false.

- [x] **Step 3: Run the focused configuration test**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_dyme_matched_runner.py::test_full_config_only_adds_visual_supervision
```

Expected: `1 passed`.

### Task 3: Add runner variant selection

**Files:**
- Modify: `scripts/test/run_dyme_matched_4epoch.sh`
- Test: `tests/test_dyme_matched_runner.py`

- [x] **Step 1: Parse and validate `--variant`**

Default to `pure`; accept only `pure` or `full`; select the title, output stem, config path, and visual flags from that value.

- [x] **Step 2: Keep the execution pipeline shared**

Use the selected values in the existing train command and diagnostics while preserving the same training budget, final-checkpoint requirement, 8-GPU ChartQA evaluation, and summary parser.

- [x] **Step 3: Run all comparator tests**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_dyme_matched_runner.py
```

Expected: `4 passed`.

### Task 4: Verify and document readiness

**Files:**
- Modify: `docs/opd_experiment_plan.md`
- Modify: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Modify: `outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/launch.info`

- [x] **Step 1: Run static and dry-run checks**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m py_compile scripts/test/config/config_dyme_matched.py scripts/test/config/config_dyme_full_matched.py
bash -n scripts/test/run_dyme_matched_4epoch.sh
bash scripts/test/run_dyme_matched_4epoch.sh --variant full --dry-run
```

Expected: all commands exit zero and no GPU compute process is created.

- [x] **Step 2: Run the relevant combined regression suite**

Run the comparator tests together with the frozen-run/configuration tests already protecting the queued oracle experiment. Expected: all pass.

- [x] **Step 3: Update paper and operations records**

Mark both matched Pure and Full DyME runners as implemented but `ready/not queued`; state that Full differs only in Visual Supervision and that neither comparator is launched before the current oracle clean OPD run completes its scheduled priority.

- [x] **Step 4: Recheck the oracle queue**

Confirm the four tmux sessions remain alive and the GPU gate still reflects only external compute or the intended oracle job.
