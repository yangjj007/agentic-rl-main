# Forensic Eval Attempt Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent failed ChartQA eval retries from overwriting a valid forensic result for the same normalized checkpoint.

**Architecture:** Collect summary and raw-log rows as attempt candidates, retain their original labels and status metadata, then reduce candidates with one deterministic selection key. Keep one selected row per normalized checkpoint so existing plots and joins remain unchanged.

**Tech Stack:** Python 3, standard-library `csv`, `re`, `pathlib`, pytest.

---

### Task 1: Reproduce The Collision

**Files:**
- Modify: `tests/test_pcd_low_score_forensics.py`

- [x] **Step 1: Write a failing regression test**

Create a temporary `eval_chartqa/summary.csv` containing a valid 2,500-sample
final attempt followed by OOM and traceback attempts. Assert that
`collect_eval_rows()` returns the valid `0.5800` row and its original label.

- [x] **Step 2: Verify RED**

Run: `pytest -q tests/test_pcd_low_score_forensics.py -k final_eval`

Expected: FAIL because the current dictionary overwrite returns the last failed
attempt and does not expose `source_label`.

### Task 2: Select The Authoritative Attempt

**Files:**
- Modify: `scripts/analysis/pcd_low_score_forensics.py`
- Modify: `tests/test_pcd_low_score_forensics.py`

- [x] **Step 1: Implement candidate selection**

Add small helpers for normalized checkpoint labels, finite numeric parsing,
ChartQA completeness, clean status, and the deterministic attempt priority.
Collect all summary and raw-log attempts before selecting one per checkpoint.

- [x] **Step 2: Preserve audit fields**

Carry `source_label`, `exit_status`, and `errors` into the selected row and add
those fields to `checkpoint_accuracy.csv`.

- [x] **Step 3: Verify GREEN**

Run: `pytest -q tests/test_pcd_low_score_forensics.py`

Expected: all tests pass.

### Task 3: Verify Historical Behavior

**Files:**
- No production file changes expected.

- [x] **Step 1: Run focused parser and paper tests**

Run: `pytest -q tests/test_pcd_low_score_forensics.py tests/test_parse_eval_chartqa_logs.py tests/test_pcd_paper_artifacts.py`

Expected: all tests pass.

- [x] **Step 2: Re-run the historical forensic fixture**

Generate the temporary forensic report for the historical student-hint run and
verify `checkpoint_accuracy.csv` selects
`eval_final_checkpoint_bsz1_gpu0_20260709_192652` with accuracy `0.5800` and
processed count `2500`.
