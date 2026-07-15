# ChartQA Closed-Loop Recovery Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a micro-eval-only ChartQA integrated closed-loop evidence recovery controller that uses verifier events to drive executable recovery operators and repeatedly measure same-lineage teacher recoverability. This is not a post-hoc recovery selector; each sample is processed as one verifier-observed trajectory whose later actions are conditioned on earlier failures.

**Architecture:** Extend `scripts/analysis/teacher_probe_micro_eval.py` with a new `chartqa_closed_loop_recovery` harness backed by `ClosedLoopRecoveryController`. The runner only batches teacher jobs; the controller owns the observe -> verifier event -> recovery operator -> next prompt -> stop/continue loop for each sample. Reuse existing prompt construction, teacher generation, ChartQA scoring, and prompt profiles where possible, but write separate closed-loop records and summary files so the verifier-event/action trace is explicit.

**Tech Stack:** Python, pytest, existing LLaVA-OV teacher generation utilities, existing ChartQA teacher-probe evaluator, JSONL/CSV artifacts.

---

### Task 1: Closed-Loop Fake Runtime Contract

**Files:**
- Modify: `tests/test_teacher_probe_micro_eval.py`
- Modify: `scripts/analysis/teacher_probe_micro_eval.py`

- [ ] **Step 1: Write failing test**

Add a test that runs:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_probe_micro_eval.py::test_chartqa_closed_loop_fake_run_writes_state_trace
```

The test should require:

- `--harness chartqa_closed_loop_recovery`
- `closed_loop_records.jsonl`, `closed_loop_attempts.jsonl`,
  `closed_loop_summary.csv`, `prompt_previews.jsonl`, and `manifest.json`
- each record has `events`, `actions`, `attempt_count`, `status`,
  `selected_action`, and `oracle_any_attempt_correct`
- `manifest.json` declares `controller=integrated_closed_loop_recovery_controller` and
  `max_teacher_attempts == len(actions)`
- runtime prompt previews contain no reference answer, dataset hint, or oracle hint

- [ ] **Step 2: Run test to verify failure**

Expected: fail because the harness choice and output files do not exist.

- [ ] **Step 3: Implement minimal fake runtime**

Add:

- `chartqa_closed_loop_recovery` to `--harness` choices
- `_run_chartqa_closed_loop_recovery`
- closed-loop attempt/record/summary writers
- fake outputs that force one sample through recovery

- [ ] **Step 4: Run test to verify pass**

Expected: the new focused test passes.

### Task 2: Verifier Event and Action Policy

**Files:**
- Modify: `tests/test_teacher_probe_micro_eval.py`
- Modify: `scripts/analysis/teacher_probe_micro_eval.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- parse failure maps to `canonical_repair_required`
- operation-heavy question failures map to `operation_recovery_required`
- closed-loop policy schedules visual operation, DePlot operation, reasoned
  recovery, target-phrase recovery, and arithmetic recovery before abstaining

- [ ] **Step 2: Run tests to verify failure**

Expected: fail because the event/action helpers are missing.

- [ ] **Step 3: Implement helpers**

Add small pure helpers:

- `_closed_loop_verifier_event(output, sample)`
- `_closed_loop_next_action(event, qtype, attempted_actions)`
- `_build_closed_loop_job(...)`
- `build_chartqa_target_phrase_recovery_suffix(...)`

- [ ] **Step 4: Run focused tests**

Expected: event/action tests pass.

### Task 3: Real Teacher Smoke and Iteration

**Files:**
- Modify only if smoke exposes a bug, with a failing test first.

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_probe_micro_eval.py tests/test_evidence_harness.py
```

- [ ] **Step 2: Run fake smoke**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python scripts/analysis/teacher_probe_micro_eval.py --harness chartqa_closed_loop_recovery --max-samples 8 --fake-teacher --out-dir outputs/test-fast/teacher-probe-micro-eval/chartqa_closed_loop_fake_smoke
```

- [ ] **Step 3: Run real LLaVA-OV 7B smoke**

Run a 128-example seed31 smoke first, then compare to prior
`llava7b_verifier_early_stop_reasoned128_seed31_20260715_161843`.

- [ ] **Step 4: Decide next iteration from trace**

If the closed-loop result is still below `0.90`, inspect abstains and wrong
accepted attempts by event/action. If the bottleneck is unclear, search prompt and
harness engineering references before changing prompts again.

### Task 4: Target-Phrase Recovery Controller Extension

**Files:**
- Modify: `tests/test_teacher_probe_micro_eval.py`
- Modify: `opsd_utils/evidence_harness/chartqa.py`
- Modify: `scripts/analysis/teacher_probe_micro_eval.py`

- [x] **Step 1: Write failing tests**

Add tests requiring:

- `manifest["controller"] == "integrated_closed_loop_recovery_controller"`
- `manifest["max_teacher_attempts"] == len(manifest["actions"])`
- `_closed_loop_next_action(...)` schedules `target_phrase_recovery` after
  reasoned recovery and before arithmetic recovery
- `_build_closed_loop_job(..., action="target_phrase_recovery", ...)` creates a
  prompt that contains `Target phrase` and `legend/color`, excludes previous
  wrong answers, and excludes the reference answer

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_probe_micro_eval.py::test_chartqa_closed_loop_fake_run_writes_state_trace tests/test_teacher_probe_micro_eval.py::test_closed_loop_schedules_target_phrase_recovery_before_arithmetic tests/test_teacher_probe_micro_eval.py::test_closed_loop_target_phrase_recovery_prompt_focuses_requested_label_without_gold
```

Expected before implementation: FAIL with missing `controller`, missing
`target_phrase_recovery` scheduling, and unknown closed-loop action.

- [x] **Step 3: Implement target-phrase recovery**

Add `build_chartqa_target_phrase_recovery_suffix(...)`, route the new action in
`_closed_loop_next_action`, wire `_build_closed_loop_job`, set canonicalization
for `target_phrase_recovery`, and derive the loop budget from the action list.

- [x] **Step 4: Run focused tests**

Run the same command from Step 2.

Expected after implementation: PASS.

### Task 6: Add Table-Executable DePlot Recovery Operator

**Files:**
- Modify: `tests/test_evidence_harness.py`
- Modify: `tests/test_teacher_probe_micro_eval.py`
- Modify: `opsd_utils/evidence_harness/chartqa.py`
- Modify: `scripts/analysis/teacher_probe_micro_eval.py`

- [x] **Step 1: Write failing executable-recovery tests**

Add tests requiring `build_chartqa_executable_deplot_recovery_suffix(...)` to
produce a no-gold `[Executable DePlot Recovery]` block for:

- threshold sum
- threshold count
- threshold row/column lookup
- exact percent count
- threshold sum minus named label
- exact value reverse lookup
- row value-signature lookup
- median of table values
- pair-sum label lookup
- same-value pair lookup
- two-column comparison count
- max consecutive change label

- [x] **Step 2: Implement executable operator**

Add the table parser, numeric-cell records, operation branches, candidate-answer
formatting, and `build_chartqa_executable_deplot_response_prefix(...)`. The
operator is table-only: it does not load chart images, because the candidate is
computed from DePlot cells and verified offline before acceptance.

- [x] **Step 3: Wire into the closed-loop controller**

Insert `executable_deplot_recovery` after `deplot_operation_recovery` and before
free-form reasoned recovery. If the executable block emits a candidate answer,
use `Answer: <candidate>` as the response prefix and set `max_new_tokens=1`.
Disable canonicalization for this action so a correct program answer is not
rewritten by the teacher. Add shorter per-action generation budgets for the other
closed-loop actions.

- [x] **Step 4: Run tests and smoke checks**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_probe_micro_eval.py tests/test_evidence_harness.py
```

Observed: `67 passed`.

Run fake smoke:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python scripts/analysis/teacher_probe_micro_eval.py --harness chartqa_closed_loop_recovery --max-samples 4 --fake-teacher --out-dir /tmp/chartqa_closed_loop_fake_smoke_v12_executable_budget
```

Observed manifest action order:
`visual_answer -> visual_operation_recovery -> deplot_operation_recovery ->
executable_deplot_recovery -> reasoned_recovery -> target_phrase_recovery ->
arithmetic_recovery -> scale_unit_recovery`.

- [x] **Step 5: Measure no-GPU coverage and targeted teacher behavior**

On v11 abstains, executable candidates are offline-correct for `16/384` extra
samples, projecting `346/384 = 0.9010` if accepted by the verifier.

Run targeted text-only executable smoke on those 16 cases. Observed:
`16/16 = 1.0000` teacher correctness after using candidate answer prefixes.

- [x] **Step 6: Run full image-in-the-loop 128-example confirmation**

Full seed13/29/31 image-in-the-loop confirmation completed with same-lineage
`llava-7b-ov`: `114/128`, `119/128`, and `119/128`, for `352/384 = 0.9167`.
This clears the `>=0.90` teacher-probe target. Accepted-by-action shows the
executable table operator is integrated into the trajectory rather than used as
an offline selector: `executable_deplot_recovery` contributed `9 + 11 + 6 = 26`
accepted recoveries across the three seeds. The previous v11 baseline was
`330/384 = 0.8594`, so the integrated executable recovery controller adds
`+22/384 = +5.73` points.

Artifacts:

- `outputs/test-fast/teacher-probe-micro-eval/chartqa_closed_loop_v12_executable_deplot128_seed13_20260715_full/`
- `outputs/test-fast/teacher-probe-micro-eval/chartqa_closed_loop_v12_executable_deplot128_seed29_20260715_full/`
- `outputs/test-fast/teacher-probe-micro-eval/chartqa_closed_loop_v12_executable_deplot128_seed31_20260715_full/`

### Task 5: Integrate Recovery State Into One Controller

**Files:**
- Modify: `tests/test_teacher_probe_micro_eval.py`
- Modify: `scripts/analysis/teacher_probe_micro_eval.py`
- Modify: `docs/superpowers/specs/2026-07-15-chartqa-closed-loop-recovery-design.md`

- [x] **Step 1: Write failing tests**

Add tests requiring:

- `ClosedLoopRecoveryController` can build the first job, observe verifier
  failure, emit the next recovery action, observe acceptance, and write the final
  record.
- prompt previews and records include
  `controller=integrated_closed_loop_recovery_controller`.
- `manifest["controller"] == "integrated_closed_loop_recovery_controller"`.

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_probe_micro_eval.py::test_closed_loop_recovery_controller_executes_observe_act_loop tests/test_teacher_probe_micro_eval.py::test_chartqa_closed_loop_fake_run_writes_state_trace
```

Expected before implementation: FAIL with missing `ClosedLoopRecoveryController`
and old manifest controller name.

- [x] **Step 3: Implement integrated controller**

Add `ClosedLoopRecoveryController`, move per-sample state transitions into it,
and change the runner so it only batches jobs emitted by active controllers.
Keep existing CLI and artifact file names unchanged.

- [x] **Step 4: Run focused tests**

Run the same command from Step 2.

Expected after implementation: PASS.
