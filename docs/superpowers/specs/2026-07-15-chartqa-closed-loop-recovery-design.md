# ChartQA Integrated Closed-Loop Evidence Recovery Controller Design

## Motivation

The current same-lineage LLaVA-OV 7B teacher was below the target recoverability
level under post-hoc multi-view selection: the best previous multi-seed evidence
was `330/384 = 0.8594` across seeds `13/29/31`. That failure mode is important:
running several prompts and selecting a verified answer is only a diagnostic
tool. It does not repair the teacher's next attempt, expose the evidence surface
that failed, or turn verifier feedback into a new action.

This design therefore treats recoverability as one executable closed-loop
trajectory, not as a recovery selector. The verifier is a runtime control signal:
each rejection is converted into the next recovery operator, and each operator
chooses its own evidence surface and generation budget. The latest full
image-in-the-loop confirmation reaches `352/384 = 0.9167` accepted trajectory
accuracy across seeds `13/29/31` (`114/128`, `119/128`, `119/128`), exceeding the
`0.90` teacher-probe admission target. The table-executable DePlot operator
accounts for `26/384` accepted recoveries inside that trajectory, so it is a
stateful recovery step rather than a post-generation answer picker.

The recovery mechanism is an integrated observe-act-recover loop. The verifier
produces a runtime event; the controller turns that event into the next
executable recovery operator, builds the allowed evidence surface, runs the
teacher or table-only operator, canonicalizes the answer where needed, and
decides whether to stop or continue. The unit of work is the whole recovery
trajectory for a sample, where later prompts are conditioned on earlier verifier
failures.

## Scope

Implement a new micro-eval-only harness named `chartqa_closed_loop_recovery`.

Do not modify trainer routing, OPD loss, student eval, 4epoch runners, or any
training config. This is a teacher-probe harness for measuring recoverability.

## Runtime Contract

Each sample runs as a bounded state machine:

1. `visual_answer`: answer from the chart image with a short answer prefix.
2. verifier event: parse the output and score it offline.
3. if not accepted, emit the next recovery operator from the event and question
   type.
4. generate the recovery prompt from the operator, verifier event, allowed
   evidence surface, and, only where useful, the previous teacher answer.
5. repeat until accepted or the attempt budget is exhausted.

The teacher prompt may include:

- chart image
- question
- DePlot table
- previous teacher answer for actions designed to repair a failed draft
- event labels such as `parse_failed`, `numeric_mismatch`, or
  `operation_recovery_required`

The teacher prompt must not include:

- reference answer
- dataset hint
- oracle hint
- verifier score
- any text that reveals the correct answer

The runtime trace may store correctness labels after generation for analysis. Those
labels must not be included in any teacher prompt.

## Recovery Actions

The implementation supports these actions:

- `visual_answer`: initial image-native short answer.
- `visual_operation_recovery`: image-native operation prompt.
- `deplot_operation_recovery`: image plus DePlot operation prompt.
- `executable_deplot_recovery`: table-only executable DePlot operator. It
  computes matched cells and candidate answers for threshold sum/count, named
  subtraction, reverse lookup, median, pair-sum, same-value pair, column
  comparison count, and max adjacent change. If a candidate answer is produced,
  it is used as the answer prefix; wrong candidates are rejected by the verifier
  and the controller continues.
- `reasoned_recovery`: image plus DePlot compact reasoning with canonical final
  answer repair.
- `target_phrase_recovery`: image plus DePlot target-grounding prompt. It first
  identifies the exact entity, category, legend/color, series, time point, or
  threshold condition requested by the question, then returns the matching value
  or label. This action avoids anchoring on previous wrong answers.
- `arithmetic_recovery`: image plus DePlot operand/equation prompt with canonical
  final answer repair.
- `scale_unit_recovery`: image plus DePlot scale/unit prompt. It uses prior
  teacher attempts only as failed drafts and repairs answer-surface errors such
  as percent sign, decimal-vs-percent scale, and unwanted labels.

The action policy is intentionally simple:

- parse failure goes directly to `reasoned_recovery`.
- operation-heavy question types (`count`, `difference`, `average`, `percent`) go
  through visual operation, DePlot operation, executable DePlot recovery,
  reasoned recovery, target-phrase recovery, arithmetic recovery, and scale/unit
  recovery.
- other evidence failures use the same sequence so label/legend/threshold errors
  can be recovered before the controller abstains.

## Metrics

Write these artifacts:

- `closed_loop_records.jsonl`
- `closed_loop_attempts.jsonl`
- `closed_loop_summary.csv`
- `prompt_previews.jsonl`
- `manifest.json`

The manifest declares `controller=integrated_closed_loop_recovery_controller`
and `controller_contract=verifier_observe_act_recover`. Each record and attempt
also carries the controller name and step index so later analysis can inspect the
actual recovery trajectory rather than only the final answer.

Summary metrics:

- accepted trajectory accuracy (`selected_accuracy` in current CSV artifacts)
- accepted coverage
- abstain rate
- mean attempts
- accepted by action
- recovery success rate
- parse-fail event rate
- event/action counts
- oracle-any-attempt accuracy for diagnostic ceiling only

## Success Gate

This harness is not eligible for training integration until same-lineage LLaVA-OV
7B reaches approximately `0.90` selected accuracy or accepted precision on a
multi-seed ChartQA probe without gold answers in teacher prompts. The immediate
goal is to produce stronger diagnostics and recover more all-wrong operation-heavy
examples than the prior `0.8073` aggregate.
