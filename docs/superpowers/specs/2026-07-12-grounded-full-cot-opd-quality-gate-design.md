# Grounded Full-CoT OPD Quality Gate Design

Date: 2026-07-12

## 1. Goal

Preserve the complete ChartQA reasoning format (`Goal`, `Observation`,
`Reasoning`, `Conclusion`, and `Answer`) while preventing unsupported or
internally inconsistent teacher trajectories from becoming OPD/SFT process
supervision.

This first implementation covers two experiments:

- G0: run a deterministic verifier as diagnostics without changing training;
- G1: use verifier quality to gate teacher full-trajectory supervision while
  leaving answer reward and official ChartQA evaluation unchanged.

Grounding, reasoning, consistency, or diversity scores will not be added to
the rollout reward in G0/G1. That is intentionally deferred until verifier
precision has been measured.

## 2. Motivation

The current local thinking reward measures POS-keyword overlap between student
reasoning and the hint. The current ChartQA format reward mostly checks for one
`Answer:` marker, a `Goal:` marker, minimum reasoning length, and obvious
degeneration. Neither mechanism verifies that chart values are supported or
that the calculation leads to the final answer.

Teacher probe acceptance currently relies on final-answer correctness. A
teacher can therefore be accepted even when its intermediate observation or
calculation is contradicted by the chart.

The training data also cannot be treated as uniformly correct process
supervision. `train_medium_vf_full.json` contains examples where the reference
answer, hint conclusion, and DePlot table disagree. The verifier must therefore
combine evidence conservatively rather than declare any single source to be
the universal process ground truth.

## 3. Non-Goals

G0/G1 will not:

- penalize the presence of ChartQA section headings;
- force answer-only or short-reasoning output;
- replace official reference answers;
- automatically rewrite or repair dataset labels;
- use DePlot token overlap as positive reward;
- add an online image-teacher judge to every rollout;
- implement a general natural-language theorem prover;
- add an additive template-diversity reward;
- change the normalized-progress or dynamic-trigger routing schedule.

## 4. Evidence Sources and Trust Policy

The verifier may observe four sources:

- `A`: official reference answer;
- `H`: dataset hint;
- `D`: DePlot table;
- `T`: generated teacher trajectory.

No source is automatically authoritative for process supervision. Official
answer evaluation continues to use `A`, but full-trajectory supervision uses
the following conservative policy:

| Quality | Meaning | Full trajectory use |
| --- | --- | --- |
| Q3 | Teacher answer is correct, structure is valid, no contradicted claim, and CoT agrees with its answer | Accept |
| Q2 | Teacher answer is correct and has no contradiction, but grounding or reasoning is partly unknown | Reject raw trajectory; eligible for a later grounded repair |
| Q1 | Teacher answer is correct but the trajectory is malformed or cannot support its conclusion | Reject trajectory; retain answer-level routing signal |
| Q0 | Teacher contains an explicit chart contradiction or CoT-answer contradiction | Reject trajectory and record conflict |

For G1 only Q3 teacher trajectories are admitted to teacher trajectory FKL.
Q2 does not trigger a new repair implementation in this change; it is logged
separately so a later experiment can add grounded full-CoT repair.

`unknown` is neutral. It must never be converted into `contradicted` merely
because the deterministic verifier does not understand a sentence.

## 5. Components

### 5.1 Structured CoT parser

Create `reward_utils/chart_cot_verifier.py` with a parser that extracts the
five ChartQA sections case-insensitively. It must:

- accept normal line breaks and escaped `\\n` sequences;
- preserve section text without lowercasing the returned content;
- use the final `Answer:` section as the answer;
- report duplicate, missing, and empty required sections;
- not require exact wording inside a section;
- not classify short but non-empty reasoning as invalid by length alone.

The output is a frozen dataclass `ParsedChartCoT` containing the section text,
`structure_valid`, `missing_sections`, `empty_sections`, and
`duplicate_sections`.

### 5.2 DePlot table parser

The same module exposes `parse_deplot_table`, returning a `ChartTable` with
column names and rows. It accepts either the JSON-encoded
`visual_fact_deplot` object used by the dataset or a raw table string.

The parser must:

- reject placeholders and empty tables;
- split pipe-delimited rows;
- retain labels and raw cell text;
- normalize whitespace;
- tolerate rows with fewer or additional cells by padding or truncating to the
  header width;
- expose normalized numeric values without discarding the raw values.

Numeric normalization supports commas, signs, percentages, decimals, and the
equivalence `0.39 == 39%` when one side explicitly uses a percent sign. It does
not generally equate `0.39` and `39` without percent context.

### 5.3 Claim-level grounding

G0/G1 uses a high-precision numeric grounding check. It extracts numeric
mentions from `Observation`, `Reasoning`, and `Conclusion`, excluding numbers
that appear only in the question when possible.

Each mention is classified as:

- `supported`: a normalized equivalent appears in the DePlot table;
- `contradicted`: the sentence binds a recognizable row/column label to a
  different table value;
- `unknown`: no reliable label-value binding can be established.

A bare number absent from DePlot is `unknown`, not contradicted. This avoids
penalizing valid derived values such as differences, counts, and averages.

The first version recognizes explicit label-value forms such as:

- `2019: 70`;
- `in 2019 ... is 70`;
- `the value for 2019 is 70`;
- `Revenue is 55` when `Revenue` is a row or column label.

### 5.4 Deterministic reasoning checks

The first version implements high-precision checks for:

- direct lookup;
- minimum and maximum values;
- argmin and argmax labels;
- count above or below a threshold;
- sum and difference when all operands are explicit;
- greater-than, less-than, and equality comparisons.

Each check returns `valid`, `invalid`, or `unknown`. Unrecognized language and
missing operands return `unknown`. A reasoning result may only be `invalid`
when the parsed operands and operation unambiguously contradict the stated
result.

### 5.5 CoT-answer consistency

The verifier compares normalized final-result candidates extracted from
`Reasoning`, `Conclusion`, and `Answer`.

- Agreement between Conclusion and Answer is `consistent`.
- An unambiguous mismatch is `inconsistent`.
- Missing or non-comparable candidates are `unknown`.
- Numeric percent/decimal equivalence follows the normalization policy above.
- Text answers use normalized case, whitespace, surrounding punctuation, and
  existing ChartQA answer normalization where reusable.

Reasoning-to-conclusion agreement is logged separately and is not required for
Q3 when no deterministic reasoning result can be extracted.

### 5.6 Trajectory quality classifier

`verify_chart_cot_trajectory` combines parser, table, grounding, reasoning,
and consistency outputs into `ChartCoTVerification`.

Q3 requires:

- teacher final answer passes the existing ChartQA answer evaluator;
- all five sections are present and non-empty;
- no duplicate `Answer:` section;
- no contradicted grounded claim;
- no deterministically invalid reasoning check;
- Conclusion and Answer are consistent.

Q0 is assigned when an explicit grounded contradiction, deterministic
reasoning contradiction, or Conclusion/Answer contradiction exists. Q1 covers
malformed trajectories. Remaining answer-correct trajectories are Q2.

The quality classifier never changes answer correctness.

## 6. Trainer Integration

Add a `chart_cot_quality_gate` section to `DYME_OPSD_CONFIG` with environment
overrides:

```text
DYME_CHART_COT_VERIFY=0
DYME_CHART_COT_GATE_MODE=diagnostic
DYME_CHART_COT_REQUIRE_Q3=1
DYME_CHART_COT_LOG_SAMPLES=1
DYME_CHART_COT_MAX_LOG_SAMPLES=8
```

Supported modes:

- `off`: no verification and legacy routing;
- `diagnostic`: compute and log quality, but preserve legacy teacher trajectory
  acceptance;
- `gate`: remove non-Q3 indices from the common process-supervision eligibility
  set after teacher generation and before either raw trajectory-loss
  tokenization or teacher-correct repair target construction.

The trainer must compute one `quality_eligible_indices` set and use it to gate
both existing process-supervision paths:

- raw teacher trajectory FKL/CE;
- `teacher_correct_repair` trajectory SFT.

Non-Q3 candidates must not be able to bypass the gate through the repair path.
The gate must not alter the completion's GRPO/OPD/SFT route in G1. It only
controls whether teacher process-supervision loss is present. This isolates the
effect of process-supervision quality from routing changes.

If a sample lacks real DePlot evidence, grounding checks become `unknown`; the
trajectory can still reach Q3 when it has valid structure, no contradictions,
and consistent Conclusion/Answer. Metrics record the missing-table condition.

## 7. Metrics and Samples

Log per optimizer step:

```text
cot_verify/enabled
cot_verify/gate_active
cot_verify/candidate_count
cot_verify/q3_rate
cot_verify/q2_rate
cot_verify/q1_rate
cot_verify/q0_rate
cot_verify/structure_valid_rate
cot_verify/deplot_available_rate
cot_verify/grounded_claim_count
cot_verify/supported_claim_rate
cot_verify/contradicted_claim_rate
cot_verify/unknown_claim_rate
cot_verify/reasoning_valid_rate
cot_verify/reasoning_invalid_rate
cot_verify/reasoning_unknown_rate
cot_verify/conclusion_answer_consistent_rate
cot_verify/conclusion_answer_inconsistent_rate
cot_verify/conclusion_answer_unknown_rate
cot_verify/teacher_traj_accepted_rate
cot_verify/teacher_traj_rejected_count
```

Candidate denominators must be explicit and zero-safe. Rates over grounding
claims use the number of extracted claims; quality rates use teacher trajectory
candidates.

Optional JSONL samples include the question, teacher response, quality class,
reason codes, parser summary, grounding claims, reasoning checks, and
consistency result. They must not include privileged provider prompts or model
logits.

## 8. Offline G0 Audit

Add `scripts/audit_chart_cot_quality.py`. It accepts:

- dataset JSON path;
- optional teacher/eval JSONL path;
- output directory;
- sample limit and random seed.

Dataset mode verifies hints as trajectories with `Answer:` appended from the
reference answer only for parser compatibility, while retaining a field that
identifies the synthesized answer line. Teacher-output mode evaluates the raw
teacher response.

Outputs:

- `chart_cot_quality_summary.json`;
- `chart_cot_quality_rows.jsonl`;
- `chart_cot_quality_conflicts.csv`.

The audit must support deterministic sampling and must never modify the input
dataset.

## 9. Template Diversity Diagnostics

G0 adds diagnostics without changing loss or reward:

- normalize reasoning by lowercasing, collapsing whitespace, replacing numeric
  values with `<NUM>`, and replacing exact chart labels with `<LABEL>`;
- compute exact normalized-template frequency;
- report unique-template rate, dominant-template rate, and the top 20
  templates;
- compute these statistics separately for Q3 and non-Q3 trajectories.

This first version does not use embedding similarity or pairwise model calls.
Exact normalized templates are deterministic, inexpensive, and sufficient to
detect severe template collapse.

## 10. Experiment Variants

Register two runner variants derived from the completed progress-scheduled
OPD experiment:

```text
deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_diagnostic
deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate
```

Both variants retain:

- full ChartQA reasoning format;
- normalized progress schedules;
- effective sampling;
- OPD decay and condition-aware overflow;
- disabled additive eval-format reward;
- existing answer accuracy reward.

The gate variant changes teacher SFT repair target style from
`student_hint_short` to `chartqa_hint` so accepted process supervision remains
full-CoT. Both raw teacher trajectory loss and teacher-correct repair are
subject to the same Q3 eligibility gate.

## 11. Validation and Go/No-Go

Before a 4-epoch G1 run, run G0 over at least 500 teacher trajectories and
manually inspect a stratified sample of at least:

- 30 Q3;
- 30 Q2;
- all Q0 up to 50;
- 30 Q1.

Proceed to G1 only when:

```text
explicit contradiction precision >= 0.90
Conclusion/Answer mismatch precision >= 0.95
CoT parser success rate >= 0.95
DePlot parse success rate >= 0.98 on real tables
verification runtime overhead <= 0.15 of rollout/reward time
```

Do not require low `unknown` rate to proceed. High precision is more important
than broad coverage for a gate that can remove teacher supervision.

The first G1 smoke run uses 20 optimizer steps. It must show nonzero candidate
counts, Q-class metrics summing to one within floating-point tolerance, and
teacher trajectory rejection only in `gate` mode.

## 12. Failure Handling

- Parser or verifier exceptions produce Q2, not Q0, and increment an error
  metric.
- Missing/placeholder DePlot produces unknown grounding.
- Empty teacher output remains Q1.
- Distributed ranks must enter identical collectives even when one rank has no
  teacher candidates.
- Quality-gate logging must be bounded and must not retain image tensors.
- Disabling the feature must preserve legacy behavior.

## 13. Deferred Follow-Up

After G1 establishes that quality-gated trajectory supervision improves or at
least preserves accuracy, a separate design may add:

- low-weight consistency and grounding rewards;
- Q2 grounded full-CoT repair;
- image-teacher adjudication for DePlot conflicts;
- diversity-aware candidate selection;
- quality-aware dynamic routing.

Those changes require separate ablations because combining them with G1 would
make the source of any accuracy change ambiguous.
