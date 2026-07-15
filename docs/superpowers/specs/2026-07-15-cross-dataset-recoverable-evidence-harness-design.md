# Cross-Dataset Recoverable-Evidence Harness Design

## Status

Approved design direction. This document specifies the architecture and
evaluation gates before implementation planning begins.

## Motivation

The current ChartQA experiments show that removing gold hints prevents
leakage, but it also reduces teacher correctness sharply. A structured
gold-hidden prompt improves the teacher, yet repeated sampling from the same
image-plus-DePlot configuration still leaves many examples wrong. Manual audit
also shows that DePlot is useful but imperfect: it may omit labels, lose
legend/color mappings, or provide evidence that the teacher selects or computes
incorrectly.

The solution must not treat DePlot as the task's authority. For ChartQA, the
chart image is the native evidence and remains visible to the production
teacher. DePlot is an optional, lossy evidence augmentation that can help with
table lookup and arithmetic but must be checked against the image when evidence
is ambiguous or conflicting.

The method must also cover the repository's other two target datasets:
A-OKVQA, where DePlot is inapplicable and recovery concerns visual grounding
and world knowledge, and GSM8K, where there is no image and recovery concerns
reasoning representation and deterministic arithmetic verification.

Therefore the design is a shared recoverability harness with dataset-specific
evidence adapters, not a ChartQA-specific table/image/fused router.

## Goals

1. Preserve every dataset's native deployment input in every production
   teacher attempt.
2. Acquire auxiliary evidence or verification tools only when the current
   attempt is unresolved, conflicting, or insufficiently supported.
3. Keep gold answers, reference rationales, and verifier outcomes out of the
   teacher prompt and out of runtime routing features.
4. Produce typed candidates whose evidence and reasoning can be validated.
5. Supply recoverable teacher distributions to OPD without hard-supervising a
   full teacher trajectory or fixed reasoning template.
6. Share controller, budget, trace, and OPD interfaces across ChartQA,
   A-OKVQA, and GSM8K while allowing task-appropriate evidence actions.
7. Separate production behavior from diagnostic ablations so experimental
   labels accurately describe which modalities were available.

## Non-Goals

- DePlot is not generalized to A-OKVQA or GSM8K.
- A table-only ChartQA attempt is not a production teacher route.
- The first implementation does not train a cross-dataset neural router.
- The first implementation does not add a contrastive loss for failed views.
- The harness does not restore gold hint SFT or full teacher-trajectory SFT.
- Valid JSON, model confidence, or answer-format compliance alone do not count
  as evidence correctness.

## Terminology

### Native Input

The information available to the student at deployment and required by the
task:

- ChartQA: question and chart image.
- A-OKVQA: question, image, and answer choices when present.
- GSM8K: problem text.

Native input is immutable across production teacher attempts.

### Evidence Action

A bounded operation that augments or verifies the native input, such as
attaching a DePlot table, selecting an object crop, extracting an equation, or
executing arithmetic.

### Evidence Configuration

The native input plus the evidence actions accumulated for one teacher
attempt. This replaces the ambiguous use of "view" in the earlier design.

### Diagnostic Isolation

An offline ablation that deliberately removes a native modality, such as a
ChartQA table-only probe. Diagnostic isolation measures marginal capability but
must never be reported as the production teacher path.

## Architecture

The architecture has four shared components and one dataset-specific boundary.

### Dataset Adapter

Each adapter declares:

```text
DatasetAdapter
- dataset_id
- native_input_schema
- allowed_actions
- candidate_schema
- quality_features
- validators
- action_costs
- terminal_answer_parser
```

The adapter owns task semantics. The shared controller never contains DePlot,
legend, object, or equation-specific logic.

### Harness State

```text
HarnessState
- example_id
- dataset_id
- native_input_digest
- attempts
- acquired_evidence
- unresolved_requirements
- conflict_state
- remaining_budget
- status
- accepted_attempt_id
```

Valid status values are:

```text
ACTIVE
ACCEPTED
ABSTAINED
BUDGET_EXHAUSTED
FATAL_FAILURE
```

### Candidate Contract

All tasks implement a common envelope:

```text
EvidenceCandidate
- answer
- evidence_refs
- operation
- unresolved_refs
- confidence
- support_relation
- raw_response_digest
```

Adapters may add typed fields. Confidence is diagnostic and cannot by itself
authorize acceptance.

### Validation Contract

Each validation stage returns:

```text
ValidationResult
- validator_id
- status: PASS | FAIL | UNKNOWN
- reason_code
- supporting_refs
- conflict_refs
- deterministic_value
```

The shared validation order is:

1. schema and parse validity;
2. provenance or source validity;
3. task-specific operation validity;
4. cross-evidence conflict detection;
5. calibrated acceptance decision.

### Shared Controller

The first controller is deterministic and auditable. It receives only
deployable features:

```text
current validation results
evidence quality features
unresolved requirements
conflict state
actions already attempted
remaining budget
```

It emits one action:

```text
GENERATE_BASE
ACQUIRE_EVIDENCE(action_id)
VERIFY
ACCEPT
ABSTAIN
```

Reference answers may label offline route outcomes and compute oracle upper
bounds, but are not controller inputs. A learned router may later be trained
from collected traces after deterministic routing is validated.

## Dataset Adapters

### ChartQA Adapter

#### Native Input

```text
question + full chart image
```

The full image remains available in every production teacher attempt.

#### Evidence Configurations

`visual_base`:

```text
question + full chart image
```

`visual_deplot`:

```text
question + full chart image + DePlot table
```

`visual_recovery`:

```text
question + full chart image + accumulated structured evidence
+ one or more targeted recovery results
```

#### Allowed Actions

```text
ATTACH_DEPLOT
TARGETED_CROP
LEGEND_GROUNDING
OCR_RECHECK
TABLE_IMAGE_CONFLICT_CHECK
EXECUTE_CHART_OPERATION
```

`ATTACH_DEPLOT` is an augmentation, not a replacement for the image.
`TARGETED_CROP`, `LEGEND_GROUNDING`, and `OCR_RECHECK` are triggered by
unresolved labels, color references, missing values, or conflicts.

#### Quality Features

- DePlot availability and placeholder status;
- table row and column count;
- missing-cell and repeated-zero rate;
- label coverage relative to question terms;
- color, legend, spatial, or rightmost/leftmost references in the question;
- unresolved series-to-color mapping;
- table/image agreement state;
- arithmetic executability.

#### Validators

- table cell and label existence;
- row/column orientation and alignment;
- legend/color-to-series grounding;
- visual locator validity;
- executable lookup, count, comparison, sum, difference, ratio, average, or
  percentage operation;
- table/image relation: `AGREE`, `CONFLICT`, or `UNKNOWN`.

#### Diagnostic-Only Configurations

`deplot_only` may be used on a fixed probe set to measure DePlot sufficiency
and isolate extraction errors. It is prohibited from production routing and
from claims about the final teacher's available evidence.

### A-OKVQA Adapter

#### Native Input

```text
question + full image + answer choices when present
```

The image and choices remain available in every production attempt. Dataset
hint and reference answer fields are forbidden teacher inputs.

#### Evidence Configurations

`visual_base`:

```text
question + full image + choices
```

`visual_facts`:

```text
native input + image-derived scene/object facts
```

`visual_recovery`:

```text
native input + targeted object/attribute crop + choice-conditioned grounding
```

`knowledge_recovery`:

```text
grounded visual evidence + bounded world-knowledge evidence
```

#### Allowed Actions

```text
ATTACH_VISUAL_FACTS
TARGET_OBJECT_CROP
ATTRIBUTE_GROUNDING
CHOICE_CONTRAST
WORLD_KNOWLEDGE_RECOVERY
VISUAL_FACT_CONFLICT_CHECK
```

#### Quality Features

- object and attribute coverage for question nouns and predicates;
- whether the question is visually direct or knowledge-dependent;
- answer-choice compatibility with visible entities;
- visual-fact/image conflict state;
- unresolved object reference;
- crop availability and grounding quality.

#### Validators

- cited object or attribute existence;
- visual-fact provenance;
- crop-to-object consistency;
- evidence-backed choice elimination;
- explicit separation of direct visual support from world-knowledge inference;
- visual-fact/image relation: `AGREE`, `CONFLICT`, or `UNKNOWN`.

### GSM8K Adapter

#### Native Input

```text
problem text
```

No image path, image token, DePlot provider, or visual-fact action is valid.
Reference rationale and answer fields are forbidden teacher inputs.

#### Evidence Configurations

`reasoning_base`:

```text
problem text + direct structured reasoning
```

`symbolic_recovery`:

```text
problem text + extracted variables/equations or executable program
```

`verification_recovery`:

```text
problem text + deterministic execution + alternative decomposition when needed
```

#### Allowed Actions

```text
EXTRACT_EQUATIONS
BUILD_EXECUTABLE_PROGRAM
EXECUTE_ARITHMETIC
CHECK_UNITS
ALTERNATIVE_DECOMPOSITION
REASONING_CONFLICT_CHECK
```

#### Quality Features

- numeric token coverage;
- unresolved variables;
- operation count and program executability;
- disagreement between natural-language reasoning and execution;
- unit consistency;
- alternative-decomposition agreement.

#### Validators

- every used quantity is grounded in the problem or a derived value;
- equations preserve the stated relationships;
- arithmetic/program execution succeeds;
- executed value equals the candidate answer;
- units and requested quantity match;
- independent decompositions agree or expose a conflict.

## Routing Policy

The initial policy is adapter-specific but follows a shared progression:

1. Generate from native input.
2. Validate the candidate.
3. Accept when all required support is present and no conflict remains.
4. Otherwise choose the cheapest untried action capable of resolving the
   current failure reason.
5. Re-generate or re-verify after the action.
6. Abstain when no allowed action can resolve the failure or the budget is
   exhausted.

The controller does not repeatedly sample the same unchanged evidence
configuration by default. A retry is valid only when the evidence, decoding
contract, or deterministic verification state changes.

Default maximum teacher generation attempts are three per example. Tool-only
validation actions may occur between generations and are logged separately.

## Acceptance Policy

A candidate is eligible for production acceptance only when:

```text
schema is valid
AND required provenance is supported
AND required deterministic checks pass
AND unresolved_refs is empty
AND conflict_state is not CONFLICT
AND calibrated risk is below the adapter threshold
```

If a task-specific check is impossible, the result is `UNKNOWN`, not `PASS`.
The controller may acquire more evidence or abstain. It must not convert model
confidence into a validator pass.

Calibration thresholds are selected on a separate calibration partition and
then frozen for evaluation.

## Gold and Leakage Boundary

Gold answer, reference hint, reference rationale, and correctness-verifier
output are prohibited from:

- teacher prompts;
- evidence actions;
- candidate validators used at runtime;
- controller features;
- acceptance decisions.

They are allowed only for:

- offline metric computation;
- oracle route and recoverability upper bounds;
- creation of router-training labels in a later, explicitly supervised stage;
- error taxonomy and ablation analysis.

Any reference-assisted selector must be named `oracle` in artifacts and paper
tables. It cannot be presented as the deployable harness.

## OPD Integration

The student continues to receive only native deployment input. The teacher
harness may acquire permitted auxiliary evidence and produce a recovered
teacher distribution. OPD consumes teacher logits/distributions, not a hard
target trajectory.

The integration preserves these invariants:

- full teacher-trajectory hard supervision remains disabled;
- dataset hints and reference answers are never copied into student targets;
- no fixed `Goal/Observation/Reasoning/Conclusion` template is required from
  the student;
- failed attempts do not silently replace the raw teacher denominator;
- teacher generation format and student evaluation format remain distinct;
- current continuous GRPO route-rate control remains independent of dataset
  evidence-action selection in the first implementation.

The harness selects or repairs the teacher signal. The existing CLRC controller
still decides the amount of OPD versus GRPO training signal.

## Observability

Every attempt records:

- dataset, example, prompt, schema, model, action, and validator versions;
- native-input digest and evidence-asset digests;
- evidence configuration and action history;
- raw output and parsed candidate;
- validation results and reason codes;
- routing, retry, acceptance, abstention, and budget-exhaustion reasons;
- generation tokens, latency, and estimated cost;
- whether any reference-only field was accessible to the component.

Shared metrics:

```text
harness/base_accept_rate
harness/recovery_trigger_rate
harness/recovered_accept_rate
harness/abstain_rate
harness/budget_exhausted_rate
harness/mean_generation_attempts
harness/mean_action_cost
evidence/schema_valid_rate
evidence/provenance_valid_rate
evidence/operation_valid_rate
evidence/unresolved_reference_rate
evidence/conflict_rate
evidence/reference_free_accept_rate
evidence/reference_free_false_accept_rate
```

Dataset-specific action rates are logged under:

```text
evidence_action/{dataset_id}/{action_id}_rate
```

Training contamination metrics remain mandatory:

```text
repair/full_teacher_trajectory_rate
repair/fixed_template_concentration
repair/reasoning_header_rate
repair/empty_answer_line_rate
repair/exact_answer_only_rate
signal/grpo_zero_loss_rate
routing/grpo_route_rate
routing/opd_route_rate
routing/sft_route_rate
```

## Evaluation Plan and Gates

### Stage 1: Adapter Contract Tests

- ChartQA production configurations always include the full image.
- A-OKVQA production configurations always include the full image and choices.
- GSM8K configurations contain no image or visual evidence.
- Gold/reference fields fail closed when accidentally passed to a production
  component.
- Each action can be enabled, disabled, and budgeted independently.
- Candidate and validator result schemas round-trip without free-form parsing.

### Stage 2: Fixed Teacher-Only Probes

Run fixed-seed probes before any training integration:

- ChartQA: 128 examples, stratified by lookup, arithmetic, counting,
  comparison, color/legend, and spatial reference.
- A-OKVQA: 128 examples, stratified by direct visual, object/attribute,
  choice contrast, and knowledge-dependent questions.
- GSM8K: 128 examples, stratified by operation structure and reasoning depth.

Report:

- base accuracy;
- accuracy after each action;
- per-action marginal recovery;
- oracle union across valid configurations;
- reference-free accepted accuracy and coverage;
- false-accept rate;
- mean generation attempts and action cost;
- controller regret relative to the offline oracle route;
- failure taxonomy over the full raw denominator.

### Stage 3: Entry Gates

An adapter may enter training only if:

1. its recovery actions improve raw teacher accuracy over the base attempt;
2. reference-free accepted precision is at least `80%` on its calibration
   probe;
3. reference-free false-accept rate is below `10%`;
4. mean teacher generations do not exceed `2.2`;
5. no gold/reference field appears in prompt or runtime route traces;
6. template-contamination checks remain zero before training integration.

For ChartQA, the production evidence-configuration oracle union should reach at
least `70%` before a full four-epoch run. Failure to reach this threshold sends
the work back to evidence extraction or teacher capability analysis rather than
same-configuration resampling.

The A-OKVQA and GSM8K gates use the same acceptance and cost criteria, while
their raw-accuracy target is set from base-teacher probe results and recorded
before tuning. This avoids retroactively selecting a convenient threshold.

### Stage 4: Training Smoke

For each eligible adapter:

- run CPU/unit tests;
- run a single-GPU teacher harness smoke;
- run a four-step distributed smoke that exercises base acceptance, recovery,
  conflict, abstention, and budget exhaustion;
- confirm rank-safe batching when different examples take different actions;
- confirm hard SFT and full-trajectory target rates remain zero.

Only after these gates pass may the adapter enter a four-epoch experiment.

## Ablations

Required paper-facing ablations are:

1. OPD without evidence recovery;
2. OPD with deterministic dataset-adaptive recovery;
3. GRPO/DyME baseline at matched generation budget;
4. oracle route upper bound, labeled as non-deployable;
5. action removal for each adapter;
6. fixed attempt count versus budgeted early stopping;
7. reference-free verifier versus answer-only confidence;
8. full-trajectory hard supervision contamination analysis, without enabling
   it in the main method.

ChartQA may additionally report `deplot_only` as a diagnostic ablation, clearly
separated from production configurations.

## Paper Positioning

The primary contribution remains the introduction of online policy
distillation to the target small-model multimodal/reasoning setting and the
demonstration that OPD complements online RL.

The harness is positioned as the mechanism that makes gold-hidden teacher
recoverability measurable and useful across heterogeneous tasks:

> We preserve each task's native input and selectively acquire
> task-appropriate evidence or verification tools when the teacher's current
> reasoning is unsupported, unresolved, or conflicting. The recovered teacher
> distribution is distilled through OPD without hard trajectory imitation.

The paper must not claim that table/image/fused routing itself is novel. It may
claim a task-adaptive recoverability mechanism only after the three adapters,
reference-free verification, matched-budget evaluation, and route-regret
analysis are implemented and validated.

## Implementation Sequence

1. Introduce shared typed state, candidate, action, and validation contracts.
2. Implement ChartQA adapter while preserving full image in every production
   configuration.
3. Run the ChartQA fixed teacher-only gate and revise extraction actions if the
   oracle union remains below threshold.
4. Implement A-OKVQA adapter using existing image-derived visual facts plus
   targeted grounding actions.
5. Implement GSM8K adapter using symbolic extraction and deterministic
   execution.
6. Add deterministic controller, budget, tracing, and calibration.
7. Integrate eligible adapters with the teacher generation path and OPD.
8. Run distributed smoke tests before any full training experiment.

This sequence deliberately validates evidence recovery before allowing it to
change the training distribution.
