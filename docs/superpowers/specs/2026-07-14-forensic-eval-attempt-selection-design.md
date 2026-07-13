# Forensic Eval Attempt Selection Design

## Problem

`scripts/analysis/pcd_low_score_forensics.py` normalizes labels such as
`eval_final_checkpoint_bsz1_gpu0_20260709_192652` to `final_checkpoint`.
It currently stores rows in a dictionary keyed by that normalized checkpoint,
so a later OOM or traceback attempt can overwrite an earlier valid 2,500-sample
evaluation. This makes the forensic report omit the authoritative result even
though `eval_chartqa/summary.csv` still contains it.

## Scope

Fix only the forensic aggregation path. The primary eval log parser continues
to retain every attempt and is not changed to choose or delete results.

## Selection Model

The forensic collector may read multiple attempts for one normalized
checkpoint. It selects exactly one row with this deterministic priority, in
descending order:

1. An accuracy value is present and finite.
2. The processed count is complete enough for ChartQA: at least
   `min(total, 2496)` when `total` is known, otherwise at least 2,496.
3. The attempt has no reported error and has exit status `0` or no recorded
   exit status.
4. More samples were processed.
5. A later timestamp-bearing source label wins the remaining tie.

Accuracy magnitude is not a selection criterion. This prevents retry
selection from becoming checkpoint-level accuracy cherry-picking.

Rows discovered directly from logs use the same candidate pool and selection
rules as rows discovered from summary CSV files.

## Auditability

`checkpoint_accuracy.csv` remains one row per normalized checkpoint and adds:

- `source_label`: the original summary label or log stem for the selected
  attempt.
- `exit_status`: the selected attempt's recorded process status.
- `errors`: parser-detected failures for the selected attempt.

The normalized `checkpoint` column remains stable for existing plots and
downstream joins.

## Tests

Add a regression fixture where a valid `0.5800`, `2500/2500` final evaluation
is followed by OOM and traceback attempts with the same normalized checkpoint.
The valid attempt must remain selected and expose its source label.

Add a priority test showing that a 2,496-sample valid attempt beats a later
partial attempt, while a later equally complete and clean attempt is selected
without comparing accuracy magnitude.

## Non-Goals

- Changing ChartQA evaluation or decoding.
- Treating a failed attempt as a model result.
- Selecting the best accuracy among retries.
- Changing checkpoint selection across different checkpoint numbers.
