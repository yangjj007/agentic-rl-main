#!/usr/bin/env bash
# Post-training ChartQA eval for E0–E3 ablations (set CHECKPOINT_DIR)
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to trained model path}"
EXPERIMENT="${EXPERIMENT:-E0}"

echo "Evaluating ${EXPERIMENT} from ${CHECKPOINT_DIR}"
python eval/eval_chartqa.py --model_path "${CHECKPOINT_DIR}"

# Suggested matrix:
# E0: trimode baseline  CHECKPOINT_DIR=./outputs/trimode-chartqa/final_checkpoint EXPERIMENT=E0
# E1: RLSD anti-leak    CHECKPOINT_DIR=./outputs/rlsd-chartqa/final_checkpoint EXPERIMENT=E1
# E2: 7B cross-OPD      CHECKPOINT_DIR=./outputs/opd-7b-chartqa/final_checkpoint EXPERIMENT=E2
# E3: opsd_only ablation (legacy providers)
