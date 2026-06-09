#!/usr/bin/env bash
# Post-training ChartQA eval for E0–E3 ablations (set CHECKPOINT_DIR)
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to trained model path}"
EXPERIMENT="${EXPERIMENT:-E0}"

echo "Evaluating ${EXPERIMENT} from ${CHECKPOINT_DIR}"
python eval/eval_chartqa.py --model_path "${CHECKPOINT_DIR}"

# Suggested matrix:
# E0: MODE=dyme  (original DyME, opsd disabled)
# E1: MODE=trimode
# E2: MODE=replace_sft
# E3: MODE=opsd_only DYME_OPSD_PROVIDERS=text|visual_facts|text,visual_facts
