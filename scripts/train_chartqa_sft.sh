#!/usr/bin/env bash
# Phase-1 offline SFT for ChartQA (hint + Answer GT), then run RLSD/OPD from the SFT checkpoint.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_SFT_OUTPUT_DIR="${DYME_SFT_OUTPUT_DIR:-./outputs/chartqa-sft}"
export DYME_SFT_EPOCHS="${DYME_SFT_EPOCHS:-2}"

ACCELERATE_CONFIG="$(resolve_accelerate_config)"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

echo "Offline ChartQA SFT -> ${DYME_SFT_OUTPUT_DIR}"
echo "After SFT, launch RLSD/OPD with:"
echo "  export DYME_PRETRAINED_MODEL=${DYME_SFT_OUTPUT_DIR}/final_checkpoint"
echo "  bash scripts/train_opd_7b_chartqa_deepspeed.sh"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main_sft.py \
  --config config/config_rlsd_chartqa.py \
  "$@"
