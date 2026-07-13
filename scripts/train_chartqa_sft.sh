#!/usr/bin/env bash
# Phase-1 offline SFT for ChartQA (hint + Answer GT), then run RLSD/OPD from the SFT checkpoint.
# Training params: config/config_rlsd_chartqa.py (training.sft_args)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-rlsd_chartqa}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

SFT_OUT="$(python -c "from config.loader import load_config; print(load_config('${DYME_CONFIG}')['training']['sft_args']['output_dir'])")"
echo "Offline ChartQA SFT -> ${SFT_OUT}"
echo "After SFT, launch RLSD/OPD with:"
echo "  export DYME_STUDENT_MODEL=${SFT_OUT}/final_checkpoint"
echo "  bash scripts/train_opd_7b_chartqa_deepspeed.sh"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main_sft.py \
  --config "${DYME_CONFIG}" \
  "$@"
