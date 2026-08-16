#!/usr/bin/env bash
# Phase-1 offline SFT for ChartQA (hint + Answer GT), then run RLSD/OPD from the SFT checkpoint.
# Training params: config/config_rlsd_chartqa.yaml (training.sft_args)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

CONFIG_PATH="config/config_rlsd_chartqa.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_chartqa_training_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

SFT_OUT="$("${PYTHON_BIN}" -c "from config.loader import load_config; print(load_config('${CONFIG_PATH}')['training']['sft_args']['output_dir'])")"
echo "Offline ChartQA SFT -> ${SFT_OUT}"
echo "For an OPD-only continuation, copy config/config_opd_only_7b_chartqa.yaml"
echo "and set model.pretrained_model_path: ${SFT_OUT}/final_checkpoint."

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main_sft.py \
  --config "${CONFIG_PATH}" \
  "$@"
