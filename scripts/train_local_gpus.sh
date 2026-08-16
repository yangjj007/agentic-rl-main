#!/usr/bin/env bash
# TriMode OPSD training on all GPUs visible to this machine.
# Training params: config/config_trimode_antidegen.yaml (default)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

CONFIG_PATH="config/config_trimode_antidegen.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

NUM_PROCESSES="$(detect_num_gpus)"
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No GPUs detected. Set NUM_GPUS=<n> or check CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

prepare_chartqa_training_data "${CONFIG_PATH}"

LOG_FILE="$(train_log_path "train_trimode_${NUM_PROCESSES}gpu")"
print_launch_plan
echo "Config: ${CONFIG_PATH}, log=${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${CONFIG_PATH}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
