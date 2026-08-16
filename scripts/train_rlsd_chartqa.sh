#!/usr/bin/env bash
# RLSD anti-leakage OPSD on ChartQA.
# Training params: config/config_rlsd_chartqa.yaml
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

CONFIG_PATH="config/config_rlsd_chartqa.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_chartqa_training_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_FILE="$(train_log_path train_rlsd)"
echo "Config: ${CONFIG_PATH}, log=${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${CONFIG_PATH}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
