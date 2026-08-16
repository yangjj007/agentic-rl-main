#!/usr/bin/env bash
# 200-step smoke for OPD 7B + RLSD anti-collapse fixes.
# Training params: config/config_opd_7b_smoke.yaml
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

CONFIG_PATH="config/config_opd_7b_smoke.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_zero2.yaml}"

prepare_chartqa_training_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_FILE="$(train_log_path train_opd_7b_smoke)"
echo "OPD 7B smoke config: ${CONFIG_PATH}, log=${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${CONFIG_PATH}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
