#!/usr/bin/env bash
# TriMode OPSD + DyME on ChartQA.
# Training params: config/config_trimode.py
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-trimode}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_FILE="$(train_log_path train_trimode)"
echo "Config: ${DYME_CONFIG}, log=${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
