#!/usr/bin/env bash
# COPSD-style cross-model OPD: frozen 7B teacher + 0.5B student on ChartQA (DDP).
# Training params: config/config_opd_7b_chartqa.py
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-opd_7b_chartqa}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_FILE="$(train_log_path train_opd_7b)"
echo "Config: ${DYME_CONFIG}"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
