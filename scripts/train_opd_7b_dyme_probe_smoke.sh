#!/usr/bin/env bash
# 200-step smoke for dyme_teacher_probe_opd (no wandb prompt, no HF eval download).
# Training params: config/config_opd_7b_dyme_probe_smoke.py
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="dyme_probe_smoke"
export DYME_MAX_STEPS=200
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_zero2.yaml}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "Smoke config: ${DYME_CONFIG} (max_steps=${DYME_MAX_STEPS})"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe_smoke)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  --no_wandb \
  2>&1 | tee "${LOG_FILE}"
