#!/usr/bin/env bash
# Cross-model OPD (7B teacher + 0.5B student) with DeepSpeed on ChartQA.
# Training params: config/config_opd_7b_chartqa_deepspeed.py
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_chartqa_deepspeed.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-opd_7b_deepspeed}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero0_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "Config: ${DYME_CONFIG} (DeepSpeed ZeRO-0 default; override ACCELERATE_CONFIG if OOM)"

LOG_FILE="$(train_log_path train_opd_7b_ds)"
accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
