#!/usr/bin/env bash
# DyME teacher-probe OPD on ChartQA — training params in config/config_opd_7b_dyme_probe.py
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe.sh
#   bash scripts/train_opd_7b_dyme_probe_smoke.sh   # 200-step short validation
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_dyme_probe.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-opd_7b_dyme_probe}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero0_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "Config: ${DYME_CONFIG} (all training hyperparameters live in the matching config/*.py file)"
echo "Snapshot: see output_dir/run_config_snapshot.json after launch"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  --no_wandb \
  2>&1 | tee "${LOG_FILE}"
