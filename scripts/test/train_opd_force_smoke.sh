#!/usr/bin/env bash
# Smoke the explicit, isolated OPD-only training stage.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

CONFIG_PATH="config/config_opd_only_7b_chartqa_smoke.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"

prepare_fast_test_data "${CONFIG_PATH}"
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"

echo "============================================================"
echo "OPD-only smoke (no GRPO, SFT, routing, or visual reward components)"
echo "config: ${CONFIG_PATH}"
print_launch_plan

LOG_FILE="$(fast_train_log_path train_opd_force_smoke)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${CONFIG_PATH}" \
  --mode rl \
  --no_wandb \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
