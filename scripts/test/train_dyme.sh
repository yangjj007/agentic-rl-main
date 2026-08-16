#!/usr/bin/env bash
# Fast baseline: pure DyME (GRPO, no OPSD) on full ChartQA (fewer epochs).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

CONFIG_PATH="config/config.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_fast_test_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "dyme" "${CONFIG_PATH}"

LOG_FILE="$(fast_train_log_path train_test_dyme)"
run_train_with_log "${LOG_FILE}" \
  accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${CONFIG_PATH}" \
    --mode rl \
    "$@"
