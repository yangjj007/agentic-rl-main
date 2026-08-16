#!/usr/bin/env bash
# Fast baseline: offline SFT on full ChartQA (fewer epochs).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

CONFIG_PATH="config/config_rlsd_chartqa.yaml"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_fast_test_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "sft" "${CONFIG_PATH}"

LOG_FILE="$(fast_train_log_path train_test_sft)"
run_train_with_log "${LOG_FILE}" \
  accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main_sft.py \
    --config "${CONFIG_PATH}" \
    "$@"
