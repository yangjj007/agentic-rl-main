#!/usr/bin/env bash
# Fast baseline: pure DyME (GRPO, no OPSD) on full ChartQA (fewer epochs).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-scripts/test/config/config.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_fast_test_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "dyme" "${DYME_CONFIG}"

LOG_FILE="$(fast_train_log_path train_test_dyme)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
