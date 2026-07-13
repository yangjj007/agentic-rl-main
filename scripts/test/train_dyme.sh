#!/usr/bin/env bash
# Fast baseline: DyME dynamic SFT/GRPO routing, with no OPD, on full ChartQA.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-scripts/test/config/config.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

prepare_fast_test_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "dyme" "${DYME_CONFIG}"

LOG_FILE="$(fast_train_log_path train_test_dyme)"
run_train_with_log "${LOG_FILE}" \
  "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${DYME_CONFIG}" \
    --mode rl \
    "$@"
