#!/usr/bin/env bash
# Fast baseline: cross-model OPD (7B teacher + 0.5B student) with DeepSpeed.
# DyME-aligned routing (teacher-probe gated OPD, no embedded cold-start, no Visual Supervision).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-scripts/test/config/config_opd_7b_chartqa_deepspeed.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"
export DYME_OPSD_HANG_DEBUG="${DYME_OPSD_HANG_DEBUG:-0}"
export DYME_OPSD_HANG_FORCE="${DYME_OPSD_HANG_FORCE:-0}"

prepare_fast_test_data "${DYME_CONFIG}"
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "opd" "${DYME_CONFIG}"

LOG_FILE="$(fast_train_log_path train_test_opd)"
run_train_with_log "${LOG_FILE}" \
  "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${DYME_CONFIG}" \
    --mode rl \
    --opsd_enabled \
    "$@"
