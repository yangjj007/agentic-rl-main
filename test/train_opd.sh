#!/usr/bin/env bash
# Fast baseline: cross-model OPD (7B teacher + 0.5B student) with DeepSpeed.
# Embedded SFT cold-start steps are included in max_steps (see test/config/fast_profile.py).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-test/config/config_opd_7b_chartqa_deepspeed.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero0_config)}"

prepare_fast_test_data
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_fast_plan "opd" "${DYME_CONFIG}"

LOG_FILE="$(fast_train_log_path train_test_opd)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
