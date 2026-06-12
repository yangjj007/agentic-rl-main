#!/usr/bin/env bash
# RLSD anti-leakage OPSD on ChartQA (no privileged visual / no gold in teacher suffix)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-format_only}"
export DYME_OPSD_PRIVILEGE_PROFILE="${DYME_OPSD_PRIVILEGE_PROFILE:-text}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/rlsd-chartqa}"
export DYME_OPSD_REQUIRE_FORMAT="${DYME_OPSD_REQUIRE_FORMAT:-0}"
export DYME_OPSD_DEBUG="${DYME_OPSD_DEBUG:-0}"

ACCELERATE_CONFIG="$(resolve_accelerate_config)"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_rlsd_$(date +%Y%m%d_%H%M%S).log"

echo "RLSD mode: ${DYME_OPSD_MODE}, providers: ${DYME_OPSD_PROVIDERS}"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_rlsd_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  --opsd_privilege_profile "${DYME_OPSD_PRIVILEGE_PROFILE}" \
  2>&1 | tee "${LOG_FILE}"
