#!/usr/bin/env bash
# 200-step smoke run for RLSD anti-leakage validation (compare logs vs trimode baseline)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_MAX_STEPS="${DYME_MAX_STEPS:-200}"
export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-format_only}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/rlsd-chartqa-shortrun}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config.yaml}"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_rlsd_shortrun_$(date +%Y%m%d_%H%M%S).log"

echo "RLSD short run: max_steps=${DYME_MAX_STEPS}, log=${LOG_FILE}"
echo "After training, compare:"
echo "  python scripts/compare_trimode_logs.py <trimode_baseline.log> ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_rlsd_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
