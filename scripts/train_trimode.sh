#!/usr/bin/env bash
# TriMode OPSD + DyME on ChartQA (LLaVA-OV 0.5B)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-trimode}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-text,visual_facts}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/trimode-chartqa}"
export DYME_OPSD_DEBUG="${DYME_OPSD_DEBUG:-1}"

ACCELERATE_CONFIG="$(resolve_accelerate_config)"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_trimode_$(date +%Y%m%d_%H%M%S).log"

echo "OPSD debug enabled: ${DYME_OPSD_DEBUG}"
echo "Writing full training log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_debug \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
