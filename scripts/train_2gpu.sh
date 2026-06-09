#!/usr/bin/env bash
# Explicit 2-GPU TriMode training (use when the node only exposes 2 GPUs).
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export NUM_GPUS=2
export DYME_OPSD_MODE="${DYME_OPSD_MODE:-trimode}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-text,visual_facts}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/trimode-chartqa}"
export DYME_OPSD_DEBUG="${DYME_OPSD_DEBUG:-1}"

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_trimode_2gpu_$(date +%Y%m%d_%H%M%S).log"

print_launch_plan
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file default_config.yaml --num_processes 2 main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_debug \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
