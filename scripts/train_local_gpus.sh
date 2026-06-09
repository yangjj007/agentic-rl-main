#!/usr/bin/env bash
# TriMode OPSD training on all GPUs visible to this machine.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

NUM_PROCESSES="$(detect_num_gpus)"
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No GPUs detected. Set NUM_GPUS=<n> or check CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

if [[ "${NUM_PROCESSES}" -ge 8 ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_8gpu.yaml}"
else
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config.yaml}"
fi

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-trimode}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-text,visual_facts}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/trimode-chartqa}"
export DYME_OPSD_DEBUG="${DYME_OPSD_DEBUG:-1}"

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_trimode_${NUM_PROCESSES}gpu_$(date +%Y%m%d_%H%M%S).log"

print_launch_plan
echo "accelerate config: ${ACCELERATE_CONFIG}"
echo "OPSD debug enabled: ${DYME_OPSD_DEBUG}"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_debug \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
