#!/usr/bin/env bash
# COPSD-style cross-model OPD: frozen 7B teacher + 0.5B student on ChartQA
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-}"
export DYME_OPSD_PRIVILEGE_PROFILE="${DYME_OPSD_PRIVILEGE_PROFILE:-text}"
export DYME_TEACHER_MODEL="${DYME_TEACHER_MODEL:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-chartqa}"
# Optional ZeRO-2 student sharding (HF Accelerate + DeepSpeed JSON). Pair with DYME_TEACHER_DEVICE_MAP=same.
# ZeRO-2 (default) or ZeRO-3 colocate for tighter memory:
#   ACCELERATE_CONFIG=default_config_zero3_colocate.yaml
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config.yaml}"
# DDP default: complement GPUs. DeepSpeed script uses default_config_zero2.yaml instead.
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
# OPSD-DETAIL: skip JSD decomposition when free GPU memory (GiB) is below this threshold
export DYME_OPSD_DETAIL_MIN_FREE_GB="${DYME_OPSD_DETAIL_MIN_FREE_GB:-4.0}"

ACCELERATE_CONFIG="$(resolve_accelerate_config)"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_opd_7b_$(date +%Y%m%d_%H%M%S).log"

echo "OPD teacher: ${DYME_TEACHER_MODEL}"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_opd_7b_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
