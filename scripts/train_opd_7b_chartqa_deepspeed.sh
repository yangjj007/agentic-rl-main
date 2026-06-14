#!/usr/bin/env bash
# Cross-model OPD (7B teacher + 0.5B student) with DeepSpeed ZeRO on ChartQA.
#
# Layout (2× GPU, recommended):
#   Each rank: student (ZeRO-sharded) + frozen 7B teacher on cuda:{LOCAL_RANK}
#
# Official refs:
#   https://huggingface.co/docs/transformers/deepspeed
#   https://huggingface.co/docs/accelerate/usage_guides/deepspeed
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-}"
export DYME_OPSD_PRIVILEGE_PROFILE="${DYME_OPSD_PRIVILEGE_PROFILE:-text}"
export DYME_TEACHER_MODEL="${DYME_TEACHER_MODEL:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-chartqa-ds}"

# ZeRO-2 (default) or ZeRO-3 colocate for tighter memory:
#   ACCELERATE_CONFIG=default_config_zero3_colocate.yaml
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_zero2.yaml}"
# auto → colocate when DeepSpeed accelerate config is set (see opsd_utils/deepspeed_utils.py)
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
export DYME_OPSD_DETAIL_MIN_FREE_GB="${DYME_OPSD_DETAIL_MIN_FREE_GB:-4.0}"
export DYME_OPSD_DETAIL_EVERY="${DYME_OPSD_DETAIL_EVERY:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DYME_GRADIENT_CHECKPOINTING="${DYME_GRADIENT_CHECKPOINTING:-1}"
# DeepSpeed ZeRO-1/2 requires non-reentrant checkpointing (set automatically in main.py).

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "DeepSpeed ZeRO OPD: ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "Teacher placement: DYME_TEACHER_DEVICE_MAP=${DYME_TEACHER_DEVICE_MAP} (auto colocates under DeepSpeed)"

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_opd_7b_ds_$(date +%Y%m%d_%H%M%S).log"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_opd_7b_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
