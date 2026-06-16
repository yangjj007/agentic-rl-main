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

ACCELERATE_CONFIG="$(resolve_accelerate_config)"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-trimode}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-text,visual_facts}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/trimode-chartqa}"
export DYME_OPSD_DEBUG="${DYME_OPSD_DEBUG:-0}"
export DYME_OPSD_DETAIL_EVERY="${DYME_OPSD_DETAIL_EVERY:-50}"
DYME_CONFIG="${DYME_CONFIG:-config/config_trimode_antidegen.py}"

export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-1}"
ensure_chartqa_vf_full
ensure_spacy_model

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_trimode_${NUM_PROCESSES}gpu_$(date +%Y%m%d_%H%M%S).log"

print_launch_plan
echo "accelerate config: ${ACCELERATE_CONFIG}"
echo "ChartQA dataset: ${DYME_CHARTQA_VF_FULL:-data/chartqa/train_medium_vf_full.json}"
echo "OPSD debug enabled: ${DYME_OPSD_DEBUG} (detail_every=${DYME_OPSD_DETAIL_EVERY})"
echo "Config: ${DYME_CONFIG}"
echo "Writing log to: ${LOG_FILE}"

OPSD_EXTRA_ARGS=()
case "${DYME_OPSD_DEBUG}" in
  1|true|yes|on|TRUE|YES|ON)
    OPSD_EXTRA_ARGS+=(--opsd_debug)
    ;;
esac

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  "${OPSD_EXTRA_ARGS[@]}" \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
