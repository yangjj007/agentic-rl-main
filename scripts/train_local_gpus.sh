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

CHARTQA_RAW="${DYME_CHARTQA_RAW:-data/chartqa/train_medium.json}"
CHARTQA_VF_FULL="${DYME_CHARTQA_VF_FULL:-data/chartqa/train_medium_vf_full.json}"
CHARTQA_VF_HINT="${DYME_CHARTQA_VF_HINT:-data/chartqa/train_medium_vf_hint.json}"
DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-1}"
DYME_DEPLOT_BATCH_SIZE="${DYME_DEPLOT_BATCH_SIZE:-8}"
DYME_DEPLOT_MAX_NEW_TOKENS="${DYME_DEPLOT_MAX_NEW_TOKENS:-384}"
DYME_DEPLOT_CACHE="${DYME_DEPLOT_CACHE:-data/chartqa/deplot_cache.json}"

DEPLOT_EXTRA_ARGS=()
case "${DYME_DEPLOT_ENABLED}" in
  0|false|no|off|FALSE|NO|OFF)
    DEPLOT_EXTRA_ARGS+=(--no-enabled)
    ;;
esac

if [[ ! -f "${CHARTQA_VF_FULL}" ]]; then
  echo "Enriched ChartQA dataset not found at ${CHARTQA_VF_FULL}; running visual-facts preprocessing..."
  if [[ ! -f "${CHARTQA_RAW}" ]]; then
    echo "Missing raw dataset: ${CHARTQA_RAW}" >&2
    exit 1
  fi
  python scripts/build_visual_facts_chartqa.py \
    --input "${CHARTQA_RAW}" \
    --output "${CHARTQA_VF_HINT}" \
    --also-set-visual-fact
  python scripts/build_visual_facts_chartqa_deplot.py \
    --input "${CHARTQA_VF_HINT}" \
    --output "${CHARTQA_VF_FULL}" \
    --batch-size "${DYME_DEPLOT_BATCH_SIZE}" \
    --max-new-tokens "${DYME_DEPLOT_MAX_NEW_TOKENS}" \
    --cache "${DYME_DEPLOT_CACHE}" \
    "${DEPLOT_EXTRA_ARGS[@]}"
fi

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_trimode_${NUM_PROCESSES}gpu_$(date +%Y%m%d_%H%M%S).log"

print_launch_plan
echo "accelerate config: ${ACCELERATE_CONFIG}"
echo "ChartQA dataset: ${CHARTQA_VF_FULL}"
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
