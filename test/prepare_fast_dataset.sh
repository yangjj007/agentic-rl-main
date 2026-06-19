#!/usr/bin/env bash
# Build data/chartqa/train_fast_<N>.json for test/ fast baselines.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/launch_utils.sh"

export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
ensure_chartqa_vf_full

VF_FULL="${DYME_CHARTQA_VF_FULL:-data/chartqa/train_medium_vf_full.json}"
MAX_SAMPLES="${DYME_FAST_MAX_SAMPLES:-512}"
FAST_JSON="${DYME_FAST_TRAIN_JSON:-data/chartqa/train_fast_${MAX_SAMPLES}.json}"

if [[ -f "${FAST_JSON}" ]]; then
  echo "Fast dataset already exists: ${FAST_JSON}"
  exit 0
fi

python test/build_fast_dataset.py \
  --input "${VF_FULL}" \
  --output "${FAST_JSON}" \
  --max-samples "${MAX_SAMPLES}"
