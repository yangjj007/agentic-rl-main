#!/usr/bin/env bash
# Short real ChartQA smoke for the OPD image-primary checker training variant.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

_USER_DYME_LOG_DIR="${DYME_LOG_DIR:-}"
source "${ROOT}/scripts/test/launch_utils.sh"

RUN_ID="${SMOKE_RUN_ID:-image-checker-timing-$(date +%Y%m%d_%H%M%S)}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-${ROOT}/outputs/test-fast/${RUN_ID}}"
if [[ -n "${_USER_DYME_LOG_DIR}" ]]; then
  export DYME_LOG_DIR="${_USER_DYME_LOG_DIR}"
else
  export DYME_LOG_DIR="${ROOT}/outputs/test-fast/logs/${RUN_ID}"
fi
export DYME_TRAIN_MAX_STEPS="${DYME_TRAIN_MAX_STEPS:-${SMOKE_STEPS:-2}}"
export DYME_MAX_COMPLETION_LENGTH="${DYME_MAX_COMPLETION_LENGTH:-${SMOKE_MAX_COMPLETION_LENGTH:-64}}"
export DYME_VISUAL_TEACHER_BATCH_SIZE="${DYME_VISUAL_TEACHER_BATCH_SIZE:-${SMOKE_VISUAL_TEACHER_BATCH_SIZE:-4}}"
export DYME_VISUAL_CHECKER_MAX_SCORE_TOKENS="${DYME_VISUAL_CHECKER_MAX_SCORE_TOKENS:-${SMOKE_CHECKER_MAX_SCORE_TOKENS:-16}}"
export DYME_VISUAL_LOG=1
export DYME_VISUAL_SAVE_ARTIFACTS=1
export DYME_VISUAL_LOG_SAMPLES="${DYME_VISUAL_LOG_SAMPLES:-2}"
# This tiny diagnostic inherits a production ChartQA config. Keep its save
# events cheap even if the caller's shell enabled checkpoint selection.
export DYME_CHECKPOINT_EVAL=0
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${DYME_OUTPUT_DIR}" "${DYME_LOG_DIR}"

before_ts="$(date +%s)"
echo "============================================================"
echo "Image-checker timing smoke"
echo "script: scripts/train_opd_7b_dyme_probe_image_checker.sh"
echo "output_dir: ${DYME_OUTPUT_DIR}"
echo "log_dir: ${DYME_LOG_DIR}"
echo "steps: ${DYME_TRAIN_MAX_STEPS}"
echo "max_completion_length: ${DYME_MAX_COMPLETION_LENGTH}"
echo "visual_teacher_batch_size: ${DYME_VISUAL_TEACHER_BATCH_SIZE}"
echo "checker_max_score_tokens: ${DYME_VISUAL_CHECKER_MAX_SCORE_TOKENS}"
echo "============================================================"

bash scripts/train_opd_7b_dyme_probe_image_checker.sh --opsd_detail_every "${SMOKE_OPSD_DETAIL_EVERY:-1}" "$@"

after_ts="$(date +%s)"
latest_log="$(ls -t "${DYME_LOG_DIR}"/train_opd_7b_dyme_probe_image_checker*.log 2>/dev/null | head -1 || true)"
if [[ -z "${latest_log}" ]]; then
  echo "No train_opd_7b_dyme_probe_image_checker log found in ${DYME_LOG_DIR}" >&2
  exit 1
fi

echo "[IMAGE-CHECKER-TIMING] wall_clock_s=$((after_ts - before_ts))"
"${PYTHON_BIN}" scripts/analysis/image_checker_timing_report.py \
  --log-file "${latest_log}" \
  --output-dir "${DYME_OUTPUT_DIR}"
