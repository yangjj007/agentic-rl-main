#!/usr/bin/env bash
# No-visual PCD run: DePlot textual evidence on, Visual Supervision off,
# all-wrong teacher-probe rescue from step 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EPOCHS="${DYME_PCD_EPOCHS:-4}"
RESUME_MODE="${DYME_PCD_RESUME:-none}"  # none | auto | /path/to/checkpoint-N
DRY_RUN="${DYME_PCD_DRY_RUN:-0}"
SPEED_PROFILE="${DYME_PCD_SPEED_PROFILE:-canonical}"  # canonical | fast60

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_pcd_no_visual.sh [EPOCHS] [--resume auto|CHECKPOINT|none] [--speed-profile canonical|fast60] [--dry-run]

Examples:
  bash scripts/test/run_pcd_no_visual.sh 4 --speed-profile canonical
  bash scripts/test/run_pcd_no_visual.sh 10 --resume auto --speed-profile canonical
  bash scripts/test/run_pcd_no_visual.sh 4 --speed-profile fast60
USAGE
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  EPOCHS="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --speed-profile)
      SPEED_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${EPOCHS}" in
  ''|*[!0-9]*)
    echo "EPOCHS must be a positive integer, got: ${EPOCHS}" >&2
    exit 2
    ;;
esac
if [[ "${EPOCHS}" -lt 1 ]]; then
  echo "EPOCHS must be >= 1, got: ${EPOCHS}" >&2
  exit 2
fi
case "${SPEED_PROFILE}" in
  canonical|fast60)
    ;;
  *)
    echo "Unknown speed profile: ${SPEED_PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

RUN_ID="${DYME_PCD_RUN_ID:-pcd_no_visual_staged}"
VARIANT="${DYME_PCD_VARIANT:-deplot_no_vs_opd_pcd}"
OUT_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual/${RUN_ID}}"
LOG_ROOT="${DYME_PCD_LOG_ROOT:-outputs/test-fast/logs/pcd_no_visual_${RUN_ID}}"
OUT_DIR="${OUT_ROOT}/${VARIANT}"
LOG_DIR="${LOG_ROOT}/${VARIANT}"
STUDENT_MODEL="${DYME_STUDENT_MODEL:-/home/deepseek_VG/deepseek/models/llava-0.5b-ov}"
TEACHER_MODEL="${DYME_TEACHER_MODEL:-/home/deepseek_VG/deepseek/models/llava-7b-ov}"
TEACHER_PROBE_BATCH_SIZE="${DYME_TEACHER_PROBE_BATCH_SIZE:-8}"
if [[ "${SPEED_PROFILE}" == "fast60" ]]; then
  TEACHER_PROBE_MAX_PER_BATCH="${DYME_TEACHER_PROBE_MAX_PER_BATCH:-16}"
  TEACHER_TRAJECTORY="${DYME_TEACHER_TRAJECTORY:-0}"
  TEACHER_PROBE_CANDIDATE_LOG="${DYME_TEACHER_PROBE_CANDIDATE_LOG:-0}"
  ONLINE_SFT_TARGET="${DYME_ONLINE_SFT_TARGET:-answer_only}"
else
  TEACHER_PROBE_MAX_PER_BATCH="${DYME_TEACHER_PROBE_MAX_PER_BATCH:-0}"
  TEACHER_TRAJECTORY="${DYME_TEACHER_TRAJECTORY:-1}"
  TEACHER_PROBE_CANDIDATE_LOG="${DYME_TEACHER_PROBE_CANDIDATE_LOG:-1}"
  ONLINE_SFT_TARGET="${DYME_ONLINE_SFT_TARGET:-hint_answer}"
fi

latest_checkpoint() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  find "${dir}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1
}

RESUME_CHECKPOINT=""
if [[ "${RESUME_MODE}" == "auto" ]]; then
  RESUME_CHECKPOINT="$(latest_checkpoint "${OUT_DIR}")"
  if [[ -z "${RESUME_CHECKPOINT}" ]]; then
    echo "No checkpoint-* found under ${OUT_DIR}; run 4epoch first or pass --resume CHECKPOINT." >&2
    exit 2
  fi
elif [[ "${RESUME_MODE}" == "none" || -z "${RESUME_MODE}" ]]; then
  if [[ -n "$(latest_checkpoint "${OUT_DIR}")" && "${DYME_PCD_ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Existing checkpoint found under ${OUT_DIR}." >&2
    echo "Set DYME_PCD_ALLOW_EXISTING=1 to continue a non-resume run, or use --resume auto." >&2
    exit 2
  fi
else
  RESUME_CHECKPOINT="${RESUME_MODE}"
  if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "No-visual PCD OPD run"
echo "epochs target: ${EPOCHS}"
echo "run id: ${RUN_ID}"
echo "variant: ${VARIANT}"
echo "speed profile: ${SPEED_PROFILE}"
echo "output dir: ${OUT_DIR}"
echo "log dir: ${LOG_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "resume from: ${RESUME_CHECKPOINT}"
else
  echo "resume from: <none>"
fi
echo "save policy: save_strategy=epoch, save_total_limit unset"
echo "candidate logs: ${OUT_DIR}/teacher_probe_candidates/rank*.jsonl (enabled=${TEACHER_PROBE_CANDIDATE_LOG})"
echo "============================================================"

TRAIN_ENV=(
  "-u" "DYME_MAX_STEPS"
  "-u" "DYME_TRAIN_MAX_STEPS"
  "-u" "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP"
  "-u" "DYME_SAVE_TOTAL_LIMIT"
  "DYME_NUM_TRAIN_EPOCHS=${EPOCHS}"
  "DYME_FAST_NUM_TRAIN_EPOCHS=${EPOCHS}"
  "DYME_SAVE_STRATEGY=epoch"
  "DYME_STUDENT_MODEL=${STUDENT_MODEL}"
  "DYME_TEACHER_MODEL=${TEACHER_MODEL}"
  "DYME_OUTPUT_DIR=${OUT_DIR}"
  "DYME_LOG_DIR=${LOG_DIR}"
  "DYME_OPSD_PRIVILEGE_PROFILE=text"
  "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot"
  "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot"
  "DYME_TEACHER_PROBE=1"
  "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0"
  "DYME_TEACHER_PROBE_BATCH_SIZE=${TEACHER_PROBE_BATCH_SIZE}"
  "DYME_TEACHER_PROBE_MAX_PER_BATCH=${TEACHER_PROBE_MAX_PER_BATCH}"
  "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96"
  "DYME_TEACHER_PROBE_CANDIDATE_LOG=${TEACHER_PROBE_CANDIDATE_LOG}"
  "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=512"
  "DYME_TEACHER_TRAJECTORY=${TEACHER_TRAJECTORY}"
  "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128"
  "DYME_ONLINE_SFT_TARGET=${ONLINE_SFT_TARGET}"
  "DYME_OPSD_LOSS_TYPE=jsd"
  "DYME_OPSD_WEIGHT=1.5"
  "DYME_OPSD_VARIANCE_ADAPTIVE=0"
  "DYME_OPSD_ADAPTIVE_STD_TARGET=0.25"
  "DYME_OPSD_ADAPTIVE_MAX_MULT=2.0"
  "DYME_GRPO_WEIGHT=1.0"
  "DYME_OPSD_SRKL_ALPHA=0.1"
  "DYME_VISUAL_CHECKER=0"
  "DYME_VISUAL_REFINER=0"
  "DYME_VISUAL_LOG=0"
  "DYME_VISUAL_SAVE_ARTIFACTS=0"
  "DYME_VISUAL_LOG_SAMPLES=0"
  "DYME_DEPLOT_ENABLED=0"
  "DYME_OPSD_HANG_DEBUG=0"
  "DYME_OPSD_HANG_FORCE=0"
  "DYME_OPSD_DETAIL_EVERY=0"
  "DYME_PERF_TIMING=${DYME_PERF_TIMING:-1}"
  "TRANSFORMERS_OFFLINE=1"
  "HF_HUB_OFFLINE=1"
  "WANDB_MODE=disabled"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ENV+=("DYME_RESUME_FROM_CHECKPOINT=${RESUME_CHECKPOINT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'env'
  printf ' %q' "${TRAIN_ENV[@]}"
  printf ' bash scripts/train_opd_7b_dyme_probe.sh --no_opsd_probe_on_generate --no_opsd_probe_first_token_logits --opsd_detail_every 0\n'
  exit 0
fi

env "${TRAIN_ENV[@]}" \
  bash scripts/train_opd_7b_dyme_probe.sh \
    --no_opsd_probe_on_generate \
    --no_opsd_probe_first_token_logits \
    --opsd_detail_every 0
