#!/usr/bin/env bash
# Two-stage experiment: positive-replay SFT warmup, then DyME/PCD 4epoch training.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
cd "${ROOT}"

DRY_RUN=0
VARIANT="${DYME_CHAIN_PCD_VARIANT:-deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh [--dry-run] [--variant NAME]

Environment:
  DYME_CHAIN_RUN_ID             shared run prefix, default replay_then_dyme_<timestamp>
  DYME_REPLAY_TRAIN_DATASET     replay_train.json path
  DYME_REPLAY_SFT_EPOCHS        SFT warmup epochs, default inherited by warmup runner
  DYME_REPLAY_SFT_OUTPUT_ROOT   warmup output root
  DYME_REPLAY_SFT_LOG_ROOT      warmup log root
  DYME_REPLAY_SFT_RUN_ID        warmup run id, default ${DYME_CHAIN_RUN_ID}_warmup
  DYME_PCD_RUN_ID               DyME run id, default ${DYME_CHAIN_RUN_ID}_dyme
  DYME_CHAIN_SKIP_WARMUP=1      skip stage 1 and require warmup final_checkpoint
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --variant)
      VARIANT="$2"
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

export DYME_CHAIN_RUN_ID="${DYME_CHAIN_RUN_ID:-replay_then_dyme_$(date +%Y%m%d_%H%M%S)}"
export DYME_REPLAY_TRAIN_DATASET="${DYME_REPLAY_TRAIN_DATASET:-outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json}"
export DYME_REPLAY_SFT_OUTPUT_ROOT="${DYME_REPLAY_SFT_OUTPUT_ROOT:-outputs/test-fast/positive-replay-sft}"
export DYME_REPLAY_SFT_LOG_ROOT="${DYME_REPLAY_SFT_LOG_ROOT:-outputs/test-fast/logs/positive-replay-sft}"
export DYME_REPLAY_SFT_RUN_ID="${DYME_REPLAY_SFT_RUN_ID:-${DYME_CHAIN_RUN_ID}_warmup}"
export DYME_SFT_OUTPUT_DIR="${DYME_SFT_OUTPUT_DIR:-${DYME_REPLAY_SFT_OUTPUT_ROOT}/${DYME_REPLAY_SFT_RUN_ID}}"
export DYME_PCD_RUN_ID="${DYME_PCD_RUN_ID:-${DYME_CHAIN_RUN_ID}_dyme}"

WARMUP_CKPT="${DYME_SFT_OUTPUT_DIR}/final_checkpoint"
SKIP_WARMUP="${DYME_CHAIN_SKIP_WARMUP:-0}"

if [[ ! -f "${DYME_REPLAY_TRAIN_DATASET}" ]]; then
  echo "Replay train dataset not found: ${DYME_REPLAY_TRAIN_DATASET}" >&2
  echo "Generate it with scripts/analysis/export_positive_replay_buffer.py first." >&2
  exit 1
fi

echo "============================================================"
echo "Replay warmup -> DyME PCD chain"
echo "run id: ${DYME_CHAIN_RUN_ID}"
echo "variant: ${VARIANT}"
echo "replay dataset: ${DYME_REPLAY_TRAIN_DATASET}"
echo "warmup run id: ${DYME_REPLAY_SFT_RUN_ID}"
echo "warmup checkpoint: ${WARMUP_CKPT}"
echo "dyme run id: ${DYME_PCD_RUN_ID}"
echo "DYME_PCD_RUN_ID=${DYME_PCD_RUN_ID}"
echo "skip warmup: ${SKIP_WARMUP}"
echo "============================================================"

echo "Stage 1: positive replay SFT warmup"
if [[ "${SKIP_WARMUP}" == "1" ]]; then
  echo "  skipped by DYME_CHAIN_SKIP_WARMUP=1"
else
  if [[ "${DRY_RUN}" == "1" ]]; then
    bash scripts/test/run_positive_replay_sft_warmup.sh --dry-run
  else
    bash scripts/test/run_positive_replay_sft_warmup.sh
  fi
fi

if [[ "${DRY_RUN}" != "1" && ! -d "${WARMUP_CKPT}" ]]; then
  echo "Warmup final checkpoint not found after stage 1: ${WARMUP_CKPT}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry-run checkpoint handoff:"
  echo "  DYME_STUDENT_MODEL=${WARMUP_CKPT}"
fi

echo "Stage 2: DyME PCD training"
if [[ "${DRY_RUN}" == "1" ]]; then
  DYME_STUDENT_MODEL="${WARMUP_CKPT}" \
    bash scripts/test/run_pcd_no_visual_4epoch.sh --dry-run --variant "${VARIANT}"
else
  DYME_STUDENT_MODEL="${WARMUP_CKPT}" \
    bash scripts/test/run_pcd_no_visual_4epoch.sh --variant "${VARIANT}"
fi
