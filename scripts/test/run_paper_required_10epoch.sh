#!/usr/bin/env bash
# Paper-required experiments outside the 10-epoch no-visual PCD main run.
#
# Default mode is dry-run. Use --run and optionally --stages to launch only the
# missing pieces needed for the paper main table and reliability checks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -x "/home/deepseek_VG/.conda/envs/dyme/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MODE="dry-run"
RUN_ID="${DYME_PAPER_REQUIRED_RUN_ID:-paper_required_10epoch}"
MAIN_RUN_ID="${DYME_PAPER_MAIN_RUN_ID:-pcd_no_visual_staged}"
EPOCHS="${DYME_PAPER_REQUIRED_EPOCHS:-10}"
STAGES="base_eval,sft_train,dyme_train,no_pcd_anchor,sanity,eval_required"
EVAL_NUM_PROCESSES="${DYME_PAPER_EVAL_NUM_PROCESSES:-8}"

STUDENT_MODEL="${DYME_STUDENT_MODEL:-/home/deepseek_VG/deepseek/models/llava-0.5b-ov}"
BASE_ROOT="${DYME_PAPER_REQUIRED_OUTPUT_ROOT:-outputs/test-fast/paper-required/${RUN_ID}}"
BASELINE_ROOT="${BASE_ROOT}/baselines"
BASELINE_LOG_ROOT="${BASE_ROOT}/logs/baselines"
ANCHOR_RUN_ID="${DYME_PAPER_ANCHOR_RUN_ID:-${RUN_ID}_no_pcd_anchor}"
ANCHOR_ROOT="${BASE_ROOT}/opd-deplot-anchor"
ANCHOR_LOG_ROOT="${BASE_ROOT}/logs/opd-deplot-anchor"
MAIN_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual/${MAIN_RUN_ID}}"
MAIN_CKPT="${DYME_PAPER_MAIN_CHECKPOINT:-${MAIN_ROOT}/deplot_no_vs_opd_pcd/final_checkpoint}"
EVAL_ROOT="${BASE_ROOT}/eval_chartqa"
SANITY_ROOT="${BASE_ROOT}/sanity"
SANITY_MANIFEST="${SANITY_ROOT}/pcd_manifest.csv"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_paper_required_10epoch.sh [--dry-run|--run] [--run-id ID] [--main-run-id ID] [--epochs N] [--stages CSV]

Stages:
  base_eval      Evaluate the untrained/local base model.
  sft_train      Train the 10-epoch SFT baseline.
  dyme_train     Train the 10-epoch DyME/GRPO baseline.
  no_pcd_anchor  Train the 10-epoch deplot_no_vs_opd anchor.
  sanity         Run no-gold/DePlot evidence sanity checks.
  eval_required  Evaluate SFT, DyME, no-PCD anchor, and the PCD main checkpoint.

Examples:
  bash scripts/test/run_paper_required_10epoch.sh --dry-run --main-run-id pcd_no_visual_10epoch
  bash scripts/test/run_paper_required_10epoch.sh --run --stages sft_train,dyme_train
  bash scripts/test/run_paper_required_10epoch.sh --run --stages no_pcd_anchor --run-id paper10
  bash scripts/test/run_paper_required_10epoch.sh --run --stages eval_required --main-run-id paper_main10
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      BASE_ROOT="outputs/test-fast/paper-required/${RUN_ID}"
      BASELINE_ROOT="${BASE_ROOT}/baselines"
      BASELINE_LOG_ROOT="${BASE_ROOT}/logs/baselines"
      ANCHOR_RUN_ID="${DYME_PAPER_ANCHOR_RUN_ID:-${RUN_ID}_no_pcd_anchor}"
      ANCHOR_ROOT="${BASE_ROOT}/opd-deplot-anchor"
      ANCHOR_LOG_ROOT="${BASE_ROOT}/logs/opd-deplot-anchor"
      EVAL_ROOT="${BASE_ROOT}/eval_chartqa"
      SANITY_ROOT="${BASE_ROOT}/sanity"
      SANITY_MANIFEST="${SANITY_ROOT}/pcd_manifest.csv"
      shift 2
      ;;
    --main-run-id)
      MAIN_RUN_ID="$2"
      MAIN_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual/${MAIN_RUN_ID}}"
      MAIN_CKPT="${DYME_PAPER_MAIN_CHECKPOINT:-${MAIN_ROOT}/deplot_no_vs_opd_pcd/final_checkpoint}"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --stages)
      STAGES="$2"
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
    echo "--epochs must be a positive integer, got: ${EPOCHS}" >&2
    exit 2
    ;;
esac
if [[ "${EPOCHS}" -lt 1 ]]; then
  echo "--epochs must be >= 1, got: ${EPOCHS}" >&2
  exit 2
fi

stage_selected() {
  local target="$1"
  local item
  IFS=',' read -ra _STAGE_LIST <<< "${STAGES}"
  for item in "${_STAGE_LIST[@]}"; do
    item="${item// /}"
    if [[ "${item}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_stages() {
  local item
  IFS=',' read -ra _STAGE_LIST <<< "${STAGES}"
  for item in "${_STAGE_LIST[@]}"; do
    item="${item// /}"
    case "${item}" in
      base_eval|sft_train|dyme_train|no_pcd_anchor|sanity|eval_required) ;;
      "")
        echo "--stages contains an empty stage" >&2
        exit 2
        ;;
      *)
        echo "Unknown stage: ${item}" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

print_header() {
  echo "============================================================"
  echo "Paper-required non-main experiments"
  echo "mode: ${MODE}"
  echo "run id: ${RUN_ID}"
  echo "main run id: ${MAIN_RUN_ID}"
  echo "budget epochs: ${EPOCHS}"
  echo "stages: ${STAGES}"
  echo "base root: ${BASE_ROOT}"
  echo "main checkpoint: ${MAIN_CKPT}"
  echo "============================================================"
}

print_stage() {
  local stage="$1"
  echo ""
  echo "============================================================"
  echo "Stage: ${stage}"
  echo "============================================================"
}

abs_path() {
  local path="$1"
  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    *) printf '%s/%s\n' "${ROOT}" "${path}" ;;
  esac
}

run_or_print() {
  if [[ "${MODE}" == "dry-run" ]]; then
    printf '%s\n' "$*"
    return 0
  fi
  "$@"
}

eval_one() {
  local label="$1"
  local model_path="$2"
  local out_dir="${EVAL_ROOT}/${label}"
  local log_file="${out_dir}/eval.log"

  if [[ "${MODE}" == "dry-run" ]]; then
    cat <<CMD
mkdir -p ${out_dir}
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled ${PYTHON_BIN} -m accelerate.commands.launch --num_processes ${EVAL_NUM_PROCESSES} -m eval.eval_chartqa --model_path ${model_path} 2>&1 | tee ${log_file}
CMD
    return 0
  fi

  mkdir -p "${out_dir}"
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  WANDB_MODE=disabled \
    "${PYTHON_BIN}" -m accelerate.commands.launch \
      --num_processes "${EVAL_NUM_PROCESSES}" \
      -m eval.eval_chartqa \
      --model_path "${model_path}" 2>&1 | tee "${log_file}"
}

run_base_eval() {
  print_stage "base_eval"
  eval_one "base" "${STUDENT_MODEL}"
}

run_sft_train() {
  print_stage "sft_train"
  if [[ "${MODE}" == "dry-run" ]]; then
    cat <<CMD
DYME_FAST_OUTPUT_ROOT=${BASELINE_ROOT} \\
DYME_LOG_DIR=${BASELINE_LOG_ROOT}/sft \\
DYME_FAST_SFT_EPOCHS=${EPOCHS} \\
DYME_FAST_NUM_TRAIN_EPOCHS=${EPOCHS} \\
WANDB_MODE=disabled \\
bash scripts/test/train_sft.sh
CMD
    return 0
  fi
  DYME_FAST_OUTPUT_ROOT="${BASELINE_ROOT}" \
  DYME_LOG_DIR="${BASELINE_LOG_ROOT}/sft" \
  DYME_FAST_SFT_EPOCHS="${EPOCHS}" \
  DYME_FAST_NUM_TRAIN_EPOCHS="${EPOCHS}" \
  WANDB_MODE=disabled \
    bash scripts/test/train_sft.sh
}

run_dyme_train() {
  print_stage "dyme_train"
  if [[ "${MODE}" == "dry-run" ]]; then
    cat <<CMD
DYME_FAST_OUTPUT_ROOT=${BASELINE_ROOT} \\
DYME_LOG_DIR=${BASELINE_LOG_ROOT}/dyme \\
DYME_FAST_NUM_TRAIN_EPOCHS=${EPOCHS} \\
WANDB_MODE=disabled \\
bash scripts/test/train_dyme.sh
CMD
    return 0
  fi
  DYME_FAST_OUTPUT_ROOT="${BASELINE_ROOT}" \
  DYME_LOG_DIR="${BASELINE_LOG_ROOT}/dyme" \
  DYME_FAST_NUM_TRAIN_EPOCHS="${EPOCHS}" \
  WANDB_MODE=disabled \
    bash scripts/test/train_dyme.sh
}

run_no_pcd_anchor() {
  print_stage "no_pcd_anchor"
  if [[ "${MODE}" == "dry-run" ]]; then
    cat <<CMD
DYME_DEPLOT_ABLATION_EPOCHS=${EPOCHS} \\
DYME_DEPLOT_ABLATION_OUTPUT_ROOT=${ANCHOR_ROOT} \\
DYME_DEPLOT_ABLATION_LOG_ROOT=${ANCHOR_LOG_ROOT} \\
bash scripts/test/run_opd_deplot_ablation.sh --run --epochs ${EPOCHS} --run-id ${ANCHOR_RUN_ID} --variants deplot_no_vs_opd
CMD
    return 0
  fi
  DYME_DEPLOT_ABLATION_EPOCHS="${EPOCHS}" \
  DYME_DEPLOT_ABLATION_OUTPUT_ROOT="${ANCHOR_ROOT}" \
  DYME_DEPLOT_ABLATION_LOG_ROOT="${ANCHOR_LOG_ROOT}" \
    bash scripts/test/run_opd_deplot_ablation.sh \
      --run \
      --epochs "${EPOCHS}" \
      --run-id "${ANCHOR_RUN_ID}" \
      --variants deplot_no_vs_opd
}

run_sanity() {
  print_stage "sanity"
  local main_run_dir
  local main_eval_log
  local main_candidate_glob
  local anchor_run_dir
  local anchor_eval_log
  local anchor_candidate_glob
  main_run_dir="$(abs_path "${MAIN_ROOT}/deplot_no_vs_opd_pcd")"
  main_eval_log="$(abs_path "${EVAL_ROOT}/ours_pcd_no_visual/eval.log")"
  main_candidate_glob="$(abs_path "${MAIN_ROOT}/deplot_no_vs_opd_pcd/teacher_probe_candidates/rank*.jsonl")"
  anchor_run_dir="$(abs_path "${ANCHOR_ROOT}/deplot_no_vs_opd")"
  anchor_eval_log="$(abs_path "${EVAL_ROOT}/no_pcd_anchor/eval.log")"
  anchor_candidate_glob="$(abs_path "${ANCHOR_ROOT}/deplot_no_vs_opd/teacher_probe_candidates/rank*.jsonl")"
  if [[ "${MODE}" == "dry-run" ]]; then
    cat <<CMD
mkdir -p ${SANITY_ROOT}
${PYTHON_BIN} scripts/check_chartqa_teacher_evidence.py --input data/chartqa/train_medium_vf_full.json --json | tee ${SANITY_ROOT}/teacher_evidence_health.json
cat > ${SANITY_MANIFEST} <<'CSV'
variant,role,run_dir,train_log,eval_log,candidate_log_glob,config_path,enabled
deplot_no_vs_opd_pcd,ours,${main_run_dir},,${main_eval_log},${main_candidate_glob},,1
deplot_no_vs_opd,w/o PCD,${anchor_run_dir},,${anchor_eval_log},${anchor_candidate_glob},,1
CSV
${PYTHON_BIN} scripts/analysis/pcd_probe_controls.py --manifest ${SANITY_MANIFEST} --variant deplot_no_vs_opd_pcd --out ${SANITY_ROOT}/pcd_probe_controls.csv
CMD
    return 0
  fi
  mkdir -p "${SANITY_ROOT}"
  "${PYTHON_BIN}" scripts/check_chartqa_teacher_evidence.py \
    --input data/chartqa/train_medium_vf_full.json \
    --json | tee "${SANITY_ROOT}/teacher_evidence_health.json"
  cat > "${SANITY_MANIFEST}" <<CSV
variant,role,run_dir,train_log,eval_log,candidate_log_glob,config_path,enabled
deplot_no_vs_opd_pcd,ours,${main_run_dir},,${main_eval_log},${main_candidate_glob},,1
deplot_no_vs_opd,w/o PCD,${anchor_run_dir},,${anchor_eval_log},${anchor_candidate_glob},,1
CSV
  "${PYTHON_BIN}" scripts/analysis/pcd_probe_controls.py \
    --manifest "${SANITY_MANIFEST}" \
    --variant deplot_no_vs_opd_pcd \
    --out "${SANITY_ROOT}/pcd_probe_controls.csv"
}

run_eval_required() {
  print_stage "eval_required"
  eval_one "sft" "${BASELINE_ROOT}/sft/final_checkpoint"
  eval_one "dyme" "${BASELINE_ROOT}/dyme/final_checkpoint"
  eval_one "no_pcd_anchor" "${ANCHOR_ROOT}/deplot_no_vs_opd/final_checkpoint"
  eval_one "ours_pcd_no_visual" "${MAIN_CKPT}"
}

validate_stages
print_header

if stage_selected "base_eval"; then
  run_base_eval
fi
if stage_selected "sft_train"; then
  run_sft_train
fi
if stage_selected "dyme_train"; then
  run_dyme_train
fi
if stage_selected "no_pcd_anchor"; then
  run_no_pcd_anchor
fi
if stage_selected "sanity"; then
  run_sanity
fi
if stage_selected "eval_required"; then
  run_eval_required
fi

if [[ "${MODE}" == "dry-run" ]]; then
  echo ""
  echo "Dry-run only. Re-run with --run and optionally --stages to launch selected work."
else
  echo ""
  echo "Paper-required non-main experiments finished: ${BASE_ROOT}"
fi
