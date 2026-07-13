#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${DYME_PCD_RUN_ID:-pcd_oracle_new_directions_4epoch}"
DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
EVAL_BATCH_SIZE="${DYME_EVAL_BATCH_SIZE:-1}"
EVAL_NUM_PROCESSES="${DYME_EVAL_NUM_PROCESSES:-1}"
PORT="${DYME_EVAL_PORT:-29610}"
OUT_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual}/${RUN_ID}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/eval_pcd_oracle_new_directions.sh [--dry-run|--run] [--run-id ID]

Evaluates checkpoint-147/294/441/588 and final_checkpoint for the three
oracle PCD new-direction variants, then writes eval_chartqa/summary.csv
under each variant directory.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run)
      DRY_RUN=0
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      OUT_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual}/${RUN_ID}"
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

VARIANTS=(
  "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward"
  "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay"
  "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay"
)
CHECKPOINTS=(
  "checkpoint-147"
  "checkpoint-294"
  "checkpoint-441"
  "checkpoint-588"
  "final_checkpoint"
)

echo "PCD oracle new-direction eval"
echo "run id: ${RUN_ID}"
echo "out root: ${OUT_ROOT}"
echo "eval batch size: ${EVAL_BATCH_SIZE}"
echo "eval num processes: ${EVAL_NUM_PROCESSES}"
echo "mode: $([[ "${DRY_RUN}" == "1" ]] && echo dry-run || echo run)"

for variant in "${VARIANTS[@]}"; do
  variant_dir="${OUT_ROOT}/${variant}"
  eval_dir="${variant_dir}/eval_chartqa"
  echo "============================================================"
  echo "variant: ${variant}"
  mkdir -p "${eval_dir}"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    model_path="${variant_dir}/${checkpoint}"
    label="${checkpoint}"
    log_file="${eval_dir}/eval_${label}_$(date +%Y%m%d_%H%M%S).log"
    eval_cmd=(
      env
      "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
      "PYTHONUNBUFFERED=1"
      "HF_DATASETS_OFFLINE=1"
      "HF_HUB_OFFLINE=1"
      "TRANSFORMERS_OFFLINE=1"
      "WANDB_MODE=disabled"
      "DYME_EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
      "${PYTHON_BIN}" -m accelerate.commands.launch
      --config_file scripts/test/accelerate_single_gpu_no_deepspeed.yaml
      --num_processes "${EVAL_NUM_PROCESSES}"
      --main_process_port "${PORT}"
      -m eval.eval_chartqa
      --model_path "${model_path}"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "checkpoint: ${checkpoint}"
      printf 'would eval:'
      printf ' %q' "${eval_cmd[@]}"
      printf ' 2>&1 | tee %q\n' "${log_file}"
    else
      if [[ ! -d "${model_path}" ]]; then
        echo "[skip] missing model path: ${model_path}" | tee -a "${eval_dir}/missing.log"
        continue
      fi
      echo "[eval] ${variant}/${checkpoint}"
      "${eval_cmd[@]}" 2>&1 | tee "${log_file}"
    fi
  done
  parse_cmd=("${PYTHON_BIN}" scripts/test/parse_eval_chartqa_logs.py "${eval_dir}" "${eval_dir}/summary.csv")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'would parse:'
    printf ' %q' "${parse_cmd[@]}"
    printf '\n'
  else
    "${parse_cmd[@]}"
  fi
done
