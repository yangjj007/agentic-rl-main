#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DRY_RUN=0
RUN_DIR=""
LABEL=""
MIN_EPOCH="${DYME_CHARTQA_EVAL_SWEEP_MIN_EPOCH:-6}"
TOTAL_EPOCHS="${DYME_CHARTQA_EVAL_SWEEP_TOTAL_EPOCHS:-10}"
STEPS_PER_EPOCH="${DYME_CHARTQA_STEPS_PER_EPOCH:-147}"
RESULTS_DIR=""
PYTHON_BIN="${DYME_PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
ACCEL_CONFIG="${DYME_CHARTQA_EVAL_ACCELERATE_CONFIG:-default_config_8gpu.yaml}"
NUM_PROCESSES="${DYME_CHARTQA_EVAL_NUM_PROCESSES:-8}"
VISIBLE_DEVICES="${DYME_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
EVAL_BATCH_SIZE="${DYME_EVAL_BATCH_SIZE:-1}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/eval_chartqa_checkpoint_sweep.sh --run-dir DIR --label LABEL
    [--min-epoch 6] [--total-epochs 10] [--steps-per-epoch 147]
    [--results-dir DIR] [--dry-run]

Evaluates checkpoint-* directories whose step is >= min_epoch * steps_per_epoch,
plus final_checkpoint. Writes eval_chartqa/summary.csv and a lightweight
sweep_manifest.csv suitable for docs/experiment_results.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --run-dir) RUN_DIR="${2:?missing run dir}"; shift 2 ;;
    --label) LABEL="${2:?missing label}"; shift 2 ;;
    --min-epoch) MIN_EPOCH="${2:?missing min epoch}"; shift 2 ;;
    --total-epochs) TOTAL_EPOCHS="${2:?missing total epochs}"; shift 2 ;;
    --steps-per-epoch) STEPS_PER_EPOCH="${2:?missing steps per epoch}"; shift 2 ;;
    --results-dir) RESULTS_DIR="${2:?missing results dir}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:?missing python bin}"; shift 2 ;;
    --accelerate-config) ACCEL_CONFIG="${2:?missing accelerate config}"; shift 2 ;;
    --num-processes) NUM_PROCESSES="${2:?missing process count}"; shift 2 ;;
    --cuda-visible-devices) VISIBLE_DEVICES="${2:?missing devices}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ -n "${LABEL}" ]] || { echo "--label is required" >&2; exit 2; }
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/eval_chartqa/results}"

for value_name in MIN_EPOCH TOTAL_EPOCHS STEPS_PER_EPOCH NUM_PROCESSES EVAL_BATCH_SIZE; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*) echo "${value_name} must be a non-negative integer, got: ${value}" >&2; exit 2 ;;
  esac
done
[[ "${STEPS_PER_EPOCH}" -ge 1 ]] || { echo "steps per epoch must be >= 1" >&2; exit 2; }

min_step=$(( MIN_EPOCH * STEPS_PER_EPOCH ))
eval_dir="${RUN_DIR}/eval_chartqa"

checkpoint_step() {
  local name
  name="$(basename "$1")"
  case "${name}" in
    checkpoint-*)
      echo "${name#checkpoint-}"
      ;;
    final_checkpoint)
      echo "999999999"
      ;;
    *)
      echo "-1"
      ;;
  esac
}

discover_checkpoints() {
  local path step
  if [[ -d "${RUN_DIR}" ]]; then
    while IFS= read -r path; do
      step="$(checkpoint_step "${path}")"
      [[ "${step}" =~ ^[0-9]+$ ]] || continue
      (( step >= min_step )) || continue
      printf '%s\n' "${path}"
    done < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    if [[ -d "${RUN_DIR}/final_checkpoint" ]]; then
      printf '%s\n' "${RUN_DIR}/final_checkpoint"
    fi
  fi
}

mapfile -t checkpoints < <(discover_checkpoints)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoints found after epoch ${MIN_EPOCH} under ${RUN_DIR}" >&2
  exit 2
fi

echo "============================================================"
echo "ChartQA checkpoint eval sweep"
echo "label: ${LABEL}"
echo "run dir: ${RUN_DIR}"
echo "min epoch: ${MIN_EPOCH}"
echo "total epochs: ${TOTAL_EPOCHS}"
echo "steps per epoch: ${STEPS_PER_EPOCH}"
echo "min step: ${min_step}"
echo "eval dir: ${eval_dir}"
echo "results dir: ${RESULTS_DIR}"
echo "checkpoints:"
printf '  %s\n' "${checkpoints[@]}"
echo "============================================================"

run_eval_for_checkpoint() {
  local checkpoint_path="$1"
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint_path}")"
  local log_path="${eval_dir}/eval_${checkpoint_name}_bsz${EVAL_BATCH_SIZE}_gpuall.log"
  local eval_env=(
    "CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
    "PYTHONUNBUFFERED=1"
    "HF_DATASETS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "WANDB_MODE=disabled"
    "DYME_EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  )
  local eval_cmd=(
    "${PYTHON_BIN}" -m accelerate.commands.launch
    --config_file "${ACCEL_CONFIG}" --num_processes "${NUM_PROCESSES}"
    -m eval.eval_chartqa --model_path "${checkpoint_path}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'EVAL %s: env' "${checkpoint_name}"
    printf ' %q' "${eval_env[@]}"
    printf ' %q' "${eval_cmd[@]}"
    printf ' 2>&1 | tee %q\n' "${log_path}"
    return 0
  fi
  mkdir -p "${eval_dir}"
  echo "[eval-sweep] ${LABEL}/${checkpoint_name}"
  env "${eval_env[@]}" "${eval_cmd[@]}" 2>&1 | tee "${log_path}"
}

for checkpoint in "${checkpoints[@]}"; do
  run_eval_for_checkpoint "${checkpoint}"
done

parse_cmd=("${PYTHON_BIN}" scripts/test/parse_eval_chartqa_logs.py "${eval_dir}" "${eval_dir}/summary.csv")
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'PARSE:'
  printf ' %q' "${parse_cmd[@]}"
  printf '\n'
  printf 'RESULT: mkdir -p %q && cp %q %q && write %q\n' \
    "${RESULTS_DIR}" "${eval_dir}/summary.csv" "${RESULTS_DIR}/summary.csv" "${RESULTS_DIR}/sweep_manifest.csv"
  exit 0
fi

"${parse_cmd[@]}"
mkdir -p "${RESULTS_DIR}"
cp "${eval_dir}/summary.csv" "${RESULTS_DIR}/summary.csv"
{
  echo "label,total_epochs,min_epoch,steps_per_epoch,min_step,run_dir,eval_dir,summary_csv"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${LABEL}" "${TOTAL_EPOCHS}" "${MIN_EPOCH}" "${STEPS_PER_EPOCH}" "${min_step}" \
    "${RUN_DIR}" "${eval_dir}" "${eval_dir}/summary.csv"
} > "${RESULTS_DIR}/sweep_manifest.csv"
