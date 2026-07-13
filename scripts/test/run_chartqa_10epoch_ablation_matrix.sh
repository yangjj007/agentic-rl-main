#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

MODE="dry-run"
RUN_ID="${DYME_CHARTQA_ABLATION_RUN_ID:-chartqa_10epoch_matrix}"
EPOCHS="${DYME_CHARTQA_ABLATION_EPOCHS:-10}"
SMOKE=0
SMOKE_STEPS="${DYME_CHARTQA_ABLATION_SMOKE_STEPS:-2}"
SHARD_INDEX=0
SHARD_COUNT=1
OUTPUT_ROOT="${DYME_CHARTQA_ABLATION_OUTPUT_ROOT:-outputs/test-fast/chartqa-ablation}"
LOG_ROOT="${DYME_CHARTQA_ABLATION_LOG_ROOT:-outputs/test-fast/logs/chartqa-ablation}"
RESULTS_ROOT="${DYME_CHARTQA_ABLATION_RESULTS_ROOT:-docs/experiment_results/chartqa-ablation}"
VARIANT_FILTER="${DYME_CHARTQA_ABLATION_VARIANTS:-}"
STAGES="${DYME_CHARTQA_ABLATION_STAGES:-train,eval}"
RESUME="${DYME_CHARTQA_ABLATION_RESUME:-none}"
SPEED_PROFILE="${DYME_CHARTQA_ABLATION_SPEED_PROFILE:-canonical}"

PYTHON_BIN="${DYME_PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
EVAL_ACCEL="${DYME_CHARTQA_ABLATION_EVAL_ACCELERATE_CONFIG:-default_config_8gpu.yaml}"
VISIBLE_DEVICES="${DYME_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
EVAL_NUM_PROCESSES="${DYME_CHARTQA_ABLATION_EVAL_NUM_PROCESSES:-${DYME_DYME_EVAL_NUM_PROCESSES:-8}}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh [--dry-run|--run]
    [--run-id ID] [--epochs N] [--smoke]
    [--shard-index I --shard-count N]
    [--variants comma,separated,labels]
    [--stages train|eval|train,eval]
    [--resume none|auto|/path/to/checkpoint]

Default is a dry-run of the full ChartQA 10epoch matrix plus PCD eval commands.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --run) MODE="run"; shift ;;
    --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
    --epochs) EPOCHS="${2:?missing epoch count}"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --shard-index) SHARD_INDEX="${2:?missing shard index}"; shift 2 ;;
    --shard-count) SHARD_COUNT="${2:?missing shard count}"; shift 2 ;;
    --variants) VARIANT_FILTER="${2:?missing variant list}"; shift 2 ;;
    --stages) STAGES="${2:?missing stage list}"; shift 2 ;;
    --resume) RESUME="${2:?missing resume value}"; shift 2 ;;
    --speed-profile) SPEED_PROFILE="${2:?missing speed profile}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in EPOCHS SMOKE_STEPS SHARD_INDEX SHARD_COUNT EVAL_NUM_PROCESSES; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*) echo "${value_name} must be a non-negative integer, got: ${value}" >&2; exit 2 ;;
  esac
done
[[ "${EPOCHS}" -ge 1 ]] || { echo "epochs must be >= 1" >&2; exit 2; }
[[ "${SMOKE_STEPS}" -ge 1 ]] || { echo "smoke steps must be >= 1" >&2; exit 2; }
[[ "${SHARD_COUNT}" -ge 1 ]] || { echo "shard count must be >= 1" >&2; exit 2; }
[[ "${SHARD_INDEX}" -lt "${SHARD_COUNT}" ]] || {
  echo "shard index ${SHARD_INDEX} must be smaller than shard count ${SHARD_COUNT}" >&2
  exit 2
}
case "${SPEED_PROFILE}" in
  canonical|fast60) ;;
  *) echo "Unknown speed profile: ${SPEED_PROFILE}" >&2; exit 2 ;;
esac

STAGES="${STAGES// /}"
for stage in ${STAGES//,/ }; do
  case "${stage}" in
    train|eval) ;;
    *) echo "Unknown stage: ${stage} (expected train, eval, or train,eval)" >&2; exit 2 ;;
  esac
done

LABELS=(
  dyme_pure_original
  dyme_full_original
  oracle_official_best_4e
  gold_hidden_no_opd
  gold_hidden_uncond_opd
  gold_hidden_routed_opd_fixed
  clrc_full
  clrc_target020
  grpo_only_matched
  opd_only_matched
  fallback_only_matched
  oracle_clean_no_full_hint
  token_reliability_clrc
  mixed_group_shortest_correct_hard_replay
)

stage_enabled() {
  local stage="$1"
  [[ ",${STAGES}," == *",${stage},"* ]]
}

label_known() {
  local needle="$1"
  local label
  for label in "${LABELS[@]}"; do
    [[ "${label}" == "${needle}" ]] && return 0
  done
  return 1
}

label_selected() {
  local needle="$1"
  local requested
  [[ -z "${VARIANT_FILTER}" ]] && return 0
  IFS=',' read -r -a requested <<< "${VARIANT_FILTER}"
  local label
  for label in "${requested[@]}"; do
    [[ "${label}" == "${needle}" ]] && return 0
  done
  return 1
}

if [[ -n "${VARIANT_FILTER}" ]]; then
  IFS=',' read -r -a requested_labels <<< "${VARIANT_FILTER}"
  for requested in "${requested_labels[@]}"; do
    if [[ "${requested}" == "vold_cold_start" || "${requested}" == "ssopd_mixed_group" ]]; then
      echo "retired near-neighbor label: ${requested}; use a true matched implementation, not an approximate alias" >&2
      exit 2
    fi
    label_known "${requested}" || { echo "Unknown matrix variant: ${requested}" >&2; exit 2; }
  done
fi

pcd_variant_for_label() {
  case "$1" in
    oracle_official_best_4e) echo "deplot_no_vs_opd_pcd_oracle_hint" ;;
    gold_hidden_no_opd) echo "deplot_no_vs_opd_pcd_gold_hidden_no_opd" ;;
    gold_hidden_uncond_opd) echo "deplot_no_vs_opd_pcd_gold_hidden_uncond_opd_no_full_hint_hard_sft" ;;
    gold_hidden_routed_opd_fixed) echo "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_fixed" ;;
    clrc_full) echo "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision" ;;
    clrc_target020) echo "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_target020" ;;
    grpo_only_matched) echo "deplot_no_vs_opd_pcd_gold_hidden_grpo_only" ;;
    opd_only_matched) echo "deplot_no_vs_opd_pcd_gold_hidden_opd_only_no_full_hint_hard_sft" ;;
    fallback_only_matched) echo "deplot_no_vs_opd_pcd_gold_hidden_fallback_only" ;;
    oracle_clean_no_full_hint) echo "deplot_no_vs_opd_pcd_oracle_hint_opd_no_full_hint_hard_sft_adaptive_supervision" ;;
    token_reliability_clrc) echo "deplot_no_vs_opd_pcd_gold_hidden_token_reliability_clrc" ;;
    mixed_group_shortest_correct_hard_replay) echo "deplot_no_vs_opd_pcd_gold_hidden_mixed_group_shortest_correct_hard_replay" ;;
    *) return 1 ;;
  esac
}

epochs_for_label() {
  if [[ "$1" == "oracle_official_best_4e" ]]; then
    echo "4"
  else
    echo "${EPOCHS}"
  fi
}

run_or_print() {
  if [[ "${MODE}" == "dry-run" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

print_or_run_pcd_eval() {
  local label="$1"
  local pcd_variant="$2"
  local out_dir="${OUTPUT_ROOT}/${RUN_ID}/${label}/${pcd_variant}"
  local eval_dir="${out_dir}/eval_chartqa"
  local eval_log="${eval_dir}/eval_final_checkpoint_bsz1_gpuall.log"
  local result_dir="${RESULTS_ROOT}/${RUN_ID}/${label}"
  local eval_env=(
    "CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
    "PYTHONUNBUFFERED=1"
    "HF_DATASETS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "WANDB_MODE=disabled"
    "DYME_EVAL_BATCH_SIZE=1"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  )
  local eval_cmd=(
    "${PYTHON_BIN}" -m accelerate.commands.launch
    --config_file "${EVAL_ACCEL}" --num_processes "${EVAL_NUM_PROCESSES}"
    -m eval.eval_chartqa --model_path "${out_dir}/final_checkpoint"
  )
  local parse_cmd=(
    "${PYTHON_BIN}" scripts/test/parse_eval_chartqa_logs.py
    "${eval_dir}" "${eval_dir}/summary.csv"
  )

  if [[ "${MODE}" == "dry-run" ]]; then
    printf 'EVAL: env'
    printf ' %q' "${eval_env[@]}"
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
    printf 'PARSE:'
    printf ' %q' "${parse_cmd[@]}"
    printf '\n'
    printf 'RESULT: mkdir -p %q && cp %q %q\n' "${result_dir}" "${eval_dir}/summary.csv" "${result_dir}/summary.csv"
    return 0
  fi

  mkdir -p "${eval_dir}"
  [[ -d "${out_dir}/final_checkpoint" ]] || {
    echo "missing final checkpoint for ${label}: ${out_dir}/final_checkpoint" >&2
    exit 2
  }
  env "${eval_env[@]}" "${eval_cmd[@]}" 2>&1 | tee "${eval_log}"
  "${parse_cmd[@]}"
  mkdir -p "${result_dir}"
  cp "${eval_dir}/summary.csv" "${result_dir}/summary.csv"
  {
    echo "label,pcd_variant,epochs,output_dir,log_dir,eval_summary,eval_log"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${pcd_variant}" "$(epochs_for_label "${label}")" "${out_dir}" "${LOG_ROOT}/${RUN_ID}/${label}/${pcd_variant}" "${eval_dir}/summary.csv" "${eval_log}"
  } > "${result_dir}/manifest.csv"
}

echo "============================================================"
echo "ChartQA matched ablation matrix"
echo "mode: ${MODE}"
echo "run id: ${RUN_ID}"
echo "epochs: ${EPOCHS}"
echo "smoke: ${SMOKE}"
echo "smoke steps: ${SMOKE_STEPS}"
echo "shard: ${SHARD_INDEX}/${SHARD_COUNT}"
echo "variants: ${VARIANT_FILTER:-<all>}"
echo "stages: ${STAGES}"
echo "resume: ${RESUME}"
echo "results root: ${RESULTS_ROOT}/${RUN_ID}"
echo "============================================================"

if [[ "${MODE}" == "run" ]]; then
  mkdir -p "${RESULTS_ROOT}/${RUN_ID}"
  {
    echo "label,epochs,runner_variant,train_enabled,eval_enabled,output_root,log_root,result_dir"
  } > "${RESULTS_ROOT}/${RUN_ID}/matrix_manifest.csv"
fi

for index in "${!LABELS[@]}"; do
  (( index % SHARD_COUNT == SHARD_INDEX )) || continue
  label="${LABELS[index]}"
  label_selected "${label}" || continue
  label_epochs="$(epochs_for_label "${label}")"

  echo ""
  echo "Variant: ${label}"

  if [[ "${label}" == "dyme_pure_original" || "${label}" == "dyme_full_original" ]]; then
    dyme_variant="pure"
    [[ "${label}" == "dyme_full_original" ]] && dyme_variant="full"
    if [[ "${MODE}" == "run" ]]; then
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${label}" "${label_epochs}" "${dyme_variant}" "$(stage_enabled train && echo 1 || echo 0)" "$(stage_enabled eval && echo 1 || echo 0)" \
        "${OUTPUT_ROOT}" "${LOG_ROOT}" "${RESULTS_ROOT}/${RUN_ID}/${label}" >> "${RESULTS_ROOT}/${RUN_ID}/matrix_manifest.csv"
    fi
    if stage_enabled train || stage_enabled eval; then
      command=(
        env
        "DYME_DYME_EPOCHS=${label_epochs}"
        "DYME_DYME_RUN_ID=${RUN_ID}_${label}"
        "DYME_DYME_OUTPUT_ROOT=${OUTPUT_ROOT}"
        "DYME_DYME_LOG_ROOT=${LOG_ROOT}"
        "DYME_DYME_RESULTS_ROOT=${RESULTS_ROOT}/${RUN_ID}/${label}"
      )
      if [[ "${SMOKE}" == "1" ]]; then
        command+=("DYME_DYME_MAX_STEPS=${SMOKE_STEPS}")
        command+=("DYME_DYME_SAVE_STRATEGY=no")
        command+=("DYME_SKIP_FINAL_SAVE=1")
      fi
      command+=(
        bash scripts/test/run_dyme_matched_4epoch.sh
        --variant "${dyme_variant}"
        --epochs "${label_epochs}"
        --resume "${RESUME}"
        --stages "${STAGES}"
      )
      run_or_print "${command[@]}"
    fi
    continue
  fi

  pcd_variant="$(pcd_variant_for_label "${label}")"
  if [[ "${MODE}" == "run" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${label_epochs}" "${pcd_variant}" "$(stage_enabled train && echo 1 || echo 0)" "$(stage_enabled eval && echo 1 || echo 0)" \
      "${OUTPUT_ROOT}/${RUN_ID}/${label}" "${LOG_ROOT}/${RUN_ID}/${label}" "${RESULTS_ROOT}/${RUN_ID}/${label}" >> "${RESULTS_ROOT}/${RUN_ID}/matrix_manifest.csv"
  fi
  if stage_enabled train; then
    command=(
      env
      "DYME_PCD_RUN_ID=${RUN_ID}_${label}"
      "DYME_PCD_OUTPUT_ROOT=${OUTPUT_ROOT}/${RUN_ID}/${label}"
      "DYME_PCD_LOG_ROOT=${LOG_ROOT}/${RUN_ID}/${label}"
    )
    if [[ "${SMOKE}" == "1" ]]; then
      command+=("DYME_PCD_MAX_STEPS=${SMOKE_STEPS}")
      command+=("DYME_SAVE_STRATEGY=no")
      command+=("DYME_SKIP_FINAL_SAVE=1")
    fi
    if [[ "${label}" == "token_reliability_clrc" ]]; then
      command+=("DYME_OPSD_TOKEN_WEIGHTING=1")
    fi
    command+=(
      bash scripts/test/run_pcd_no_visual.sh "${label_epochs}"
      --resume "${RESUME}"
      --variant "${pcd_variant}"
      --speed-profile "${SPEED_PROFILE}"
    )
    run_or_print "${command[@]}"
  fi
  if stage_enabled eval; then
    print_or_run_pcd_eval "${label}" "${pcd_variant}"
  fi
done
