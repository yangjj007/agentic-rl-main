#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DRY_RUN=0
RESUME="none"
VARIANT="pure"
EPOCHS="${DYME_DYME_EPOCHS:-4}"
STAGES="${DYME_DYME_STAGES:-train,eval}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --resume) RESUME="${2:?missing resume value}"; shift 2 ;;
        --variant) VARIANT="${2:?missing variant value}"; shift 2 ;;
        --epochs) EPOCHS="${2:?missing epoch value}"; shift 2 ;;
        --stages) STAGES="${2:?missing stages value}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${EPOCHS}" in
    ''|*[!0-9]*) echo "--epochs must be a positive integer, got: ${EPOCHS}" >&2; exit 2 ;;
esac
if [[ "${EPOCHS}" -lt 1 ]]; then
    echo "--epochs must be >= 1, got: ${EPOCHS}" >&2
    exit 2
fi
STAGES="${STAGES// /}"
stage_enabled() {
    local stage="$1"
    [[ ",${STAGES}," == *",${stage},"* ]]
}
for stage in ${STAGES//,/ }; do
    case "${stage}" in
        train|eval) ;;
        *) echo "Unknown stage: ${stage} (expected train, eval, or train,eval)" >&2; exit 2 ;;
    esac
done

case "${VARIANT}" in
    pure)
        VARIANT_TITLE="Pure"
        VARIANT_STEM="pure_dyme_matched"
        CONFIG_PATH="scripts/test/config/config_dyme_matched.py"
        VISUAL_CHECKER=0
        VISUAL_REFINER=0
        VISUAL_PREFETCH_IC=0
        ;;
    full)
        VARIANT_TITLE="Full"
        VARIANT_STEM="full_dyme_matched"
        CONFIG_PATH="scripts/test/config/config_dyme_full_matched.py"
        VISUAL_CHECKER=1
        VISUAL_REFINER=1
        VISUAL_PREFETCH_IC=1
        ;;
    *)
        echo "Unsupported variant: ${VARIANT} (expected pure or full)" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${DYME_PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
RUN_ID="${DYME_DYME_RUN_ID:-dyme_matched_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${DYME_DYME_OUTPUT_ROOT:-outputs/test-fast/dyme-matched}"
LOG_ROOT="${DYME_DYME_LOG_ROOT:-outputs/test-fast/logs/dyme-matched}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}/${VARIANT_STEM}"
LOG_DIR="${LOG_ROOT}/${RUN_ID}/${VARIANT_STEM}"
TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_DIR="${OUT_DIR}/eval_chartqa"
EVAL_LOG="${EVAL_DIR}/eval_final_checkpoint_bsz1_gpuall.log"
RESULTS_ROOT="${DYME_DYME_RESULTS_ROOT:-}"
SAVE_STRATEGY="${DYME_DYME_SAVE_STRATEGY:-steps}"
SAVE_STEPS="${DYME_DYME_SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${DYME_DYME_SAVE_TOTAL_LIMIT:-3}"
TRAIN_ACCEL="${DYME_DYME_ACCELERATE_CONFIG:-default_config_8gpu_deepspeed_zero1.yaml}"
EVAL_ACCEL="${DYME_DYME_EVAL_ACCELERATE_CONFIG:-default_config_8gpu.yaml}"
VISIBLE_DEVICES="${DYME_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
NUM_PROCESSES="${DYME_DYME_NUM_PROCESSES:-8}"
EVAL_NUM_PROCESSES="${DYME_DYME_EVAL_NUM_PROCESSES:-${NUM_PROCESSES}}"
MAX_STEPS="${DYME_DYME_MAX_STEPS:-}"
MODEL_ROOT="${DYME_MODEL_ROOT:-${ROOT}/models}"
STUDENT_MODEL="${DYME_STUDENT_MODEL:-${MODEL_ROOT}/llava-0.5b-ov}"

resume_args=()
if [[ "${RESUME}" == "auto" ]]; then
    latest="$(find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1)"
    [[ -n "${latest}" ]] || { echo "No checkpoint found under ${OUT_DIR}" >&2; exit 2; }
    resume_args=(--resume_from_checkpoint "${latest}")
elif [[ "${RESUME}" != "none" ]]; then
    resume_args=(--resume_from_checkpoint "${RESUME}")
fi

train_env=(
    CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
    PYTHONUNBUFFERED=1
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    WANDB_MODE=disabled
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    DYME_OUTPUT_DIR="${OUT_DIR}"
    DYME_LOG_DIR="${LOG_DIR}"
    DYME_NUM_TRAIN_EPOCHS="${EPOCHS}"
    DYME_SAVE_STRATEGY="${SAVE_STRATEGY}"
    DYME_SAVE_STEPS="${SAVE_STEPS}"
    DYME_SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}"
    DYME_VISUAL_CHECKER="${VISUAL_CHECKER}"
    DYME_VISUAL_REFINER="${VISUAL_REFINER}"
    DYME_VISUAL_PREFETCH_IC="${VISUAL_PREFETCH_IC}"
    DYME_STUDENT_MODEL="${STUDENT_MODEL}"
)
if [[ -n "${MAX_STEPS}" ]]; then
    train_env+=(DYME_MAX_STEPS="${MAX_STEPS}")
fi
train_cmd=(
    "${PYTHON_BIN}" -m accelerate.commands.launch
    --config_file "${TRAIN_ACCEL}" --num_processes "${NUM_PROCESSES}"
    main.py --config "${CONFIG_PATH}" --mode rl
    "${resume_args[@]}"
)
eval_env=(
    CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
    PYTHONUNBUFFERED=1
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    WANDB_MODE=disabled
    DYME_EVAL_BATCH_SIZE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
eval_cmd=(
    "${PYTHON_BIN}" -m accelerate.commands.launch
    --config_file "${EVAL_ACCEL}" --num_processes "${EVAL_NUM_PROCESSES}"
    -m eval.eval_chartqa --model_path "${OUT_DIR}/final_checkpoint"
)
parse_cmd=(
    "${PYTHON_BIN}" scripts/test/parse_eval_chartqa_logs.py
    "${EVAL_DIR}" "${EVAL_DIR}/summary.csv"
)

echo "============================================================"
echo "Matched ${VARIANT_TITLE} DyME ${EPOCHS}epoch run"
echo "run id: ${RUN_ID}"
echo "variant: ${VARIANT}"
echo "output dir: ${OUT_DIR}"
echo "log dir: ${LOG_DIR}"
echo "resume: ${RESUME}"
echo "stages: ${STAGES}"
if [[ -n "${MAX_STEPS}" ]]; then
    echo "max steps: ${MAX_STEPS}"
else
    echo "max steps: <none>"
fi
if [[ -n "${RESULTS_ROOT}" ]]; then
    echo "results root: ${RESULTS_ROOT}"
fi
echo "============================================================"
if stage_enabled train; then
    printf 'TRAIN: env'; printf ' %q' "${train_env[@]}"; printf ' %q' "${train_cmd[@]}"; printf '\n'
fi
if stage_enabled eval; then
    printf 'EVAL: env'; printf ' %q' "${eval_env[@]}"; printf ' %q' "${eval_cmd[@]}"; printf '\n'
    printf 'PARSE:'; printf ' %q' "${parse_cmd[@]}"; printf '\n'
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    exit 0
fi

mkdir -p "${LOG_DIR}"
if stage_enabled train; then
    env "${train_env[@]}" "${train_cmd[@]}" 2>&1 | tee "${TRAIN_LOG}"
fi
if stage_enabled eval; then
    mkdir -p "${EVAL_DIR}"
    [[ -d "${OUT_DIR}/final_checkpoint" ]] || { echo "missing final checkpoint" >&2; exit 2; }
    env "${eval_env[@]}" "${eval_cmd[@]}" 2>&1 | tee "${EVAL_LOG}"
    "${parse_cmd[@]}"
    if [[ -n "${RESULTS_ROOT}" ]]; then
        mkdir -p "${RESULTS_ROOT}"
        cp "${EVAL_DIR}/summary.csv" "${RESULTS_ROOT}/summary.csv"
        {
            echo "label,variant,epochs,output_dir,log_dir,eval_summary"
            printf '%s,%s,%s,%s,%s,%s\n' \
                "${RUN_ID}" "${VARIANT}" "${EPOCHS}" "${OUT_DIR}" "${LOG_DIR}" "${EVAL_DIR}/summary.csv"
        } > "${RESULTS_ROOT}/manifest.csv"
    fi
fi
