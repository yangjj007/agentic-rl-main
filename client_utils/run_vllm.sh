#!/usr/bin/env bash
# Start one OpenAI-compatible vLLM server per GPU for DyME Qwen rewrites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${VLLM_MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
BASE_PORT="${VLLM_BASE_PORT:-23333}"
HOST="${VLLM_HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
DTYPE="${VLLM_DTYPE:-half}"
QUANTIZATION="${VLLM_QUANTIZATION:-awq}"
START_DELAY="${VLLM_START_DELAY:-2}"
LOG_DIR="${VLLM_LOG_DIR:-${PROJECT_ROOT}/outputs/vllm_qwen25_logs}"

GPU_LIST_RAW="${VLLM_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
GPU_LIST_RAW="${GPU_LIST_RAW//,/ }"
read -r -a GPUS <<<"${GPU_LIST_RAW}"

if ((${#GPUS[@]} == 0)); then
    echo "[error] no GPUs configured; set VLLM_GPU_LIST" >&2
    exit 2
fi

resolve_vllm_bin() {
    if [[ -n "${VLLM_BIN:-}" ]]; then
        printf '%s\n' "${VLLM_BIN}"
        return
    fi
    command -v vllm 2>/dev/null || true
}

VLLM_EXECUTABLE="$(resolve_vllm_bin)"
if [[ -z "${VLLM_EXECUTABLE}" || ! -x "${VLLM_EXECUTABLE}" ]]; then
    echo "[error] vllm executable not found; set VLLM_BIN=/path/to/vllm" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

pid_file_for() {
    local index="$1"
    local port=$((BASE_PORT + index))
    printf '%s/gpu%s_port%s.pid\n' "${LOG_DIR}" "${index}" "${port}"
}

log_file_for() {
    local index="$1"
    local port=$((BASE_PORT + index))
    printf '%s/gpu%s_port%s.log\n' "${LOG_DIR}" "${index}" "${port}"
}

start_servers() {
    echo "Starting ${#GPUS[@]} vLLM server instance(s)..."
    echo "Model source: ${MODEL_PATH}"
    echo "Served model: ${SERVED_MODEL_NAME}"
    echo "vLLM executable: ${VLLM_EXECUTABLE}"

    local index gpu port pid_file log_file existing_pid
    local -a vllm_args
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        port=$((BASE_PORT + index))
        pid_file="$(pid_file_for "${index}")"
        log_file="$(log_file_for "${index}")"

        existing_pid=""
        if [[ -f "${pid_file}" ]]; then
            existing_pid="$(<"${pid_file}")"
        fi
        if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            echo "--> GPU ${gpu}, port ${port}: already running (PID ${existing_pid})"
            continue
        fi

        echo "--> Launching GPU ${gpu}, port ${port}; log=${log_file}"
        vllm_args=(
            serve "${MODEL_PATH}"
            --host "${HOST}"
            --port "${port}"
            --served-model-name "${SERVED_MODEL_NAME}"
            --dtype "${DTYPE}"
            --max-model-len "${MAX_MODEL_LEN}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        )
        if [[ -n "${QUANTIZATION}" ]]; then
            vllm_args+=(--quantization "${QUANTIZATION}")
        fi

        nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${VLLM_EXECUTABLE}" "${vllm_args[@]}" \
            >"${log_file}" 2>&1 &
        echo "$!" >"${pid_file}"
        sleep "${START_DELAY}"
    done

    echo
    echo "Servers are loading in the background. Check readiness with:"
    echo "  bash client_utils/run_vllm.sh status"
    echo "All ports must be ready before running Qwen rewrite preprocessing."
}

status_servers() {
    local index gpu port pid_file pid state
    local ready=0
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        port=$((BASE_PORT + index))
        pid_file="$(pid_file_for "${index}")"
        pid=""
        [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"

        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            state="ready"
            ready=$((ready + 1))
        elif [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            state="loading"
        else
            state="stopped"
        fi
        echo "GPU ${gpu} port ${port}: ${state}${pid:+ (PID ${pid})}"
    done
    echo "Ready: ${ready}/${#GPUS[@]}"
    [[ "${ready}" -eq "${#GPUS[@]}" ]]
}

stop_servers() {
    local index port pid_file pid cmdline
    for index in "${!GPUS[@]}"; do
        port=$((BASE_PORT + index))
        pid_file="$(pid_file_for "${index}")"
        if [[ ! -f "${pid_file}" ]]; then
            echo "--> Port ${port}: no PID file"
            continue
        fi

        pid="$(<"${pid_file}")"
        if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
            echo "--> Port ${port}: stale PID file (${pid:-empty})"
            continue
        fi

        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        if [[ "${cmdline}" != *vllm* || "${cmdline}" != *serve* ]]; then
            echo "[warning] refusing to stop PID ${pid}: not a vLLM serve process" >&2
            continue
        fi

        echo "--> Stopping port ${port}, PID ${pid}"
        kill "${pid}"
        rm -f -- "${pid_file}"
    done
}

case "${1:-start}" in
    start)
        start_servers
        ;;
    status)
        status_servers
        ;;
    stop)
        stop_servers
        ;;
    *)
        echo "Usage: bash client_utils/run_vllm.sh [start|status|stop]" >&2
        exit 2
        ;;
esac
