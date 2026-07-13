#!/usr/bin/env bash
# Download the ChartQA ablation student/teacher models into this project tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${DYME_MODEL_ROOT:-${ROOT}/models}"
STUDENT_REPO="${DYME_STUDENT_REPO:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
TEACHER_REPO="${DYME_TEACHER_REPO:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
DRY_RUN=0
SKIP_STUDENT=0
SKIP_TEACHER=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/download_local_models.sh [--model-root DIR] [--dry-run]
    [--student-repo HF_ID] [--teacher-repo HF_ID]
    [--skip-student] [--skip-teacher]

Defaults:
  model root : ./models
  student    : llava-hf/llava-onevision-qwen2-0.5b-ov-hf -> ./models/llava-0.5b-ov
  teacher    : llava-hf/llava-onevision-qwen2-7b-ov-hf   -> ./models/llava-7b-ov
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root) MODEL_ROOT="${2:?missing model root}"; shift 2 ;;
    --student-repo) STUDENT_REPO="${2:?missing student repo}"; shift 2 ;;
    --teacher-repo) TEACHER_REPO="${2:?missing teacher repo}"; shift 2 ;;
    --skip-student) SKIP_STUDENT=1; shift ;;
    --skip-teacher) SKIP_TEACHER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

download_one() {
  local role="$1"
  local repo="$2"
  local dir="$3"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DOWNLOAD[%s]: huggingface-cli download %q --repo-type model --local-dir %q --resume-download\n' \
      "${role}" "${repo}" "${dir}"
    return 0
  fi

  command -v huggingface-cli >/dev/null 2>&1 || {
    echo "ERROR: huggingface-cli not found. Install huggingface_hub in the active environment." >&2
    exit 1
  }
  mkdir -p "${dir}"
  huggingface-cli download "${repo}" --repo-type model --local-dir "${dir}" --resume-download
}

MODEL_ROOT="$(readlink -m "${MODEL_ROOT}")"
STUDENT_DIR="${MODEL_ROOT}/llava-0.5b-ov"
TEACHER_DIR="${MODEL_ROOT}/llava-7b-ov"

echo "model root: ${MODEL_ROOT}"
if [[ "${SKIP_STUDENT}" == "0" ]]; then
  download_one "student" "${STUDENT_REPO}" "${STUDENT_DIR}"
fi
if [[ "${SKIP_TEACHER}" == "0" ]]; then
  download_one "teacher" "${TEACHER_REPO}" "${TEACHER_DIR}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "VERIFY: bash scripts/prepare_local_models.sh ${STUDENT_DIR} ${TEACHER_DIR}"
else
  bash "${ROOT}/scripts/prepare_local_models.sh" "${STUDENT_DIR}" "${TEACHER_DIR}"
fi

echo "Use these paths for training:"
echo "  export DYME_MODEL_ROOT=${MODEL_ROOT}"
echo "  export DYME_STUDENT_MODEL=${STUDENT_DIR}"
echo "  export DYME_TEACHER_MODEL=${TEACHER_DIR}"
