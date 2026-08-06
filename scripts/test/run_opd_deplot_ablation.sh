#!/usr/bin/env bash
# Run DePlot OPD ablations.
#
# Default mode is dry-run so this script cannot accidentally disturb active GPU
# jobs. Use --smoke for a tiny max_steps validation, or --run for full epoch-budget
# sequential training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODE="dry-run"
SMOKE_STEPS="${DYME_DEPLOT_ABLATION_SMOKE_STEPS:-2}"
RUN_ID="${DYME_DEPLOT_ABLATION_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${DYME_DEPLOT_ABLATION_EPOCHS:-4}"
VARIANTS="deplot_vs_opd,deplot_vs_srkl,deplot_no_vs_opd,deplot_no_vs_opd_va,deplot_no_vs_opd_pcd,deplot_no_vs_opd_va_pcd"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_opd_deplot_ablation.sh [--dry-run|--smoke|--run] [--smoke-steps N] [--run-id ID] [--epochs N] [--variants CSV]

Variants:
  deplot_vs_opd     : visual_facts_deplot on, visual_supervision on,  OPD loss=jsd, GRPO enabled
  deplot_vs_srkl    : visual_facts_deplot on, visual_supervision on,  OPD loss=srkl
  deplot_no_vs_opd  : visual_facts_deplot on, visual_supervision off, OPD loss=jsd, GRPO enabled
  deplot_no_vs_opd_va     : no-VS OPD + variance-adaptive OPD weight
  deplot_no_vs_opd_pcd    : no-VS OPD + all-wrong teacher-probe rescue from step 0
  deplot_no_vs_opd_va_pcd : no-VS OPD + both variance-adaptive weight and all-wrong rescue

Examples:
  bash scripts/test/run_opd_deplot_ablation.sh --dry-run
  bash scripts/test/run_opd_deplot_ablation.sh --smoke --smoke-steps 2
  bash scripts/test/run_opd_deplot_ablation.sh --run
  bash scripts/test/run_opd_deplot_ablation.sh --run --run-id deplot_4epoch_va_pcd --variants deplot_no_vs_opd_va,deplot_no_vs_opd_pcd,deplot_no_vs_opd_va_pcd
  bash scripts/test/run_opd_deplot_ablation.sh --run --epochs 10 --run-id deplot_10epoch_anchor --variants deplot_no_vs_opd
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --smoke)
      MODE="smoke"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --smoke-steps)
      SMOKE_STEPS="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --variants)
      VARIANTS="$2"
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

case "${SMOKE_STEPS}" in
  ''|*[!0-9]*)
    echo "--smoke-steps must be a positive integer, got: ${SMOKE_STEPS}" >&2
    exit 2
    ;;
esac
if [[ "${SMOKE_STEPS}" -lt 2 ]]; then
  echo "--smoke-steps must be >= 2, got: ${SMOKE_STEPS}" >&2
  exit 2
fi
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

LOG_ROOT="${DYME_DEPLOT_ABLATION_LOG_ROOT:-outputs/test-fast/logs/opd_deplot_ablation_${RUN_ID}}"
OUT_ROOT="${DYME_DEPLOT_ABLATION_OUTPUT_ROOT:-outputs/test-fast/opd-deplot-ablation/${RUN_ID}}"
STUDENT_MODEL="${DYME_STUDENT_MODEL:-models/llava-0.5b-ov}"
TEACHER_MODEL="${DYME_TEACHER_MODEL:-models/llava-7b-ov}"

print_gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "============================================================"
    echo "GPU snapshot before DePlot ablation"
    if ! nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; then
      echo "WARN: nvidia-smi GPU snapshot unavailable in this shell."
    fi
    echo "Active compute apps:"
    if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; then
      echo "WARN: nvidia-smi compute-app snapshot unavailable in this shell."
    fi
    echo "============================================================"
  fi
}

variant_visual_flag() {
  case "$1" in
    deplot_no_vs_opd|deplot_no_vs_opd_va|deplot_no_vs_opd_pcd|deplot_no_vs_opd_va_pcd) echo "0" ;;
    deplot_vs_opd|deplot_vs_srkl) echo "1" ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

variant_loss_type() {
  case "$1" in
    deplot_no_vs_opd|deplot_no_vs_opd_va|deplot_no_vs_opd_pcd|deplot_no_vs_opd_va_pcd|deplot_vs_opd) echo "jsd" ;;
    deplot_vs_srkl) echo "srkl" ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

variant_adaptive_flag() {
  case "$1" in
    deplot_no_vs_opd_va|deplot_no_vs_opd_va_pcd) echo "1" ;;
    deplot_no_vs_opd|deplot_no_vs_opd_pcd|deplot_vs_opd|deplot_vs_srkl) echo "0" ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

variant_probe_all_wrong_after_step() {
  case "$1" in
    deplot_no_vs_opd_pcd|deplot_no_vs_opd_va_pcd) echo "0" ;;
    deplot_no_vs_opd|deplot_no_vs_opd_va|deplot_vs_opd|deplot_vs_srkl) echo "" ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

print_variant_command() {
  local variant="$1"
  local visual_flag="$2"
  local loss_type="$3"
  local adaptive_flag="$4"
  local probe_all_wrong_after_step="$5"
  local out_dir="$6"
  local log_dir="$7"
  local max_steps_line=""
  local probe_all_wrong_line=""

  if [[ "${MODE}" == "smoke" ]]; then
    max_steps_line=$'DYME_TRAIN_MAX_STEPS='"${SMOKE_STEPS}"$' \\\nDYME_MAX_STEPS='"${SMOKE_STEPS}"$' \\'
  fi
  if [[ -n "${probe_all_wrong_after_step}" ]]; then
    probe_all_wrong_line=$'DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP='"${probe_all_wrong_after_step}"$' \\\n'
  fi

  cat <<CMD
${max_steps_line}
DYME_NUM_TRAIN_EPOCHS=${EPOCHS} \\
DYME_FAST_NUM_TRAIN_EPOCHS=${EPOCHS} \\
DYME_STUDENT_MODEL=${STUDENT_MODEL} \\
DYME_TEACHER_MODEL=${TEACHER_MODEL} \\
DYME_OUTPUT_DIR=${out_dir} \\
DYME_LOG_DIR=${log_dir} \\
DYME_OPSD_PRIVILEGE_PROFILE=text \\
DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot \\
DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot \\
DYME_TEACHER_PROBE=1 \\
DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96 \\
DYME_TEACHER_PROBE_BATCH_SIZE=1 \\
DYME_TEACHER_PROBE_MAX_PER_BATCH=0 \\
${probe_all_wrong_line}DYME_TEACHER_PROBE_CANDIDATE_LOG=1 \\
DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=256 \\
DYME_TEACHER_TRAJECTORY=1 \\
DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128 \\
DYME_OPSD_LOSS_TYPE=${loss_type} \\
DYME_OPSD_WEIGHT=1.5 \\
DYME_OPSD_VARIANCE_ADAPTIVE=${adaptive_flag} \\
DYME_OPSD_ADAPTIVE_STD_TARGET=0.25 \\
DYME_OPSD_ADAPTIVE_MAX_MULT=2.0 \\
DYME_GRPO_WEIGHT=1.0 \\
DYME_OPSD_SRKL_ALPHA=0.1 \\
DYME_VISUAL_CHECKER=${visual_flag} \\
DYME_VISUAL_REFINER=${visual_flag} \\
DYME_VISUAL_PREFETCH_IC=${visual_flag} \\
DYME_VISUAL_LOG=${visual_flag} \\
DYME_VISUAL_SAVE_ARTIFACTS=0 \\
DYME_VISUAL_LOG_SAMPLES=1 \\
DYME_DEPLOT_ENABLED=0 \\
DYME_OPSD_HANG_DEBUG=0 \\
DYME_OPSD_HANG_FORCE=0 \\
DYME_OPSD_DETAIL_EVERY=0 \\
TRANSFORMERS_OFFLINE=1 \\
HF_HUB_OFFLINE=1 \\
WANDB_MODE=disabled \\
bash scripts/train_opd_7b_dyme_probe.sh --no_opsd_probe_on_generate --no_opsd_probe_first_token_logits --opsd_detail_every 0
CMD
}

check_deplot_data() {
  "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

path = Path("data/chartqa/train_medium_vf_full.json")
data = json.loads(path.read_text(encoding="utf-8"))
real = placeholder = missing = 0
for row in data:
    text = str(row.get("visual_fact_deplot") or "")
    if not text.strip():
        missing += 1
    elif "deplot_placeholder" in text:
        placeholder += 1
    elif "google/deplot" in text or '"source": "deplot"' in text or "'source': 'deplot'" in text:
        real += 1
print(f"[DyME-DATA-CHECK] path={path} n={len(data)} deplot_real={real} placeholder={placeholder} missing={missing}")
if placeholder or missing or real == 0:
    raise SystemExit("DePlot data check failed: expected real DePlot evidence and no placeholders/missing values")
PY
}

check_log_has_required_metrics() {
  local log_file="$1"
  local variant="$2"
  "${PYTHON_BIN}" - "$log_file" "$variant" <<'PY'
import ast
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
variant = sys.argv[2]
text = log_path.read_text(encoding="utf-8", errors="replace")
if "OPSD-HANGDBG" in text:
    raise SystemExit(f"{variant}: OPSD-HANGDBG leaked into {log_path}")
for marker in ("[DyME-RUN-CONFIG]", "[DyME-DATA]"):
    if marker not in text:
        raise SystemExit(f"{variant}: missing startup marker {marker} in {log_path}")

rows = []
for line in text.splitlines():
    match = re.search(r"\{.*\}", line)
    if not match:
        continue
    try:
        row = ast.literal_eval(match.group())
    except (SyntaxError, ValueError):
        continue
    if isinstance(row, dict) and any(k.startswith(("routing/", "teacher_probe/", "visual/", "loss/", "signal/")) for k in row):
        rows.append(row)

required = [
    "routing/teacher_probe_candidate_rate",
    "routing/teacher_probe_correct_rate",
    "routing/teacher_probe_wrong_rate",
    "routing/teacher_probe_skipped_no_evidence_rate",
    "routing/teacher_probe_deplot_real_rate",
    "routing/teacher_probe_visual_fact_used_rate",
    "teacher_probe/generated_tokens_mean",
    "teacher_probe/generated_tokens_p95",
    "teacher_probe/clipped_rate",
    "loss/opsd",
    "signal/reward_std_mean",
    "loss/opsd_effective_weight",
    "loss/opsd_adaptive_multiplier",
]
missing = [key for key in required if not any(key in row for row in rows)]
if missing:
    raise SystemExit(f"{variant}: missing required metrics in {log_path}: {missing}")

visual_disabled = variant.startswith("deplot_no_vs_opd")
if visual_disabled and any("visual/ic_ok_rate" in row for row in rows):
    raise SystemExit(f"{variant}: visual supervision metrics should be absent when visual_supervision is disabled")
if not visual_disabled and not any("visual/ic_ok_rate" in row for row in rows):
    raise SystemExit(f"{variant}: expected visual supervision metrics when visual_supervision is enabled")

adaptive_multiplier = [
    float(row["loss/opsd_adaptive_multiplier"])
    for row in rows
    if row.get("loss/opsd_adaptive_multiplier") is not None
]
if "_va" in variant and not any(abs(value - 1.0) > 1e-6 for value in adaptive_multiplier):
    raise SystemExit(f"{variant}: expected non-default adaptive multiplier in {log_path}")

print(f"### {variant} smoke metric check: {log_path}")
print(f"metric_rows={len(rows)}")
for key in required:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if values:
        print(f"{key}: last={values[-1]:.4f}")
PY
}

run_variant() {
  local variant="$1"
  local visual_flag
  local loss_type
  local adaptive_flag
  local probe_all_wrong_after_step
  visual_flag="$(variant_visual_flag "${variant}")"
  loss_type="$(variant_loss_type "${variant}")"
  adaptive_flag="$(variant_adaptive_flag "${variant}")"
  probe_all_wrong_after_step="$(variant_probe_all_wrong_after_step "${variant}")"

  local out_dir="${OUT_ROOT}/${variant}"
  local log_dir="${LOG_ROOT}/${variant}"
  if [[ "${MODE}" == "smoke" ]]; then
    out_dir="${OUT_ROOT}/smoke/${variant}"
    log_dir="${LOG_ROOT}/smoke/${variant}"
  fi

  echo ""
  echo "============================================================"
  echo "Variant: ${variant}"
  echo "mode: ${MODE}"
  echo "visual_supervision: ${visual_flag}"
  echo "loss_type: ${loss_type}"
  echo "DYME_OPSD_VARIANCE_ADAPTIVE: ${adaptive_flag}"
  if [[ -n "${probe_all_wrong_after_step}" ]]; then
    echo "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP: ${probe_all_wrong_after_step}"
  else
    echo "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP: <unset>"
  fi
  echo "teacher_probe_max_new_tokens: 96"
  echo "teacher_traj_max_new_tokens: 128"
  echo "output dir: ${out_dir}"
  echo "log dir: ${log_dir}"
  echo "============================================================"

  if [[ "${MODE}" == "dry-run" ]]; then
    print_variant_command "${variant}" "${visual_flag}" "${loss_type}" "${adaptive_flag}" "${probe_all_wrong_after_step}" "${out_dir}" "${log_dir}"
    return 0
  fi

  mkdir -p "${log_dir}" "${out_dir}"
  local -a train_env=(
    "DYME_NUM_TRAIN_EPOCHS=${EPOCHS}"
    "DYME_FAST_NUM_TRAIN_EPOCHS=${EPOCHS}"
    "DYME_STUDENT_MODEL=${STUDENT_MODEL}"
    "DYME_TEACHER_MODEL=${TEACHER_MODEL}"
    "DYME_OUTPUT_DIR=${out_dir}"
    "DYME_LOG_DIR=${log_dir}"
    "DYME_OPSD_PRIVILEGE_PROFILE=text"
    "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot"
    "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot"
    "DYME_TEACHER_PROBE=1"
    "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96"
    "DYME_TEACHER_PROBE_BATCH_SIZE=1"
    "DYME_TEACHER_PROBE_MAX_PER_BATCH=0"
    "DYME_TEACHER_PROBE_CANDIDATE_LOG=1"
    "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=256"
    "DYME_TEACHER_TRAJECTORY=1"
    "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128"
    "DYME_OPSD_LOSS_TYPE=${loss_type}"
    "DYME_OPSD_WEIGHT=1.5"
    "DYME_OPSD_VARIANCE_ADAPTIVE=${adaptive_flag}"
    "DYME_OPSD_ADAPTIVE_STD_TARGET=0.25"
    "DYME_OPSD_ADAPTIVE_MAX_MULT=2.0"
    "DYME_GRPO_WEIGHT=1.0"
    "DYME_OPSD_SRKL_ALPHA=0.1"
    "DYME_VISUAL_CHECKER=${visual_flag}"
    "DYME_VISUAL_REFINER=${visual_flag}"
    "DYME_VISUAL_PREFETCH_IC=${visual_flag}"
    "DYME_VISUAL_LOG=${visual_flag}"
    "DYME_VISUAL_SAVE_ARTIFACTS=0"
    "DYME_VISUAL_LOG_SAMPLES=1"
    "DYME_DEPLOT_ENABLED=0"
    "DYME_OPSD_HANG_DEBUG=0"
    "DYME_OPSD_HANG_FORCE=0"
    "DYME_OPSD_DETAIL_EVERY=0"
    "TRANSFORMERS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "WANDB_MODE=disabled"
  )
  if [[ -n "${probe_all_wrong_after_step}" ]]; then
    train_env+=("DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=${probe_all_wrong_after_step}")
  fi
  if [[ "${MODE}" == "smoke" ]]; then
    train_env=("DYME_TRAIN_MAX_STEPS=${SMOKE_STEPS}" "DYME_MAX_STEPS=${SMOKE_STEPS}" "${train_env[@]}")
    env -u DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP "${train_env[@]}" \
      bash scripts/train_opd_7b_dyme_probe.sh --no_opsd_probe_on_generate --no_opsd_probe_first_token_logits --opsd_detail_every 0
  else
    env -u DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP "${train_env[@]}" \
      bash scripts/train_opd_7b_dyme_probe.sh --no_opsd_probe_on_generate --no_opsd_probe_first_token_logits --opsd_detail_every 0
  fi

  local latest_log
  latest_log="$(ls -t "${log_dir}"/train_opd_7b_dyme_probe_*.log | head -1)"
  check_log_has_required_metrics "${latest_log}" "${variant}"
}

echo "${EPOCHS}epoch DePlot OPD ablation runner"
echo "mode: ${MODE}"
echo "run id: ${RUN_ID}"
echo "variants: ${VARIANTS}"
echo "log root: ${LOG_ROOT}"
echo "output root: ${OUT_ROOT}"
if [[ "${MODE}" == "smoke" ]]; then
  echo "smoke steps: ${SMOKE_STEPS}"
fi
check_deplot_data
print_gpu_snapshot

IFS=',' read -ra _VARIANT_LIST <<< "${VARIANTS}"
for variant in "${_VARIANT_LIST[@]}"; do
  variant="${variant// /}"
  [[ -n "${variant}" ]] || continue
  run_variant "${variant}"
done

if [[ "${MODE}" == "dry-run" ]]; then
  echo ""
  echo "Dry-run only. Use --smoke for an isolated tiny run, or --run when GPUs are free."
elif [[ "${MODE}" == "smoke" ]]; then
  echo ""
  echo "Smoke finished. Logs: ${LOG_ROOT}/smoke"
else
  echo ""
  echo "Full ${EPOCHS}epoch ablation finished. Logs: ${LOG_ROOT}"
fi
