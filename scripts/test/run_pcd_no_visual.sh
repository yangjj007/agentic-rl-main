#!/usr/bin/env bash
# No-visual PCD run: DePlot textual evidence on, Visual Supervision off,
# all-wrong teacher-probe rescue from step 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EPOCHS="${DYME_PCD_EPOCHS:-4}"
MAX_STEPS="${DYME_PCD_MAX_STEPS:-}"
RESUME_MODE="${DYME_PCD_RESUME:-none}"  # none | auto | /path/to/checkpoint-N
DRY_RUN="${DYME_PCD_DRY_RUN:-0}"
SPEED_PROFILE="${DYME_PCD_SPEED_PROFILE:-canonical}"  # canonical | fast60
VARIANT="${DYME_PCD_VARIANT:-deplot_no_vs_opd_pcd}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_pcd_no_visual.sh [EPOCHS] [--resume auto|CHECKPOINT|none] [--speed-profile canonical|fast60] [--variant NAME] [--dry-run]

Examples:
  bash scripts/test/run_pcd_no_visual.sh 4 --speed-profile canonical
  bash scripts/test/run_pcd_no_visual.sh 10 --resume auto --speed-profile canonical
  bash scripts/test/run_pcd_no_visual.sh 4 --speed-profile fast60
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_route_guard
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling
  bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format
  bash scripts/test/run_pcd_no_visual.sh 10 --variant deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision
  bash scripts/test/run_pcd_no_visual.sh 10 --variant deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair
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
if [[ -n "${MAX_STEPS}" ]]; then
  case "${MAX_STEPS}" in
    ''|*[!0-9]*)
      echo "DYME_PCD_MAX_STEPS must be a positive integer, got: ${MAX_STEPS}" >&2
      exit 2
      ;;
  esac
  if [[ "${MAX_STEPS}" -lt 1 ]]; then
    echo "DYME_PCD_MAX_STEPS must be >= 1, got: ${MAX_STEPS}" >&2
    exit 2
  fi
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
case "${VARIANT}" in
  deplot_no_vs_opd_pcd|deplot_no_vs_opd_pcd_oracle_hint|deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward|deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay|deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay|deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair|deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style|deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter|deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition|deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only|deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_diagnostic|deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate|deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision|deplot_no_vs_opd_pcd_oracle_hint_opd_no_hard_imitation_adaptive_supervision|deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision|deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair|deplot_no_vs_opd_pcd_route_guard|deplot_no_vs_opd_pcd_oracle_hint_route_guard|deplot_no_vs_opd_pcd_route_guard_perception_teacher|deplot_no_vs_opd_pcd_route_guard_perception_hint)
    ;;
  *)
    echo "Unknown PCD variant: ${VARIANT}" >&2
    usage >&2
    exit 2
    ;;
esac

RUN_ID="${DYME_PCD_RUN_ID:-pcd_no_visual_staged}"
OUT_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual/${RUN_ID}}"
LOG_ROOT="${DYME_PCD_LOG_ROOT:-outputs/test-fast/logs/pcd_no_visual_${RUN_ID}}"
OUT_DIR="${OUT_ROOT}/${VARIANT}"
LOG_DIR="${LOG_ROOT}/${VARIANT}"
STUDENT_MODEL="${DYME_STUDENT_MODEL:-/home/deepseek_VG/deepseek/models/llava-0.5b-ov}"
TEACHER_MODEL="${DYME_TEACHER_MODEL:-/home/deepseek_VG/deepseek/models/llava-7b-ov}"
if [[ "${SPEED_PROFILE}" == "fast60" ]]; then
  TEACHER_PROBE_BATCH_SIZE="${DYME_TEACHER_PROBE_BATCH_SIZE:-8}"
  TEACHER_PROBE_MAX_PER_BATCH="${DYME_TEACHER_PROBE_MAX_PER_BATCH:-16}"
  TEACHER_TRAJECTORY="${DYME_TEACHER_TRAJECTORY:-0}"
  TEACHER_PROBE_CANDIDATE_LOG="${DYME_TEACHER_PROBE_CANDIDATE_LOG:-0}"
else
  # Canonical profile mirrors run_opd_deplot_ablation.sh:deplot_no_vs_opd_pcd.
  TEACHER_PROBE_BATCH_SIZE="${DYME_TEACHER_PROBE_BATCH_SIZE:-1}"
  TEACHER_PROBE_MAX_PER_BATCH="${DYME_TEACHER_PROBE_MAX_PER_BATCH:-0}"
  TEACHER_TRAJECTORY="${DYME_TEACHER_TRAJECTORY:-1}"
  TEACHER_PROBE_CANDIDATE_LOG="${DYME_TEACHER_PROBE_CANDIDATE_LOG:-1}"
fi
TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS="${DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS:-256}"

TEACHER_PROVIDERS="format_only,visual_facts_deplot"
TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_short_answer}"
TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-96}"
TEACHER_TRAJ_MAX_NEW_TOKENS="${DYME_TEACHER_TRAJ_MAX_NEW_TOKENS:-128}"
TEACHER_TRAJ_WEIGHT_DECAY="${DYME_TEACHER_TRAJ_WEIGHT_DECAY:-0}"
TEACHER_TRAJ_DECAY_START_STEP="${DYME_TEACHER_TRAJ_DECAY_START_STEP:-294}"
TEACHER_TRAJ_DECAY_END_STEP="${DYME_TEACHER_TRAJ_DECAY_END_STEP:-441}"
TEACHER_TRAJ_DECAY_START_PROGRESS="${DYME_TEACHER_TRAJ_DECAY_START_PROGRESS:-0.25}"
TEACHER_TRAJ_DECAY_END_PROGRESS="${DYME_TEACHER_TRAJ_DECAY_END_PROGRESS:-0.50}"
TEACHER_TRAJ_FINAL_WEIGHT="${DYME_TEACHER_TRAJ_FINAL_WEIGHT:-0.0}"
ORACLE_GOLD_SUFFIX_EXPECTED=0
OPSD_SKIP_DEGENERATE="${DYME_OPSD_SKIP_DEGENERATE:-1}"
ROUTE_GUARD_ENABLED=0
PERCEPTION_REWARD=0
PERCEPTION_REWARD_SOURCE="${DYME_PERCEPTION_REWARD_SOURCE:-image_teacher}"
PERCEPTION_REWARD_WEIGHT="${DYME_PERCEPTION_REWARD_WEIGHT:-0.2}"
PERCEPTION_REWARD_BATCH_SIZE="${DYME_PERCEPTION_REWARD_BATCH_SIZE:-4}"
PERCEPTION_REWARD_MAX_NEW_TOKENS="${DYME_PERCEPTION_REWARD_MAX_NEW_TOKENS:-8}"
EVAL_FORMAT_REWARD="${DYME_EVAL_FORMAT_REWARD:-0}"
EVAL_FORMAT_REWARD_WEIGHT="${DYME_EVAL_FORMAT_REWARD_WEIGHT:-0.1}"
CHART_COT_VERIFY="${DYME_CHART_COT_VERIFY:-0}"
CHART_COT_GATE_MODE="${DYME_CHART_COT_GATE_MODE:-off}"
CHART_COT_REQUIRE_Q3="${DYME_CHART_COT_REQUIRE_Q3:-1}"
CHART_COT_LOG_SAMPLES="${DYME_CHART_COT_LOG_SAMPLES:-1}"
CHART_COT_MAX_LOG_SAMPLES="${DYME_CHART_COT_MAX_LOG_SAMPLES:-8}"
TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-opd}"
TEACHER_SFT_REPAIR_SCOPE="${DYME_TEACHER_SFT_REPAIR_SCOPE:-all_wrong}"
TEACHER_SFT_REPAIR_SLOTS="${DYME_TEACHER_SFT_REPAIR_SLOTS:-1}"
TEACHER_SFT_TARGET_MAX_TOKENS="${DYME_TEACHER_SFT_TARGET_MAX_TOKENS:-256}"
TEACHER_SFT_SANITIZE_PRIVILEGED="${DYME_TEACHER_SFT_SANITIZE_PRIVILEGED:-1}"
TEACHER_SFT_TARGET_CONSTRAINT="${DYME_TEACHER_SFT_TARGET_CONSTRAINT:-chartqa_hint}"
TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-chartqa_hint}"
VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-0}"
VISUAL_REFINER="${DYME_VISUAL_REFINER:-0}"
VISUAL_PREFETCH_IC="${DYME_VISUAL_PREFETCH_IC:-0}"
VISUAL_LOG="${DYME_VISUAL_LOG:-0}"
VISUAL_SAVE_ARTIFACTS="${DYME_VISUAL_SAVE_ARTIFACTS:-0}"
VISUAL_LOG_SAMPLES="${DYME_VISUAL_LOG_SAMPLES:-1}"
OPSD_WEIGHT_DECAY="${DYME_OPSD_WEIGHT_DECAY:-0}"
OPSD_DECAY_START_STEP="${DYME_OPSD_DECAY_START_STEP:-294}"
OPSD_DECAY_END_STEP="${DYME_OPSD_DECAY_END_STEP:-441}"
OPSD_DECAY_START_PROGRESS="${DYME_OPSD_DECAY_START_PROGRESS:-0.50}"
OPSD_DECAY_END_PROGRESS="${DYME_OPSD_DECAY_END_PROGRESS:-0.75}"
OPSD_FINAL_WEIGHT="${DYME_OPSD_FINAL_WEIGHT:-0.5}"
OPSD_MAX_PER_PROMPT_AFTER_STEP="${DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP:-0}"
OPSD_MAX_PER_PROMPT="${DYME_OPSD_MAX_PER_PROMPT:-0}"
OPSD_ROUTE_CAP_START_PROGRESS="${DYME_OPSD_ROUTE_CAP_START_PROGRESS:-0.50}"
OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-sft}"
EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-0}"
EFFECTIVE_SAMPLING_AFTER_STEP="${DYME_EFFECTIVE_SAMPLING_AFTER_STEP:-294}"
EFFECTIVE_SAMPLING_START_PROGRESS="${DYME_EFFECTIVE_SAMPLING_START_PROGRESS:-0.50}"
EFFECTIVE_SAMPLING_MIXED_WEIGHT="${DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT:-4.0}"
EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT="${DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT:-1.0}"
EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT="${DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT:-0.7}"
EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT="${DYME_EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT:-1.0}"
EFFECTIVE_SAMPLING_REWARD_STD_BONUS="${DYME_EFFECTIVE_SAMPLING_REWARD_STD_BONUS:-2.0}"
POSITIVE_REPLAY="${DYME_POSITIVE_REPLAY:-0}"
POSITIVE_REPLAY_DATASET="${DYME_POSITIVE_REPLAY_DATASET:-outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json}"
POSITIVE_REPLAY_WEIGHT="${DYME_POSITIVE_REPLAY_WEIGHT:-0.1}"
POSITIVE_REPLAY_BATCH_SIZE="${DYME_POSITIVE_REPLAY_BATCH_SIZE:-1}"
POSITIVE_REPLAY_AFTER_STEP="${DYME_POSITIVE_REPLAY_AFTER_STEP:-0}"
POSITIVE_REPLAY_UNTIL_STEP="${DYME_POSITIVE_REPLAY_UNTIL_STEP:-0}"
POSITIVE_REPLAY_MAX_ROWS="${DYME_POSITIVE_REPLAY_MAX_ROWS:-0}"
POSITIVE_REPLAY_SEED="${DYME_POSITIVE_REPLAY_SEED:-13}"
ROLLOUT_REPLAY="${DYME_ROLLOUT_REPLAY:-0}"
ROLLOUT_REPLAY_WEIGHT="${DYME_ROLLOUT_REPLAY_WEIGHT:-0.05}"
ROLLOUT_REPLAY_CAPACITY="${DYME_ROLLOUT_REPLAY_CAPACITY:-256}"
ROLLOUT_REPLAY_BATCH_SIZE="${DYME_ROLLOUT_REPLAY_BATCH_SIZE:-2}"
ROLLOUT_REPLAY_AFTER_STEP="${DYME_ROLLOUT_REPLAY_AFTER_STEP:-50}"
ROLLOUT_REPLAY_UNTIL_STEP="${DYME_ROLLOUT_REPLAY_UNTIL_STEP:-0}"
ROLLOUT_REPLAY_MAX_AGE_STEPS="${DYME_ROLLOUT_REPLAY_MAX_AGE_STEPS:-64}"
ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE="${DYME_ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE:-0.05}"
ROLLOUT_REPLAY_CORRECT_THRESHOLD="${DYME_ROLLOUT_REPLAY_CORRECT_THRESHOLD:-0.5}"
ROLLOUT_REPLAY_PRIORITY_ALPHA="${DYME_ROLLOUT_REPLAY_PRIORITY_ALPHA:-1.0}"
ROLLOUT_REPLAY_POSITIVE_ONLY="${DYME_ROLLOUT_REPLAY_POSITIVE_ONLY:-1}"
ROLLOUT_REPLAY_SEED="${DYME_ROLLOUT_REPLAY_SEED:-17}"
EFFECTIVE_GROUP_FILTER="${DYME_EFFECTIVE_GROUP_FILTER:-0}"
EFFECTIVE_GROUP_FILTER_AFTER_STEP="${DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP:-294}"
EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP="${DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP:-1}"
EFFECTIVE_GROUP_FILTER_ALL_CORRECT="${DYME_EFFECTIVE_GROUP_FILTER_ALL_CORRECT:-1}"
PHASE_SCHEDULE_MODE="${DYME_PHASE_SCHEDULE_MODE:-step}"
DYNAMIC_TRIGGER_MONITOR="${DYME_DYNAMIC_TRIGGER_MONITOR:-1}"
DYNAMIC_TRIGGER_EMA_ALPHA="${DYME_DYNAMIC_TRIGGER_EMA_ALPHA:-0.10}"
DYNAMIC_TRIGGER_MIN_PROGRESS="${DYME_DYNAMIC_TRIGGER_MIN_PROGRESS:-0.20}"
DYNAMIC_TRIGGER_PATIENCE="${DYME_DYNAMIC_TRIGGER_PATIENCE:-20}"
DYNAMIC_TRIGGER_SAMPLING_MIXED_MAX="${DYME_DYNAMIC_TRIGGER_SAMPLING_MIXED_MAX:-0.20}"
DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN="${DYME_DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN:-0.70}"
DYNAMIC_TRIGGER_RL_MIXED_MIN="${DYME_DYNAMIC_TRIGGER_RL_MIXED_MIN:-0.30}"
DYNAMIC_TRIGGER_RL_ZERO_MAX="${DYME_DYNAMIC_TRIGGER_RL_ZERO_MAX:-0.30}"
ADAPTIVE_SUPERVISION="${DYME_ADAPTIVE_SUPERVISION:-0}"
ADAPTIVE_READINESS_SOURCE="${DYME_ADAPTIVE_READINESS_SOURCE:-mixed_zero}"
ADAPTIVE_EMA_ALPHA="${DYME_ADAPTIVE_EMA_ALPHA:-0.10}"
ADAPTIVE_TARGET_READINESS="${DYME_ADAPTIVE_TARGET_READINESS:-0.20}"
ADAPTIVE_OPSD_INITIAL_WEIGHT="${DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT:-1.5}"
ADAPTIVE_OPSD_FINAL_WEIGHT="${DYME_ADAPTIVE_OPSD_FINAL_WEIGHT:-0.5}"
ADAPTIVE_TEACHER_INITIAL_WEIGHT="${DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT:-0.5}"
ADAPTIVE_TEACHER_FINAL_WEIGHT="${DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT:-0.0}"
ADAPTIVE_OPSD_INITIAL_CAP="${DYME_ADAPTIVE_OPSD_INITIAL_CAP:-8}"
ADAPTIVE_OPSD_FINAL_CAP="${DYME_ADAPTIVE_OPSD_FINAL_CAP:-2}"
GLOBAL_SIGNAL_LOGGING="${DYME_GLOBAL_SIGNAL_LOGGING:-0}"
case "${VARIANT}" in
  deplot_no_vs_opd_pcd_route_guard|deplot_no_vs_opd_pcd_oracle_hint_route_guard|deplot_no_vs_opd_pcd_route_guard_perception_teacher|deplot_no_vs_opd_pcd_route_guard_perception_hint)
    ROUTE_GUARD_ENABLED=1
    ;;
esac
case "${VARIANT}" in
  deplot_no_vs_opd_pcd_route_guard_perception_teacher)
    PERCEPTION_REWARD=1
    PERCEPTION_REWARD_SOURCE="${DYME_PERCEPTION_REWARD_SOURCE:-image_teacher}"
    ;;
  deplot_no_vs_opd_pcd_route_guard_perception_hint)
    PERCEPTION_REWARD=1
    PERCEPTION_REWARD_SOURCE="${DYME_PERCEPTION_REWARD_SOURCE:-trusted_hint}"
    ;;
esac
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_route_guard" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot,oracle_hint"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_oracle_hint}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-500}"
  TEACHER_TRAJ_MAX_NEW_TOKENS="${DYME_TEACHER_TRAJ_MAX_NEW_TOKENS:-500}"
  ORACLE_GOLD_SUFFIX_EXPECTED=1
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay" ]]; then
  EVAL_FORMAT_REWARD="${DYME_EVAL_FORMAT_REWARD:-1}"
  EVAL_FORMAT_REWARD_WEIGHT="${DYME_EVAL_FORMAT_REWARD_WEIGHT:-0.1}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" ]]; then
  EVAL_FORMAT_REWARD="${DYME_EVAL_FORMAT_REWARD:-1}"
  EVAL_FORMAT_REWARD_WEIGHT="${DYME_EVAL_FORMAT_REWARD_WEIGHT:-0.2}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay" ]]; then
  TEACHER_TRAJ_WEIGHT_DECAY="${DYME_TEACHER_TRAJ_WEIGHT_DECAY:-1}"
  TEACHER_TRAJ_DECAY_START_STEP="${DYME_TEACHER_TRAJ_DECAY_START_STEP:-294}"
  TEACHER_TRAJ_DECAY_END_STEP="${DYME_TEACHER_TRAJ_DECAY_END_STEP:-441}"
  TEACHER_TRAJ_FINAL_WEIGHT="${DYME_TEACHER_TRAJ_FINAL_WEIGHT:-0.0}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only" ]]; then
  TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-traj_sft}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style" ]]; then
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-student_short}"
elif [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-student_hint_short}"
elif [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only" ]]; then
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-answer_only}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  TEACHER_TRAJ_WEIGHT_DECAY="${DYME_TEACHER_TRAJ_WEIGHT_DECAY:-1}"
  TEACHER_TRAJ_DECAY_START_STEP="${DYME_TEACHER_TRAJ_DECAY_START_STEP:-147}"
  TEACHER_TRAJ_DECAY_END_STEP="${DYME_TEACHER_TRAJ_DECAY_END_STEP:-294}"
  TEACHER_TRAJ_FINAL_WEIGHT="${DYME_TEACHER_TRAJ_FINAL_WEIGHT:-0.0}"
  OPSD_WEIGHT_DECAY="${DYME_OPSD_WEIGHT_DECAY:-1}"
  OPSD_DECAY_START_STEP="${DYME_OPSD_DECAY_START_STEP:-294}"
  OPSD_DECAY_END_STEP="${DYME_OPSD_DECAY_END_STEP:-441}"
  OPSD_FINAL_WEIGHT="${DYME_OPSD_FINAL_WEIGHT:-0.5}"
  OPSD_MAX_PER_PROMPT_AFTER_STEP="${DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP:-294}"
  OPSD_MAX_PER_PROMPT="${DYME_OPSD_MAX_PER_PROMPT:-2}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  EFFECTIVE_SAMPLING_AFTER_STEP="${DYME_EFFECTIVE_SAMPLING_AFTER_STEP:-294}"
  EFFECTIVE_SAMPLING_MIXED_WEIGHT="${DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT:-4.0}"
  EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT="${DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT:-1.0}"
  EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT="${DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT:-0.7}"
  EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT="${DYME_EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT:-1.0}"
  EFFECTIVE_SAMPLING_REWARD_STD_BONUS="${DYME_EFFECTIVE_SAMPLING_REWARD_STD_BONUS:-2.0}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  POSITIVE_REPLAY="${DYME_POSITIVE_REPLAY:-1}"
  POSITIVE_REPLAY_WEIGHT="${DYME_POSITIVE_REPLAY_WEIGHT:-0.1}"
  POSITIVE_REPLAY_BATCH_SIZE="${DYME_POSITIVE_REPLAY_BATCH_SIZE:-1}"
  POSITIVE_REPLAY_AFTER_STEP="${DYME_POSITIVE_REPLAY_AFTER_STEP:-0}"
  POSITIVE_REPLAY_UNTIL_STEP="${DYME_POSITIVE_REPLAY_UNTIL_STEP:-0}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  ROLLOUT_REPLAY="${DYME_ROLLOUT_REPLAY:-1}"
  ROLLOUT_REPLAY_WEIGHT="${DYME_ROLLOUT_REPLAY_WEIGHT:-0.05}"
  ROLLOUT_REPLAY_CAPACITY="${DYME_ROLLOUT_REPLAY_CAPACITY:-256}"
  ROLLOUT_REPLAY_BATCH_SIZE="${DYME_ROLLOUT_REPLAY_BATCH_SIZE:-2}"
  ROLLOUT_REPLAY_AFTER_STEP="${DYME_ROLLOUT_REPLAY_AFTER_STEP:-50}"
  ROLLOUT_REPLAY_UNTIL_STEP="${DYME_ROLLOUT_REPLAY_UNTIL_STEP:-0}"
  ROLLOUT_REPLAY_MAX_AGE_STEPS="${DYME_ROLLOUT_REPLAY_MAX_AGE_STEPS:-64}"
  ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE="${DYME_ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE:-0.05}"
  ROLLOUT_REPLAY_CORRECT_THRESHOLD="${DYME_ROLLOUT_REPLAY_CORRECT_THRESHOLD:-0.5}"
  ROLLOUT_REPLAY_PRIORITY_ALPHA="${DYME_ROLLOUT_REPLAY_PRIORITY_ALPHA:-1.0}"
  ROLLOUT_REPLAY_POSITIVE_ONLY="${DYME_ROLLOUT_REPLAY_POSITIVE_ONLY:-1}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  EFFECTIVE_GROUP_FILTER="${DYME_EFFECTIVE_GROUP_FILTER:-1}"
  EFFECTIVE_GROUP_FILTER_AFTER_STEP="${DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP:-294}"
  EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP="${DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP:-1}"
  EFFECTIVE_GROUP_FILTER_ALL_CORRECT="${DYME_EFFECTIVE_GROUP_FILTER_ALL_CORRECT:-1}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition" ]]; then
  # The rl_transition variant is intended for the second stage after replay SFT
  # warmup. Keep static positive replay out of the DyME phase so RL/rollout
  # signals can take over and to avoid the extra CE forward memory cost.
  POSITIVE_REPLAY="${DYME_POSITIVE_REPLAY:-0}"
  POSITIVE_REPLAY_WEIGHT="${DYME_POSITIVE_REPLAY_WEIGHT:-0.0}"
  POSITIVE_REPLAY_BATCH_SIZE="${DYME_POSITIVE_REPLAY_BATCH_SIZE:-0}"
  POSITIVE_REPLAY_UNTIL_STEP="${DYME_POSITIVE_REPLAY_UNTIL_STEP:-0}"
  OPSD_FINAL_WEIGHT="${DYME_OPSD_FINAL_WEIGHT:-0.0}"
  OPSD_MAX_PER_PROMPT="${DYME_OPSD_MAX_PER_PROMPT:-1}"
  EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP="${DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP:-0}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot,oracle_hint"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_oracle_hint}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-500}"
  TEACHER_TRAJ_MAX_NEW_TOKENS="${DYME_TEACHER_TRAJ_MAX_NEW_TOKENS:-500}"
  ORACLE_GOLD_SUFFIX_EXPECTED=1
  TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-traj_sft}"
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-student_hint_short}"
  TEACHER_TRAJ_WEIGHT_DECAY="${DYME_TEACHER_TRAJ_WEIGHT_DECAY:-1}"
  OPSD_WEIGHT_DECAY="${DYME_OPSD_WEIGHT_DECAY:-1}"
  OPSD_FINAL_WEIGHT="${DYME_OPSD_FINAL_WEIGHT:-0.5}"
  OPSD_MAX_PER_PROMPT="${DYME_OPSD_MAX_PER_PROMPT:-2}"
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  PHASE_SCHEDULE_MODE="${DYME_PHASE_SCHEDULE_MODE:-progress}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  EVAL_FORMAT_REWARD="${DYME_EVAL_FORMAT_REWARD:-0}"
  POSITIVE_REPLAY="${DYME_POSITIVE_REPLAY:-0}"
  ROLLOUT_REPLAY="${DYME_ROLLOUT_REPLAY:-0}"
  DYNAMIC_TRIGGER_MONITOR="${DYME_DYNAMIC_TRIGGER_MONITOR:-1}"
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_diagnostic" || "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot,oracle_hint"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_oracle_hint}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-500}"
  TEACHER_TRAJ_MAX_NEW_TOKENS="${DYME_TEACHER_TRAJ_MAX_NEW_TOKENS:-500}"
  ORACLE_GOLD_SUFFIX_EXPECTED=1
  TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-traj_sft}"
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-chartqa_hint}"
  TEACHER_TRAJ_WEIGHT_DECAY="${DYME_TEACHER_TRAJ_WEIGHT_DECAY:-1}"
  OPSD_WEIGHT_DECAY="${DYME_OPSD_WEIGHT_DECAY:-1}"
  OPSD_FINAL_WEIGHT="${DYME_OPSD_FINAL_WEIGHT:-0.5}"
  OPSD_MAX_PER_PROMPT="${DYME_OPSD_MAX_PER_PROMPT:-2}"
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  PHASE_SCHEDULE_MODE="${DYME_PHASE_SCHEDULE_MODE:-progress}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  EVAL_FORMAT_REWARD="${DYME_EVAL_FORMAT_REWARD:-0}"
  POSITIVE_REPLAY="${DYME_POSITIVE_REPLAY:-0}"
  ROLLOUT_REPLAY="${DYME_ROLLOUT_REPLAY:-0}"
  DYNAMIC_TRIGGER_MONITOR="${DYME_DYNAMIC_TRIGGER_MONITOR:-1}"
  CHART_COT_VERIFY="${DYME_CHART_COT_VERIFY:-1}"
  if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate" ]]; then
    CHART_COT_GATE_MODE="${DYME_CHART_COT_GATE_MODE:-gate}"
  else
    CHART_COT_GATE_MODE="${DYME_CHART_COT_GATE_MODE:-diagnostic}"
  fi
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot,oracle_hint"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_oracle_hint}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-500}"
  TEACHER_TRAJ_MAX_NEW_TOKENS="${DYME_TEACHER_TRAJ_MAX_NEW_TOKENS:-500}"
  ORACLE_GOLD_SUFFIX_EXPECTED=1
  TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-traj_sft}"
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-chartqa_hint}"
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  OPSD_WEIGHT_DECAY=0
  TEACHER_TRAJ_WEIGHT_DECAY=0
  OPSD_MAX_PER_PROMPT=0
  DYNAMIC_TRIGGER_MONITOR=0
  EVAL_FORMAT_REWARD=0
  CHART_COT_VERIFY=1
  CHART_COT_GATE_MODE=gate
  ADAPTIVE_SUPERVISION=1
  ADAPTIVE_READINESS_SOURCE=global_grpo_route
  ADAPTIVE_TARGET_READINESS=0.30
  GLOBAL_SIGNAL_LOGGING=1
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_oracle_hint_opd_no_hard_imitation_adaptive_supervision" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot,oracle_hint"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_oracle_hint}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-500}"
  ORACLE_GOLD_SUFFIX_EXPECTED=1
  TEACHER_TRAJECTORY=0
  TEACHER_CORRECT_REPAIR_MODE=opd
  OPSD_SKIP_DEGENERATE=0
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  OPSD_WEIGHT_DECAY=0
  TEACHER_TRAJ_WEIGHT_DECAY=0
  OPSD_MAX_PER_PROMPT=0
  DYNAMIC_TRIGGER_MONITOR=0
  EVAL_FORMAT_REWARD=0
  CHART_COT_VERIFY=1
  CHART_COT_GATE_MODE=gate
  ADAPTIVE_SUPERVISION=1
  ADAPTIVE_READINESS_SOURCE=global_grpo_route
  ADAPTIVE_TARGET_READINESS=0.30
  ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.0
  ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0
  GLOBAL_SIGNAL_LOGGING=1
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision" ||
      "${VARIANT}" == "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair" ]]; then
  TEACHER_PROVIDERS="format_only,visual_facts_deplot"
  TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_short_answer}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-96}"
  ORACLE_GOLD_SUFFIX_EXPECTED=0
  TEACHER_TRAJECTORY=0
  TEACHER_CORRECT_REPAIR_MODE=opd
  OPSD_SKIP_DEGENERATE=0
  EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
  EFFECTIVE_SAMPLING_MIXED_WEIGHT="${DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT:-6.0}"
  OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
  OPSD_WEIGHT_DECAY=0
  TEACHER_TRAJ_WEIGHT_DECAY=0
  OPSD_MAX_PER_PROMPT=0
  DYNAMIC_TRIGGER_MONITOR=0
  EVAL_FORMAT_REWARD=0
  CHART_COT_VERIFY=1
  CHART_COT_GATE_MODE=gate
  ADAPTIVE_SUPERVISION=1
  ADAPTIVE_READINESS_SOURCE=global_grpo_route
  ADAPTIVE_TARGET_READINESS=0.15
  ADAPTIVE_OPSD_INITIAL_WEIGHT="${DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT:-1.0}"
  ADAPTIVE_OPSD_FINAL_WEIGHT="${DYME_ADAPTIVE_OPSD_FINAL_WEIGHT:-0.25}"
  ADAPTIVE_OPSD_INITIAL_CAP="${DYME_ADAPTIVE_OPSD_INITIAL_CAP:-4}"
  ADAPTIVE_OPSD_FINAL_CAP="${DYME_ADAPTIVE_OPSD_FINAL_CAP:-1}"
  ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.0
  ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0
  GLOBAL_SIGNAL_LOGGING=1
fi
if [[ "${VARIANT}" == "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair" ]]; then
  TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-refiner_sft}"
  TEACHER_SFT_REPAIR_SCOPE="${DYME_TEACHER_SFT_REPAIR_SCOPE:-all_wrong}"
  TEACHER_SFT_REPAIR_SLOTS="${DYME_TEACHER_SFT_REPAIR_SLOTS:-4}"
  TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-answer_only}"
  TEACHER_SFT_TARGET_MAX_TOKENS="${DYME_TEACHER_SFT_TARGET_MAX_TOKENS:-64}"
  TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-320}"
  TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS="${DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS:-1024}"
  ROUTE_GUARD_ENABLED=1
  ADAPTIVE_OPSD_INITIAL_CAP="${DYME_ADAPTIVE_OPSD_INITIAL_CAP:-2}"
  VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-0}"
  VISUAL_REFINER="${DYME_VISUAL_REFINER:-1}"
  VISUAL_PREFETCH_IC="${DYME_VISUAL_PREFETCH_IC:-1}"
  VISUAL_LOG="${DYME_VISUAL_LOG:-1}"
  VISUAL_SAVE_ARTIFACTS="${DYME_VISUAL_SAVE_ARTIFACTS:-0}"
  VISUAL_LOG_SAMPLES="${DYME_VISUAL_LOG_SAMPLES:-1}"
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
if [[ -n "${MAX_STEPS}" ]]; then
  echo "max steps override: ${MAX_STEPS}"
else
  echo "max steps override: <none>"
fi
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
echo "teacher providers: ${TEACHER_PROVIDERS}"
echo "teacher prompt profile: ${TEACHER_PROBE_PROMPT_PROFILE}"
echo "teacher probe max new tokens: ${TEACHER_PROBE_MAX_NEW_TOKENS}"
echo "teacher trajectory max new tokens: ${TEACHER_TRAJ_MAX_NEW_TOKENS}"
echo "phase schedule: mode=${PHASE_SCHEDULE_MODE} teacher_traj=${TEACHER_TRAJ_DECAY_START_PROGRESS}->${TEACHER_TRAJ_DECAY_END_PROGRESS} effective_sampling=${EFFECTIVE_SAMPLING_START_PROGRESS} route_cap=${OPSD_ROUTE_CAP_START_PROGRESS} opd_decay=${OPSD_DECAY_START_PROGRESS}->${OPSD_DECAY_END_PROGRESS}"
echo "teacher trajectory weight decay: enabled=${TEACHER_TRAJ_WEIGHT_DECAY} start=${TEACHER_TRAJ_DECAY_START_STEP} end=${TEACHER_TRAJ_DECAY_END_STEP} final=${TEACHER_TRAJ_FINAL_WEIGHT}"
echo "OPD weight decay/cap: decay=${OPSD_WEIGHT_DECAY} start=${OPSD_DECAY_START_STEP} end=${OPSD_DECAY_END_STEP} final=${OPSD_FINAL_WEIGHT} cap_after=${OPSD_MAX_PER_PROMPT_AFTER_STEP} max_per_prompt=${OPSD_MAX_PER_PROMPT}"
echo "effective sampling: enabled=${EFFECTIVE_SAMPLING} after=${EFFECTIVE_SAMPLING_AFTER_STEP} mixed=${EFFECTIVE_SAMPLING_MIXED_WEIGHT} all_wrong=${EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT} all_correct=${EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT} unknown=${EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT} std_bonus=${EFFECTIVE_SAMPLING_REWARD_STD_BONUS}"
echo "positive replay: enabled=${POSITIVE_REPLAY} dataset=${POSITIVE_REPLAY_DATASET} weight=${POSITIVE_REPLAY_WEIGHT} batch_size=${POSITIVE_REPLAY_BATCH_SIZE} after=${POSITIVE_REPLAY_AFTER_STEP} until=${POSITIVE_REPLAY_UNTIL_STEP} max_rows=${POSITIVE_REPLAY_MAX_ROWS}"
echo "rollout replay: enabled=${ROLLOUT_REPLAY} weight=${ROLLOUT_REPLAY_WEIGHT} capacity=${ROLLOUT_REPLAY_CAPACITY} batch_size=${ROLLOUT_REPLAY_BATCH_SIZE} after=${ROLLOUT_REPLAY_AFTER_STEP} until=${ROLLOUT_REPLAY_UNTIL_STEP} max_age=${ROLLOUT_REPLAY_MAX_AGE_STEPS} min_abs_adv=${ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE} priority_alpha=${ROLLOUT_REPLAY_PRIORITY_ALPHA} positive_only=${ROLLOUT_REPLAY_POSITIVE_ONLY}"
echo "effective group filter: enabled=${EFFECTIVE_GROUP_FILTER} after=${EFFECTIVE_GROUP_FILTER_AFTER_STEP} all_wrong_keep=${EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP} all_correct=${EFFECTIVE_GROUP_FILTER_ALL_CORRECT}"
echo "oracle gold suffix expected: ${ORACLE_GOLD_SUFFIX_EXPECTED}"
echo "route guard enabled: ${ROUTE_GUARD_ENABLED}"
echo "teacher correct repair: mode=${TEACHER_CORRECT_REPAIR_MODE} scope=${TEACHER_SFT_REPAIR_SCOPE} slots=${TEACHER_SFT_REPAIR_SLOTS} target_max_tokens=${TEACHER_SFT_TARGET_MAX_TOKENS} sanitize_privileged=${TEACHER_SFT_SANITIZE_PRIVILEGED} target_constraint=${TEACHER_SFT_TARGET_CONSTRAINT} target_style=${TEACHER_SFT_TARGET_STYLE}"
echo "perception reward: enabled=${PERCEPTION_REWARD} source=${PERCEPTION_REWARD_SOURCE} weight=${PERCEPTION_REWARD_WEIGHT} batch_size=${PERCEPTION_REWARD_BATCH_SIZE} max_new_tokens=${PERCEPTION_REWARD_MAX_NEW_TOKENS}"
echo "eval-format reward: enabled=${EVAL_FORMAT_REWARD} weight=${EVAL_FORMAT_REWARD_WEIGHT}"
echo "Chart CoT quality: enabled=${CHART_COT_VERIFY} mode=${CHART_COT_GATE_MODE} require_q3=${CHART_COT_REQUIRE_Q3} log_samples=${CHART_COT_LOG_SAMPLES} max_log_samples=${CHART_COT_MAX_LOG_SAMPLES}"
echo "============================================================"

TRAIN_ENV=(
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
  "DYME_OPSD_PROVIDERS=${TEACHER_PROVIDERS}"
  "DYME_TEACHER_PROBE_PROVIDERS=${TEACHER_PROVIDERS}"
  "DYME_TEACHER_PROBE=1"
  "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0"
  "DYME_TEACHER_PROBE_BATCH_SIZE=${TEACHER_PROBE_BATCH_SIZE}"
  "DYME_TEACHER_PROBE_MAX_PER_BATCH=${TEACHER_PROBE_MAX_PER_BATCH}"
  "DYME_TEACHER_PROBE_PROMPT_PROFILE=${TEACHER_PROBE_PROMPT_PROFILE}"
  "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=${TEACHER_PROBE_MAX_NEW_TOKENS}"
  "DYME_TEACHER_PROBE_CANDIDATE_LOG=${TEACHER_PROBE_CANDIDATE_LOG}"
  "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=${TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS}"
  "DYME_TEACHER_TRAJECTORY=${TEACHER_TRAJECTORY}"
  "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=${TEACHER_TRAJ_MAX_NEW_TOKENS}"
  "DYME_TEACHER_TRAJ_WEIGHT_DECAY=${TEACHER_TRAJ_WEIGHT_DECAY}"
  "DYME_TEACHER_TRAJ_DECAY_START_STEP=${TEACHER_TRAJ_DECAY_START_STEP}"
  "DYME_TEACHER_TRAJ_DECAY_END_STEP=${TEACHER_TRAJ_DECAY_END_STEP}"
  "DYME_TEACHER_TRAJ_DECAY_START_PROGRESS=${TEACHER_TRAJ_DECAY_START_PROGRESS}"
  "DYME_TEACHER_TRAJ_DECAY_END_PROGRESS=${TEACHER_TRAJ_DECAY_END_PROGRESS}"
  "DYME_TEACHER_TRAJ_FINAL_WEIGHT=${TEACHER_TRAJ_FINAL_WEIGHT}"
  "DYME_OPSD_LOSS_TYPE=jsd"
  "DYME_OPSD_SKIP_DEGENERATE=${OPSD_SKIP_DEGENERATE}"
  "DYME_OPSD_WEIGHT=1.5"
  "DYME_OPSD_WEIGHT_DECAY=${OPSD_WEIGHT_DECAY}"
  "DYME_OPSD_DECAY_START_STEP=${OPSD_DECAY_START_STEP}"
  "DYME_OPSD_DECAY_END_STEP=${OPSD_DECAY_END_STEP}"
  "DYME_OPSD_DECAY_START_PROGRESS=${OPSD_DECAY_START_PROGRESS}"
  "DYME_OPSD_DECAY_END_PROGRESS=${OPSD_DECAY_END_PROGRESS}"
  "DYME_OPSD_FINAL_WEIGHT=${OPSD_FINAL_WEIGHT}"
  "DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP=${OPSD_MAX_PER_PROMPT_AFTER_STEP}"
  "DYME_OPSD_MAX_PER_PROMPT=${OPSD_MAX_PER_PROMPT}"
  "DYME_OPSD_ROUTE_CAP_START_PROGRESS=${OPSD_ROUTE_CAP_START_PROGRESS}"
  "DYME_OPSD_OVERFLOW_ROUTE=${OPSD_OVERFLOW_ROUTE}"
  "DYME_EFFECTIVE_SAMPLING=${EFFECTIVE_SAMPLING}"
  "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=${EFFECTIVE_SAMPLING_AFTER_STEP}"
  "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=${EFFECTIVE_SAMPLING_START_PROGRESS}"
  "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=${EFFECTIVE_SAMPLING_MIXED_WEIGHT}"
  "DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT=${EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT}"
  "DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT=${EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT}"
  "DYME_EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT=${EFFECTIVE_SAMPLING_UNKNOWN_WEIGHT}"
  "DYME_EFFECTIVE_SAMPLING_REWARD_STD_BONUS=${EFFECTIVE_SAMPLING_REWARD_STD_BONUS}"
  "DYME_POSITIVE_REPLAY=${POSITIVE_REPLAY}"
  "DYME_POSITIVE_REPLAY_DATASET=${POSITIVE_REPLAY_DATASET}"
  "DYME_POSITIVE_REPLAY_WEIGHT=${POSITIVE_REPLAY_WEIGHT}"
  "DYME_POSITIVE_REPLAY_BATCH_SIZE=${POSITIVE_REPLAY_BATCH_SIZE}"
  "DYME_POSITIVE_REPLAY_AFTER_STEP=${POSITIVE_REPLAY_AFTER_STEP}"
  "DYME_POSITIVE_REPLAY_UNTIL_STEP=${POSITIVE_REPLAY_UNTIL_STEP}"
  "DYME_POSITIVE_REPLAY_MAX_ROWS=${POSITIVE_REPLAY_MAX_ROWS}"
  "DYME_POSITIVE_REPLAY_SEED=${POSITIVE_REPLAY_SEED}"
  "DYME_ROLLOUT_REPLAY=${ROLLOUT_REPLAY}"
  "DYME_ROLLOUT_REPLAY_WEIGHT=${ROLLOUT_REPLAY_WEIGHT}"
  "DYME_ROLLOUT_REPLAY_CAPACITY=${ROLLOUT_REPLAY_CAPACITY}"
  "DYME_ROLLOUT_REPLAY_BATCH_SIZE=${ROLLOUT_REPLAY_BATCH_SIZE}"
  "DYME_ROLLOUT_REPLAY_AFTER_STEP=${ROLLOUT_REPLAY_AFTER_STEP}"
  "DYME_ROLLOUT_REPLAY_UNTIL_STEP=${ROLLOUT_REPLAY_UNTIL_STEP}"
  "DYME_ROLLOUT_REPLAY_MAX_AGE_STEPS=${ROLLOUT_REPLAY_MAX_AGE_STEPS}"
  "DYME_ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE=${ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE}"
  "DYME_ROLLOUT_REPLAY_CORRECT_THRESHOLD=${ROLLOUT_REPLAY_CORRECT_THRESHOLD}"
  "DYME_ROLLOUT_REPLAY_PRIORITY_ALPHA=${ROLLOUT_REPLAY_PRIORITY_ALPHA}"
  "DYME_ROLLOUT_REPLAY_POSITIVE_ONLY=${ROLLOUT_REPLAY_POSITIVE_ONLY}"
  "DYME_ROLLOUT_REPLAY_SEED=${ROLLOUT_REPLAY_SEED}"
  "DYME_EFFECTIVE_GROUP_FILTER=${EFFECTIVE_GROUP_FILTER}"
  "DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP=${EFFECTIVE_GROUP_FILTER_AFTER_STEP}"
  "DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP=${EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP}"
  "DYME_EFFECTIVE_GROUP_FILTER_ALL_CORRECT=${EFFECTIVE_GROUP_FILTER_ALL_CORRECT}"
  "DYME_PHASE_SCHEDULE_MODE=${PHASE_SCHEDULE_MODE}"
  "DYME_DYNAMIC_TRIGGER_MONITOR=${DYNAMIC_TRIGGER_MONITOR}"
  "DYME_DYNAMIC_TRIGGER_EMA_ALPHA=${DYNAMIC_TRIGGER_EMA_ALPHA}"
  "DYME_DYNAMIC_TRIGGER_MIN_PROGRESS=${DYNAMIC_TRIGGER_MIN_PROGRESS}"
  "DYME_DYNAMIC_TRIGGER_PATIENCE=${DYNAMIC_TRIGGER_PATIENCE}"
  "DYME_DYNAMIC_TRIGGER_SAMPLING_MIXED_MAX=${DYNAMIC_TRIGGER_SAMPLING_MIXED_MAX}"
  "DYME_DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN=${DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN}"
  "DYME_DYNAMIC_TRIGGER_RL_MIXED_MIN=${DYNAMIC_TRIGGER_RL_MIXED_MIN}"
  "DYME_DYNAMIC_TRIGGER_RL_ZERO_MAX=${DYNAMIC_TRIGGER_RL_ZERO_MAX}"
  "DYME_ADAPTIVE_SUPERVISION=${ADAPTIVE_SUPERVISION}"
  "DYME_ADAPTIVE_READINESS_SOURCE=${ADAPTIVE_READINESS_SOURCE}"
  "DYME_ADAPTIVE_EMA_ALPHA=${ADAPTIVE_EMA_ALPHA}"
  "DYME_ADAPTIVE_TARGET_READINESS=${ADAPTIVE_TARGET_READINESS}"
  "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=${ADAPTIVE_OPSD_INITIAL_WEIGHT}"
  "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=${ADAPTIVE_OPSD_FINAL_WEIGHT}"
  "DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT=${ADAPTIVE_TEACHER_INITIAL_WEIGHT}"
  "DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT=${ADAPTIVE_TEACHER_FINAL_WEIGHT}"
  "DYME_ADAPTIVE_OPSD_INITIAL_CAP=${ADAPTIVE_OPSD_INITIAL_CAP}"
  "DYME_ADAPTIVE_OPSD_FINAL_CAP=${ADAPTIVE_OPSD_FINAL_CAP}"
  "DYME_GLOBAL_SIGNAL_LOGGING=${GLOBAL_SIGNAL_LOGGING}"
  "DYME_OPSD_VARIANCE_ADAPTIVE=0"
  "DYME_OPSD_ADAPTIVE_STD_TARGET=0.25"
  "DYME_OPSD_ADAPTIVE_MAX_MULT=2.0"
  "DYME_GRPO_WEIGHT=1.0"
  "DYME_OPSD_SRKL_ALPHA=0.1"
  "DYME_VISUAL_CHECKER=${VISUAL_CHECKER}"
  "DYME_VISUAL_REFINER=${VISUAL_REFINER}"
  "DYME_VISUAL_PREFETCH_IC=${VISUAL_PREFETCH_IC}"
  "DYME_VISUAL_LOG=${VISUAL_LOG}"
  "DYME_VISUAL_SAVE_ARTIFACTS=${VISUAL_SAVE_ARTIFACTS}"
  "DYME_VISUAL_LOG_SAMPLES=${VISUAL_LOG_SAMPLES}"
  "DYME_DEPLOT_ENABLED=0"
  "DYME_OPSD_HANG_DEBUG=0"
  "DYME_OPSD_HANG_FORCE=0"
  "DYME_OPSD_DETAIL_EVERY=0"
  "TRANSFORMERS_OFFLINE=1"
  "HF_HUB_OFFLINE=1"
  "WANDB_MODE=disabled"
  "DYME_EVAL_FORMAT_REWARD=${EVAL_FORMAT_REWARD}"
  "DYME_EVAL_FORMAT_REWARD_WEIGHT=${EVAL_FORMAT_REWARD_WEIGHT}"
  "DYME_CHART_COT_VERIFY=${CHART_COT_VERIFY}"
  "DYME_CHART_COT_GATE_MODE=${CHART_COT_GATE_MODE}"
  "DYME_CHART_COT_REQUIRE_Q3=${CHART_COT_REQUIRE_Q3}"
  "DYME_CHART_COT_LOG_SAMPLES=${CHART_COT_LOG_SAMPLES}"
  "DYME_CHART_COT_MAX_LOG_SAMPLES=${CHART_COT_MAX_LOG_SAMPLES}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_ENV+=("DYME_TRAIN_MAX_STEPS=${MAX_STEPS}")
else
  TRAIN_ENV=("-u" "DYME_MAX_STEPS" "-u" "DYME_TRAIN_MAX_STEPS" "${TRAIN_ENV[@]}")
fi
if [[ "${ROUTE_GUARD_ENABLED}" == "1" ]]; then
  TRAIN_ENV+=(
    "DYME_SIGNAL_AWARE_ROUTING=1"
    "DYME_SIGNAL_REWARD_STD_MIN=${DYME_SIGNAL_REWARD_STD_MIN:-0.05}"
    "DYME_DEGENERATE_HARD_OVERRIDE=1"
    "DYME_CLIPPED_HARD_OVERRIDE=1"
    "DYME_PERCEPTION_REWARD=${PERCEPTION_REWARD}"
    "DYME_PERCEPTION_REWARD_SOURCE=${PERCEPTION_REWARD_SOURCE}"
    "DYME_PERCEPTION_REWARD_WEIGHT=${PERCEPTION_REWARD_WEIGHT}"
    "DYME_PERCEPTION_REWARD_BATCH_SIZE=${PERCEPTION_REWARD_BATCH_SIZE}"
    "DYME_PERCEPTION_REWARD_MAX_NEW_TOKENS=${PERCEPTION_REWARD_MAX_NEW_TOKENS}"
  )
fi
if [[ "${TEACHER_CORRECT_REPAIR_MODE}" != "opd" ]]; then
  TRAIN_ENV+=(
    "DYME_TEACHER_CORRECT_REPAIR_MODE=${TEACHER_CORRECT_REPAIR_MODE}"
    "DYME_TEACHER_SFT_REPAIR_SCOPE=${TEACHER_SFT_REPAIR_SCOPE}"
    "DYME_TEACHER_SFT_REPAIR_SLOTS=${TEACHER_SFT_REPAIR_SLOTS}"
    "DYME_TEACHER_SFT_TARGET_MAX_TOKENS=${TEACHER_SFT_TARGET_MAX_TOKENS}"
    "DYME_TEACHER_SFT_SANITIZE_PRIVILEGED=${TEACHER_SFT_SANITIZE_PRIVILEGED}"
    "DYME_TEACHER_SFT_TARGET_CONSTRAINT=${TEACHER_SFT_TARGET_CONSTRAINT}"
    "DYME_TEACHER_SFT_TARGET_STYLE=${TEACHER_SFT_TARGET_STYLE}"
  )
fi
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
