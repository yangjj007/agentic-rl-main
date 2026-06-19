#!/usr/bin/env bash
# Run all three fast baselines sequentially: SFT -> DyME -> OPD.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

RL_EPOCHS="${DYME_FAST_NUM_TRAIN_EPOCHS:-4}"
SFT_EPOCHS="${DYME_FAST_SFT_EPOCHS:-4}"
EST_STEPS="${DYME_FAST_EST_STEPS_PER_EPOCH:-600}"
COLD_FRAC="${DYME_FAST_COLD_START_FRAC:-0.08}"
EST_RL_STEPS=$((RL_EPOCHS * EST_STEPS))

cold_steps="$(python - <<PY
frac = float("${COLD_FRAC}")
steps = int("${EST_RL_STEPS}")
print(max(1, int(steps * frac)) if frac > 0 and steps > 0 else 0)
PY
)"

echo "============================================================"
echo "test/ fast baselines — sequential run"
echo "  dataset: full ChartQA (train_medium_vf_full.json)"
echo "  SFT epochs: ${SFT_EPOCHS}"
echo "  RL epochs (DyME + OPD): ${RL_EPOCHS} (~${EST_RL_STEPS} steps)"
echo "  OPD cold-start: ~${cold_steps}/${EST_RL_STEPS} steps (embedded SFT, counted in total)"
echo "  outputs:"
echo "    SFT  -> test/outputs/sft/final_checkpoint"
echo "    DyME -> test/outputs/dyme/"
echo "    OPD  -> test/outputs/opd-7b-ds/"
echo "============================================================"

bash "${TEST_DIR}/train_sft.sh"
bash "${TEST_DIR}/train_dyme.sh"
bash "${TEST_DIR}/train_opd.sh"

echo "All fast baselines finished."
