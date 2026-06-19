#!/usr/bin/env bash
# Run all three fast baselines sequentially: SFT -> DyME -> OPD.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

MAX_SAMPLES="${DYME_FAST_MAX_SAMPLES:-512}"
MAX_STEPS="${DYME_FAST_MAX_STEPS:-500}"
COLD_FRAC="${DYME_FAST_COLD_START_FRAC:-0.08}"

cold_steps="$(python - <<PY
frac = float("${COLD_FRAC}")
steps = int("${MAX_STEPS}")
print(max(1, int(steps * frac)) if frac > 0 and steps > 0 else 0)
PY
)"

echo "============================================================"
echo "test/ fast baselines — sequential run"
echo "  samples: ${MAX_SAMPLES}"
echo "  RL max_steps (DyME + OPD): ${MAX_STEPS}"
echo "  OPD cold-start: ${cold_steps}/${MAX_STEPS} steps (embedded SFT, counted in total)"
echo "  outputs:"
echo "    SFT  -> outputs/test-fast/sft/final_checkpoint"
echo "    DyME -> outputs/test-fast/dyme/"
echo "    OPD  -> outputs/test-fast/opd-7b-ds/"
echo "============================================================"

bash "${TEST_DIR}/train_sft.sh"
bash "${TEST_DIR}/train_dyme.sh"
bash "${TEST_DIR}/train_opd.sh"

echo "All fast baselines finished."
