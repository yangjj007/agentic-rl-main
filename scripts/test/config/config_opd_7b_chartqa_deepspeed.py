"""
Fast cross-model OPD (7B teacher + 0.5B student) — DeepSpeed default for scripts/test/.

Embedded SFT cold-start steps count toward total steps (see opsd gate in fast_profile).

Usage:
  bash scripts/test/train_opd.sh
"""
import os
import sys

_test_cfg_dir = os.path.dirname(os.path.abspath(__file__))
if _test_cfg_dir not in sys.path:
    sys.path.insert(0, _test_cfg_dir)

import config.config_opd_7b_chartqa_deepspeed as base
from fast_profile import OUTPUT_ROOT, apply_to_config

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "opd-7b-ds")

CONFIG = apply_to_config(
    base.CONFIG,
    output_dir=OUTPUT_DIR,
    enable_visual_supervision=True,
)
