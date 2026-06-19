"""
Fast cross-model OPD (7B teacher + 0.5B student) — DeepSpeed default for test/.

Embedded SFT cold-start steps count toward total steps (see fast_profile gate).

Usage (from test/):
  bash train_opd.sh
"""
import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import config.config_opd_7b_chartqa_deepspeed as base
from fast_profile import OUTPUT_ROOT, apply_to_config

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "opd-7b-ds")

CONFIG = apply_to_config(
    base.CONFIG,
    output_dir=OUTPUT_DIR,
)
