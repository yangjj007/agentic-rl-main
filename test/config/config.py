"""
Fast DyME baseline (pure RL, no OPSD) on full ChartQA with fewer epochs.

Usage:
  bash test/train_dyme.sh
  accelerate launch main.py --config test/config/config.py --mode rl
"""
import os
import sys

_test_cfg_dir = os.path.dirname(os.path.abspath(__file__))
if _test_cfg_dir not in sys.path:
    sys.path.insert(0, _test_cfg_dir)

import config.config as base
from fast_profile import OUTPUT_ROOT, apply_to_config

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "dyme")

CONFIG = apply_to_config(
    base.CONFIG,
    output_dir=OUTPUT_DIR,
    opsd_enabled=False,
)
