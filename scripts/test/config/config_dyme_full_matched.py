"""Full DyME comparator with matched optimization and Visual Supervision."""

from __future__ import annotations

import copy
import os
import sys

from config.visual_supervision_defaults import build_visual_supervision_config


_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

import config_dyme_matched as pure


CONFIG = copy.deepcopy(pure.CONFIG)
CONFIG["opsd"]["enabled"] = False
CONFIG["opsd"]["visual_supervision"] = build_visual_supervision_config()
