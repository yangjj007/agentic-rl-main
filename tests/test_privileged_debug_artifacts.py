import json
import os
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged.debug_artifacts import maybe_save_privileged_images


def test_max_samples_per_detail():
    opsd_debug.configure(enabled=True, detail_every=1, rank=0, world_size=1)
    cfg = {"save_images": True, "image_subdir": "logs/images", "max_samples_per_detail": 1}
    img = Image.new("RGB", (16, 16))
    with tempfile.TemporaryDirectory() as tmp:
        p0 = maybe_save_privileged_images(1, 0, img, None, meta={}, output_dir=tmp, privileged_debug_cfg=cfg)
        p1 = maybe_save_privileged_images(1, 1, img, None, meta={}, output_dir=tmp, privileged_debug_cfg=cfg)
        assert p0 is not None
        assert p1 is None


def test_meta_sidecar():
    opsd_debug.configure(enabled=True, detail_every=1, rank=0, world_size=1)
    img = Image.new("RGB", (16, 16))
    with tempfile.TemporaryDirectory() as tmp:
        prefix = maybe_save_privileged_images(
            1,
            0,
            img,
            img,
            meta={"privileged_profile": "hybrid", "crop_strategy": "bbox"},
            output_dir=tmp,
        )
        with open(f"{prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["privileged_profile"] == "hybrid"
        assert meta["crop_strategy"] == "bbox"
