"""Tests for ChartQA/A-OKVQA image path resolution."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.paths import CHARTQA_IMAGES_DIR, resolve_image_path


def test_resolve_legacy_chartqa_path(tmp_path, monkeypatch):
    import data_utils.paths as paths_mod

    img_dir = tmp_path / "data" / "images" / "chartqa" / "images"
    img_dir.mkdir(parents=True)
    img_file = img_dir / "train_002843.png"
    img_file.write_bytes(b"x")

    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(paths_mod, "CHARTQA_IMAGES_DIR", str(img_dir))
    monkeypatch.setattr(
        paths_mod,
        "CHARTQA_DIR",
        str(tmp_path / "data" / "images" / "chartqa"),
    )

    resolved = paths_mod.resolve_image_path("/chartqa_output/images/train_002843.png")
    assert resolved == str(img_file.resolve())
    assert os.path.exists(resolved)
