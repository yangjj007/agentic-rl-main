import json
import os
import sys
import tempfile

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.privileged_schema import (
    heuristic_bbox_from_visual_fact,
    normalize_evidence_bbox,
    parse_visual_fact,
    resolve_crop_bbox,
)
from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged import build_privileged_context, maybe_save_privileged_images
from opsd_utils.privileged.image_utils import crop_image, load_rgb, resolve_teacher_images
from opsd_utils.privileged.profiles import effective_profile


def _make_image(path: str, size=(100, 100), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def test_text_provider():
    sample = {"hint": "Rep=67", "answer": "Answer: 131"}
    suffix, images = build_privileged_context(sample, ["text"], privileged_profile="text")
    assert "Rep=67" in suffix
    assert "131" in suffix
    assert images == []


def test_hybrid_provider_suffix():
    img = Image.new("RGB", (32, 32))
    sample = {"hint": "step", "visual_fact": "bar=3", "answer": "Answer: 3", "image": img}
    suffix, images = build_privileged_context(
        sample,
        privileged_profile="hybrid",
        opsd_config={"privileged_image": {"mode": "dual"}},
    )
    assert "Visual Facts" in suffix
    assert "Reference" in suffix
    assert len(images) == 2


def test_hybrid_default_single_image_for_chartqa():
    img = Image.new("RGB", (32, 32))
    sample = {"hint": "step", "visual_fact": "bar=3", "answer": "Answer: 3", "image": img}
    suffix, images = build_privileged_context(sample, privileged_profile="hybrid")
    assert "Visual Facts" in suffix
    assert "Reference" in suffix
    assert len(images) == 1


def test_visual_profile_excludes_answer():
    img = Image.new("RGB", (32, 32))
    sample = {"hint": "secret", "visual_fact": '{"objects":[]}', "answer": "Answer: 3", "image": img}
    suffix, _ = build_privileged_context(sample, privileged_profile="visual")
    assert "Visual Facts" in suffix
    assert "Reference Answer" not in suffix


def test_math_lm_downgrade():
    sample = {"hint": "step", "answer": "Answer: 1"}
    profile = effective_profile(sample, "hybrid")
    assert profile == "text"


def test_normalize_evidence_bbox_c2():
    assert normalize_evidence_bbox([0.1, 0.2, 0.8, 0.9]) == [0.1, 0.2, 0.8, 0.9]
    assert normalize_evidence_bbox([0.1, 0.2, 1.5, 0.9]) is None


def test_heuristic_bbox_d2():
    vf = json.dumps({"objects": [{"name": "cat", "position": "center"}]})
    bbox = heuristic_bbox_from_visual_fact(vf)
    assert bbox == [0.25, 0.25, 0.75, 0.75]


def test_crop_image_normalized_bbox():
    img = Image.new("RGB", (100, 100), (0, 255, 0))
    crop, strategy = crop_image(img, bbox_norm=[0.2, 0.2, 0.8, 0.8], strategy="bbox")
    assert strategy == "bbox"
    assert crop.size[0] > 0


def test_resolve_teacher_images_dual():
    img = Image.new("RGB", (80, 80), (0, 0, 255))
    sample = {
        "image": img,
        "visual_fact": json.dumps({"objects": [{"position": "top"}]}),
    }
    images, meta = resolve_teacher_images(sample, "hybrid", crop_cfg={"mode": "dual"})
    assert len(images) == 2
    assert meta["num_teacher_images"] == 2
    assert meta["crop_strategy"] in ("heuristic", "center", "center_fallback", "bbox")


def test_chartqa_enriched_visual_fact_hint():
    """Enriched ChartQA records (F1+F2) should activate VisualFactsProvider."""
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "hint": "Goal: Find the lowest value.\nObservation: values are 70, 72, 77.",
        "answer": "Answer: 70",
        "visual_fact_hint": "Goal: Find the lowest value.\nObservation: values are 70, 72, 77.",
        "visual_fact": "Goal: Find the lowest value.\nObservation: values are 70, 72, 77.",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | Value\n2019 | 70\n2020 | 72"
        ),
        "image": Image.new("RGB", (64, 64)),
    }
    suffix, images = build_privileged_context(
        sample,
        ["text", "visual_facts"],
        privileged_profile="hybrid",
    )
    assert "Visual Facts - Hint" in suffix
    assert "Visual Facts - DePlot" in suffix
    assert "2019 | 70" in suffix
    assert "Reference Reasoning" in suffix
    assert len(images) == 1
    vf_raw = sample.get("visual_fact") or sample.get("visual_facts")
    assert vf_raw and len(vf_raw.strip()) > 0


def test_visual_facts_f1_f2_merge():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "visual_fact_hint": "hint table",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Col | Val\nA | 1"
        ),
        "image": Image.new("RGB", (32, 32)),
    }
    suffix, _ = build_privileged_context(sample, privileged_profile="hybrid")
    assert "Visual Facts - Hint" in suffix
    assert "Visual Facts - DePlot" in suffix
    assert "Col | Val" in suffix


def test_parse_visual_fact_b1():
    raw = {"objects": [{"name": "a"}]}
    text = parse_visual_fact(raw)
    assert "objects" in text


def test_debug_artifacts_respect_detail_every():
    opsd_debug.configure(enabled=True, detail_every=10, rank=0, world_size=1)
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.new("RGB", (32, 32))
        path = maybe_save_privileged_images(5, 0, img, img, meta={"crop_strategy": "center"}, output_dir=tmp)
        assert path is None
        assert not os.path.exists(os.path.join(tmp, "logs", "images"))

        path = maybe_save_privileged_images(10, 0, img, img, meta={"crop_strategy": "center"}, output_dir=tmp)
        assert path is not None
        assert os.path.exists(f"{path}_full.png")
        assert os.path.exists(f"{path}_meta.json")


if __name__ == "__main__":
    test_text_provider()
    test_hybrid_provider_suffix()
    test_math_lm_downgrade()
    test_debug_artifacts_respect_detail_every()
    print("Privileged provider tests passed.")
