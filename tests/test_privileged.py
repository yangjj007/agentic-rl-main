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
from opsd_utils.privileged.providers import split_teacher_response_prefix
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
    sample = {
        "hint": "secret",
        "visual_fact_hint": "Goal: leak\nReasoning: use the answer.\nAnswer: 3",
        "visual_fact": '{"objects":[]}',
        "answer": "Answer: 3",
        "image": img,
    }
    suffix, _ = build_privileged_context(sample, privileged_profile="visual")
    assert "Visual Facts" in suffix
    assert "Goal: leak" not in suffix
    assert "Reference Answer" not in suffix


def test_format_only_chartqa_short_answer_profile():
    suffix, images = build_privileged_context(
        {"prompt": "What is the value?"},
        ["format_only"],
        privileged_profile="hybrid",
        opsd_config={
            "teacher_probe": {
                "prompt_profile": "chartqa_short_answer",
            },
        },
    )

    assert "Answer: <short answer>" in suffix
    assert images == []


def test_format_only_chartqa_deplot_reasoned_profile_is_structured_and_gold_hidden():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    secret_hint = "Goal: secret dataset hint\nAnswer: 70"
    sample = {
        "prompt": "What is the lowest value?",
        "hint": secret_hint,
        "answer": "Answer: 70",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | Value\n2019 | 70\n2020 | 72"
        ),
        "image": Image.new("RGB", (32, 32)),
    }

    raw_suffix, _ = build_privileged_context(
        sample,
        ["visual_facts_deplot", "format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_deplot_reasoned"},
        },
    )

    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)
    heading_positions = [suffix.index(f"{heading}:") for heading in (
        "Goal", "Observation", "Reasoning", "Conclusion", "Answer"
    )]
    assert heading_positions == sorted(heading_positions)
    assert "Do not count column headers or series names as data categories" in suffix
    assert "align each row label with the correct series column" in suffix
    assert "show the exact arithmetic" in suffix
    assert "Each of the first four sections must contain exactly one sentence of at most 25 words" in suffix
    assert "The final non-empty line must be exactly: Answer: <single short answer>" in suffix
    assert "[Visual Facts - DePlot]" in suffix
    assert "[Verified Hint]" not in suffix
    assert "[Reference Answer]" not in suffix
    assert secret_hint not in suffix
    assert response_prefix.startswith("Goal:")
    assert response_prefix.rstrip().endswith("Observation:")
    assert "70" not in response_prefix


def test_chartqa_visual_reasoned_prompt_profile_is_gold_hidden_and_image_native():
    sample = {
        "question": "What is the difference between A and B?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": "Year | A | B\n2020 | 70 | 50",
        "image": Image.new("RGB", (64, 64)),
    }

    suffix, images = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_visual_reasoned"},
        },
    )

    assert len(images) == 1
    assert "full chart image" in suffix.lower()
    assert "question type" in suffix.lower()
    assert "Answer: <single short answer>" in suffix
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix
    assert "Visual Facts - DePlot" not in suffix
    assert "70 | 50" not in suffix


def test_chartqa_visual_chain_prompt_exports_response_prefix_without_gold_or_deplot():
    sample = {
        "question": "What is the difference between A and B?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": "Year | A | B\n2020 | 70 | 50",
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_visual_chain_of_charts"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "full chart image" in suffix.lower()
    assert "Visual Evidence:" in suffix
    assert "Computation:" in suffix
    assert response_prefix == "Task:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix
    assert "Visual Facts - DePlot" not in suffix
    assert "70 | 50" not in suffix


def test_chartqa_visual_answer_prefix_profile_forces_short_answer_continuation():
    sample = {
        "question": "What is the difference between A and B?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": "Year | A | B\n2020 | 70 | 50",
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_visual_answer_prefix"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "Return only the final answer text" in suffix
    assert "Do not include units for numeric answers" in suffix
    assert response_prefix == "Answer:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix
    assert "Visual Facts - DePlot" not in suffix


def test_chartqa_visual_answer_prefix_numeric_profile_adds_numeric_surface_rules():
    sample = {
        "question": "How many years are above 30 percent?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": "Year | A\n2020 | 70",
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_visual_answer_prefix_numeric"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "Use Arabic numerals for counts and numeric answers" in suffix
    assert "include a percent sign" in suffix
    assert "Do not include units after numbers except %" in suffix
    assert response_prefix == "Answer:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix
    assert "Visual Facts - DePlot" not in suffix


def test_chartqa_deplot_answer_prefix_profile_keeps_deplot_auxiliary_and_gold_hidden():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "question": "What is the difference between A and B?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | A | B\n2020 | 70 | 50"
        ),
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["visual_facts_deplot", "format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_deplot_answer_prefix"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "Visual Facts - DePlot" in suffix
    assert "70 | 50" in suffix
    assert "fallible OCR" in suffix
    assert "Return only the final answer text" in suffix
    assert response_prefix == "Answer:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix


def test_chartqa_visual_operation_answer_prefix_profile_is_gold_hidden():
    sample = {
        "question": "What is the sum of the bars above 200?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": "Category | Value\nA | 707\nB | 216",
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_visual_operation_answer_prefix"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "operation" in suffix.lower()
    assert "operand" in suffix.lower()
    assert "count only" in suffix.lower()
    assert "Return only the final answer text" in suffix
    assert response_prefix == "Answer:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix
    assert "Visual Facts - DePlot" not in suffix
    assert "707" not in suffix


def test_chartqa_deplot_operation_answer_prefix_profile_uses_table_as_fallible_evidence():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "question": "What is the sum of the bars above 200?",
        "hint": "SECRET_HINT",
        "answer": "SECRET_ANSWER",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Category | Value\nA | 707\nB | 216\nC | 104"
        ),
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, images = build_privileged_context(
        sample,
        ["visual_facts_deplot", "format_only"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_deplot_operation_answer_prefix"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert len(images) == 1
    assert "Visual Facts - DePlot" in suffix
    assert "707" in suffix
    assert "fallible OCR" in suffix
    assert "row/column orientation" in suffix
    assert "ignore headers" in suffix.lower()
    assert "perform the requested arithmetic" in suffix
    assert response_prefix == "Answer:"
    assert "SECRET_HINT" not in suffix
    assert "SECRET_ANSWER" not in suffix


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
    """Enriched ChartQA records must not expose hint-derived visual fields."""
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    hint_cot = "Goal: Find the lowest value.\nObservation: values are 70, 72, 77.\nAnswer: 70"
    sample = {
        "hint": hint_cot,
        "answer": "Answer: 70",
        "visual_fact_hint": hint_cot,
        "visual_fact": None,
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
    assert "Visual Facts - Hint" not in suffix
    assert "Visual Facts - DePlot" in suffix
    assert "2019 | 70" in suffix
    assert "Reference Reasoning" in suffix
    assert "[Visual Facts - Hint]\nGoal" not in suffix
    assert len(images) == 1
    vf_raw = sample.get("visual_fact") or sample.get("visual_facts")
    assert vf_raw is None


def test_visual_facts_f1_f2_merge():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "visual_fact_hint": "Goal: hint table\nAnswer: 1",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Col | Val\nA | 1"
        ),
        "image": Image.new("RGB", (32, 32)),
    }
    suffix, _ = build_privileged_context(sample, privileged_profile="hybrid")
    assert "Visual Facts - Hint" not in suffix
    assert "Visual Facts - DePlot" in suffix
    assert "Col | Val" in suffix
    assert "Goal: hint table" not in suffix


def test_deplot_only_provider_skips_hint():
    """Deplot-only must not inject hint/CoT (F1) into teacher suffix."""
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    hint_cot = "Goal: Find the lowest value.\nObservation: values are 70, 72, 77."
    sample = {
        "hint": hint_cot,
        "answer": "Answer: 70",
        "visual_fact_hint": hint_cot,
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | Value\n2019 | 70\n2020 | 72"
        ),
        "image": Image.new("RGB", (64, 64)),
    }
    suffix, _ = build_privileged_context(
        sample,
        ["format_only", "visual_facts_deplot"],
        privileged_profile="text",
        opsd_config={"text_include_gold": False},
    )
    assert "Visual Facts - DePlot" in suffix
    assert "2019 | 70" in suffix
    assert "Visual Facts - Hint" not in suffix
    assert "Reference Reasoning" not in suffix
    assert "Reference Answer" not in suffix
    assert hint_cot not in suffix


def test_oracle_hint_provider_exports_response_prefix_and_preserves_evidence_order():
    """Official oracle-hint keeps DePlot as evidence and pre-fills hint structure."""
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    hint_cot = (
        "Goal: Find the difference in percentage of male voters who voted for Clinton and Trump.\n"
        "Observation: Hillary Clinton is 41%, and Donald Trump is 53%.\n"
        "Reasoning: Subtracting Clinton from Trump gives 53 - 41 = 12.\n"
        "Conclusion: The difference in percentage of male voters who voted for Clinton and Trump is 13%."
    )
    sample = {
        "hint": hint_cot,
        "answer": "Answer: 13",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Candidate | Male\nClinton | 41\nTrump | 53"
        ),
        "image": Image.new("RGB", (64, 64)),
    }

    raw_suffix, _ = build_privileged_context(
        sample,
        ["format_only", "visual_facts_deplot", "oracle_hint"],
        privileged_profile="text",
        opsd_config={
            "text_include_gold": False,
            "teacher_probe": {"prompt_profile": "chartqa_oracle_hint"},
        },
    )
    suffix, response_prefix = split_teacher_response_prefix(raw_suffix)

    assert "[Visual Facts - DePlot]" in suffix
    assert suffix.index("[Visual Facts - DePlot]") < suffix.index("[Verified Hint]")
    assert "Candidate | Male" in suffix
    assert "[Verified Hint]" in suffix
    assert suffix.index("[Verified Hint]") < suffix.index("[Reference Answer]")
    assert "[Reference Answer]\n13" in suffix
    assert "Do not output a short answer only." in suffix
    assert "Do not transcribe the DePlot table" in suffix
    assert "[Teacher Response Prefix]" not in suffix
    assert "The final non-empty line must be exactly:\nAnswer: 13" in suffix
    assert response_prefix.startswith("Goal: Find the difference in percentage")
    assert "\nObservation: Hillary Clinton is 41%, and Donald Trump is 53%." in response_prefix
    assert "\nReasoning: Subtracting Clinton from Trump gives 53 - 41 = 12." in response_prefix
    assert "\nConclusion: The difference in percentage" in response_prefix
    assert response_prefix.rstrip().endswith("Answer:")
    assert "Answer: 13" not in response_prefix


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
