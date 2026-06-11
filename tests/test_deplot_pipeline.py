import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.chart.deplot_pipeline import (
    build_deplot_visual_fact,
    enrich_entries_with_deplot,
    format_deplot_for_teacher,
    has_real_deplot,
    is_deplot_placeholder,
    load_deplot_cache,
    placeholder_deplot_table,
    save_deplot_cache,
)
from opsd_utils.privileged.providers import VisualFactsProvider


def test_is_deplot_placeholder():
    ph = placeholder_deplot_table({"question": "q"})
    assert is_deplot_placeholder(ph)
    real = build_deplot_visual_fact({"question": "q"}, "A | 1\nB | 2")
    assert not is_deplot_placeholder(real)
    assert has_real_deplot(real)


def test_format_deplot_for_teacher():
    real = build_deplot_visual_fact({"question": "q"}, "Year | Value\n2020 | 10")
    assert format_deplot_for_teacher(real) == "Year | Value\n2020 | 10"
    assert format_deplot_for_teacher(placeholder_deplot_table({"question": "q"})) == ""
    assert format_deplot_for_teacher(None) == ""
    assert format_deplot_for_teacher("") == ""


def test_visual_facts_provider_skips_placeholder_and_missing():
    provider = VisualFactsProvider()
    sample_ph = {
        "visual_fact_deplot": placeholder_deplot_table({"question": "q"}),
        "visual_fact_hint": "hint text",
    }
    suffix = provider.build_teacher_suffix(sample_ph)
    assert "Visual Facts - Hint" in suffix
    assert "Visual Facts - DePlot" not in suffix

    sample_none = {"visual_fact_hint": "only hint"}
    suffix2 = provider.build_teacher_suffix(sample_none)
    assert "Visual Facts - DePlot" not in suffix2
    assert "Visual Facts - Hint" in suffix2


def test_visual_facts_provider_real_deplot_table():
    provider = VisualFactsProvider()
    table = "Category | 2019 | 2020\nA | 1 | 2"
    sample = {
        "visual_fact_deplot": build_deplot_visual_fact({"question": "q"}, table),
    }
    suffix = provider.build_teacher_suffix(sample)
    assert "Visual Facts - DePlot" in suffix
    assert table in suffix
    assert '"parsed_table"' not in suffix


def test_deplot_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        save_deplot_cache(path, {"/a.png": "table text"})
        loaded = load_deplot_cache(path)
        assert loaded["/a.png"] == "table text"


def test_enrich_disabled_uses_placeholder():
    entries = [{"question": "q1", "image": "missing.png"}]
    stats = enrich_entries_with_deplot(entries, enabled=False)
    assert stats["placeholder"] == 1
    assert is_deplot_placeholder(entries[0]["visual_fact_deplot"])


def test_enrich_with_mock_runner():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "chart.png")
        Image.new("RGB", (32, 32)).save(img_path)
        cache_path = os.path.join(tmp, "deplot_cache.json")
        entries = [{"question": "What?", "image": img_path}]

        class _FakeRunner:
            def load(self):
                return True

            def generate_batch_with_oom_retry(self, paths, batch_size=8):
                return ["Col | Val\nA | 1" for _ in paths]

        with patch("data_utils.chart.deplot_pipeline.DePlotRunner", return_value=_FakeRunner()):
            stats = enrich_entries_with_deplot(
                entries,
                enabled=True,
                cache_path=cache_path,
            )
        assert stats["real"] == 1
        assert has_real_deplot(entries[0]["visual_fact_deplot"])
        assert "Col | Val" in format_deplot_for_teacher(entries[0]["visual_fact_deplot"])
        assert os.path.isfile(cache_path)


def test_build_script_disabled(tmp_path):
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    inp.write_text(json.dumps([{"question": "q", "hint": "h"}]), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            os.path.join(root, "scripts", "build_visual_facts_chartqa_deplot.py"),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--no-enabled",
        ],
        check=True,
        cwd=root,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert is_deplot_placeholder(data[0]["visual_fact_deplot"])
