from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main5_campaign_model_download_excludes_onnx_artifacts() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert "snapshot_download(" in text
    assert "ignore_patterns=" in text
    assert '"onnx/*"' in text


def test_main5_campaign_model_download_uses_stable_required_file_patterns() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("HF_HUB_DISABLE_XET", "1")' in text
    assert "allow_patterns=" in text
    for pattern in ('"*.safetensors"', '"*.json"', '"*.txt"', '"video_processor/*.json"'):
        assert pattern in text


def test_main5_campaign_disables_xet_before_huggingface_import() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'HF_HUB_DISABLE_XET=1 \\' in text
    assert text.index('os.environ.setdefault("HF_HUB_DISABLE_XET", "1")') < text.index(
        "from huggingface_hub import snapshot_download"
    )


def test_main5_campaign_model_ready_checks_indexed_safetensor_shards() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert "model.safetensors.index.json" in text
    assert "weight_map" in text
    assert "missing model shard" in text


def test_main5_campaign_download_uses_single_worker_resume() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert "max_workers=1" in text
