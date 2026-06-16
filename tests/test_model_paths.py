"""Tests for local model path resolution."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.paths import resolve_model_path, validate_local_model_dir


def test_hub_id_unchanged():
    assert resolve_model_path("llava-hf/llava-onevision-qwen2-0.5b-ov-hf") == (
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    )


def test_tilde_expanded_when_dir_exists(tmp_path, monkeypatch):
    model = tmp_path / "models" / "llava"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(model) if p.startswith("~") else p)
    resolved = resolve_model_path("~/models/llava")
    assert resolved == str(model.resolve())


def test_validate_local_requires_weights(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        validate_local_model_dir(str(empty), role="student")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "student" in str(exc)
