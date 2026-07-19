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


def test_main5_campaign_defaults_to_refiner_sft_repair_variant_only() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    repair_variant = (
        '  "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair"'
    )
    base_variant = '  "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision"'

    assert repair_variant in text
    assert base_variant not in text


def test_main5_campaign_training_active_uses_exact_proc_cmdline_matching() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'Path("/proc").iterdir()' in text
    assert 'cmdline.split(b"\\0")' in text
    assert 'b"main.py" in parts' in text
    assert 'b"opd_7b_dyme_probe" in parts' in text
    assert "scripts/test/run_pcd_no_visual.sh" in text
    assert "scripts/train_opd_7b_dyme_probe.sh" in text
    assert "pgrep -af" not in text


def test_main5_campaign_defaults_to_sixty_gb_free_gpu_gate() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'GPU_MAX_USED_MB="${DYME_MAIN5_GPU_MAX_USED_MB:-20000}"' in text
    assert 'max_used = float(sys.argv[1])' in text
    assert '"${GPU_MAX_USED_MB}" "${GPU_MAX_UTIL_PCT}"' in text


def test_main5_campaign_gpu_gate_rejects_busy_compute_cards() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'GPU_MAX_UTIL_PCT="${DYME_MAIN5_GPU_MAX_UTIL_PCT:-20}"' in text
    assert "--query-gpu=index,memory.used,utilization.gpu" in text
    assert "float(used.strip()) <= max_used" in text
    assert "float(util.strip()) <= max_util" in text
    assert "max_util_pct=${GPU_MAX_UTIL_PCT}" in text


def test_main5_campaign_launches_on_selected_ready_gpu_subset() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert "ready_gpu_indices()" in text
    assert "SELECTED_CUDA_VISIBLE_DEVICES" in text
    assert 'local cuda_devices="${SELECTED_CUDA_VISIBLE_DEVICES}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${cuda_devices}" \\' in text
    assert 'NUM_GPUS="${REQUIRED_GPUS}" \\' in text
    assert 'DYME_EVAL_NUM_PROCESSES="${DYME_EVAL_NUM_PROCESSES:-${REQUIRED_GPUS}}"' in text


def test_main5_campaign_retries_failed_training_runs() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'TRAIN_MAX_ATTEMPTS="${DYME_MAIN5_TRAIN_MAX_ATTEMPTS:-3}"' in text
    assert "train_failed" in text
    assert "train_rc=$?" in text
    assert "retry in ${TRAIN_RETRY_DELAY}s" in text


def test_main5_campaign_preserves_status_on_restart() -> None:
    script = ROOT / "scripts" / "test" / "run_main5_10epoch_campaign.sh"
    text = script.read_text(encoding="utf-8")

    assert 'if [[ ! -s "${STATUS_TSV}" ]]; then' in text
    assert 'printf "timestamp\\tevent\\tvariant\\tdetail\\n" > "${STATUS_TSV}"' in text
