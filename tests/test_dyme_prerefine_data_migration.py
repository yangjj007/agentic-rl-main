import hashlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.loader import load_config
from data_utils.aokvqa.data_collector import prepare_world_rl_data, prepare_world_sft_data
from data_utils.chart.data_collector import prepare_chart_rl_data, prepare_chart_sft_data
from data_utils.commom_util import define_task_data_func
from data_utils.lm_math.data_collector import prepare_math_lm_rl_data

PREREFINE_DATASET = os.path.join(PROJECT_ROOT, "data", "chartqa", "train_new_prerefine.json")
AOKVQA_DATASET = os.path.join(PROJECT_ROOT, "data", "aokvqa", "train.json")
GSM8K_DATASET = os.path.join(PROJECT_ROOT, "data", "gsm8k", "train.json")
AOKVQA_QWEN25_REFINE_DATASET = os.path.join(PROJECT_ROOT, "data", "aokvqa", "train_qwen25_refine.json")
GSM8K_QWEN25_REFINE_DATASET = os.path.join(PROJECT_ROOT, "data", "gsm8k", "train_qwen25_refine.json")


def test_chartqa_prerefine_dataset_is_migrated_and_dyme_shaped():
    assert os.path.exists(PREREFINE_DATASET)
    with open(PREREFINE_DATASET, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == (
            "3c57bb96b3c3fddc5d993703948fca7d84f7744773acb43eb6ad2183b7979f6d"
        )

    with open(PREREFINE_DATASET, "r", encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == 4576
    assert all(row.get("human_or_machine") == 0 for row in rows)
    assert all(row.get("hint") for row in rows)
    assert all(row.get("question") and row.get("answer") and row.get("image") for row in rows)
    assert sum("Goal:" in row["hint"] for row in rows) >= 4500


def test_change_config_reads_migrated_prerefine_dataset():
    cfg = load_config("change")
    assert cfg["dataset"]["train_dataset"] == PREREFINE_DATASET

    rl_rows = prepare_chart_rl_data(cfg["dataset"]["train_dataset"])
    sft_rows = prepare_chart_sft_data(cfg["dataset"]["train_dataset"])
    assert len(rl_rows) == 4576
    assert len(sft_rows) == 4576
    assert rl_rows[0]["answer"].startswith("Answer:")
    assert rl_rows[0]["reference_answer"] == rl_rows[0]["answer"]
    assert "Goal:" in sft_rows[0]["answer"]
    assert sft_rows[0]["reference_answer"].startswith("Answer:")
    assert "Goal:" not in sft_rows[0]["reference_answer"]
    # Image assets are intentionally not versioned with the JSON metadata.
    # The loader preserves a canonical project-relative path; strict training
    # recipes validate that the real local images are mounted before launch.
    assert rl_rows[0]["image"].endswith("train_000048.png")


def test_dyme_three_task_configs_read_migrated_jsons():
    chart_cfg = load_config("change")
    aok_cfg = load_config("aok")
    gsm_cfg = load_config("llm")

    assert chart_cfg["dataset"]["train_dataset"] == PREREFINE_DATASET
    assert aok_cfg["dataset"]["train_dataset"] == AOKVQA_QWEN25_REFINE_DATASET
    assert gsm_cfg["dataset"]["train_dataset"] == GSM8K_QWEN25_REFINE_DATASET
    assert all(
        os.path.exists(cfg["dataset"]["train_dataset"])
        for cfg in (chart_cfg, aok_cfg, gsm_cfg)
    )

    chart_rows = prepare_chart_rl_data(chart_cfg["dataset"]["train_dataset"])
    aok_rows = prepare_world_rl_data(aok_cfg["dataset"]["train_dataset"])
    gsm_rows = prepare_math_lm_rl_data(gsm_cfg["dataset"]["train_dataset"])

    assert len(chart_rows) == 4576
    assert len(aok_rows) == 17055
    assert len(gsm_rows) == 7473
    assert chart_rows[0]["answer"].startswith("Answer:")
    assert aok_rows[0]["answer"].startswith("Answer:")
    assert gsm_rows[0]["answer"].startswith("Answer:")
    assert aok_rows[0]["image"].endswith("train_0000000.png")


def test_qwen25_refined_jsons_preserve_rows_and_record_provenance():
    for raw_path, refined_path, expected_len in (
        (AOKVQA_DATASET, AOKVQA_QWEN25_REFINE_DATASET, 17055),
        (GSM8K_DATASET, GSM8K_QWEN25_REFINE_DATASET, 7473),
    ):
        assert os.path.exists(refined_path)
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_rows = json.load(f)
        with open(refined_path, "r", encoding="utf-8") as f:
            refined_rows = json.load(f)

        assert len(refined_rows) == expected_len
        assert len(refined_rows) == len(raw_rows)
        first = refined_rows[0]
        assert first["question"] == raw_rows[0]["question"]
        assert first["answer"] == raw_rows[0]["answer"]
        assert first.get("dyme_rewrite", {}).get("model") == "Qwen/Qwen2.5-14B-Instruct-AWQ"
        assert first.get("dyme_rewrite", {}).get("status") == "ok"
        assert "Goal:" in first["hint"]
        assert "Conclusion:" in first["hint"]


def test_grpo_and_opd_modes_use_rl_data_collectors():
    for mode in ("grpo", "opd"):
        assert define_task_data_func("chart", mode=mode) is prepare_chart_rl_data
        assert define_task_data_func("world", mode=mode) is prepare_world_rl_data


def test_sft_mode_keeps_sft_data_collectors():
    assert define_task_data_func("chart", mode="sft") is prepare_chart_sft_data
    assert define_task_data_func("world", mode="sft") is prepare_world_sft_data
