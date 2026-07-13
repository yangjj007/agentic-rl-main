from __future__ import annotations

import json
from pathlib import Path


def test_positive_replay_buffer_builds_sft_samples_from_target_field(tmp_path: Path) -> None:
    from opsd_utils.positive_replay import PositiveReplayConfig, PositiveReplayBuffer

    replay_path = tmp_path / "replay_train.json"
    replay_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is the lowest value?",
                    "image": "/chartqa_output/images/train_001.png",
                    "answer": "70",
                    "hint": "Reasoning: This hint should not be concatenated.",
                    "target": "Reasoning: Compare values.\nAnswer: 70",
                }
            ]
        ),
        encoding="utf-8",
    )

    buffer = PositiveReplayBuffer(
        PositiveReplayConfig(enabled=True, dataset_path=str(replay_path), batch_size=1, weight=0.2),
        process_index=3,
    )
    rows = buffer.sample(global_step=0)

    assert len(rows) == 1
    assert rows[0]["question_wo_prompt"] == "What is the lowest value?"
    assert rows[0]["answer"] == "Reasoning: Compare values.\nAnswer: 70"
    assert rows[0]["answer"].count("Answer:") == 1
    assert "What is the lowest value?" in rows[0]["prompt"]
    assert rows[0]["image"] == "/chartqa_output/images/train_001.png"


def test_positive_replay_buffer_respects_step_window_and_batch_size(tmp_path: Path) -> None:
    from opsd_utils.positive_replay import PositiveReplayConfig, PositiveReplayBuffer

    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question": f"Question {idx}?",
                    "image": f"image_{idx}.png",
                    "answer": str(idx),
                    "hint": f"Reasoning: {idx}",
                }
            )
            for idx in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    buffer = PositiveReplayBuffer(
        PositiveReplayConfig(
            enabled=True,
            dataset_path=str(replay_path),
            batch_size=2,
            weight=0.1,
            after_step=10,
            until_step=20,
            seed=7,
        ),
        process_index=0,
    )

    assert buffer.enabled_for_step(9) is False
    assert buffer.sample(global_step=9) == []
    assert buffer.enabled_for_step(10) is True
    assert len(buffer.sample(global_step=10)) == 2
    assert buffer.enabled_for_step(21) is False


def test_positive_replay_config_disabled_when_missing_path() -> None:
    from opsd_utils.positive_replay import PositiveReplayConfig, PositiveReplayBuffer

    buffer = PositiveReplayBuffer(
        PositiveReplayConfig(enabled=True, dataset_path="/tmp/does-not-exist-replay.json", batch_size=2),
        process_index=0,
    )

    assert buffer.available is False
    assert buffer.sample(global_step=0) == []
