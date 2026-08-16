import json

from data_utils.chart.data_collector import prepare_chart_rl_data, prepare_chart_sft_data


def test_chart_reference_answer_stays_short_when_sft_target_contains_hint(tmp_path):
    dataset_path = tmp_path / "chart.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is the minimum?",
                    "answer": "70",
                    "image": "missing.png",
                    "human_or_machine": 0,
                    "hint": "Goal: inspect the chart.\nConclusion: the minimum is 70.",
                }
            ]
        ),
        encoding="utf-8",
    )

    rl_row = prepare_chart_rl_data(str(dataset_path))[0]
    sft_row = prepare_chart_sft_data(str(dataset_path))[0]

    assert rl_row["reference_answer"] == "Answer: 70"
    assert rl_row["answer"] == "Answer: 70"
    assert sft_row["reference_answer"] == "Answer: 70"
    assert sft_row["answer"].startswith("Goal: inspect the chart.")
    assert sft_row["answer"].endswith("Answer: 70")
