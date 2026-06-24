from __future__ import annotations

from opsd_utils.indexing import source_row_index


def test_expanded_completion_row_uses_same_row_sample_and_answer() -> None:
    num_generations = 8
    expanded_count = 16
    samples = (
        [{"question": "Which year?", "answer": "2009"} for _ in range(num_generations)]
        + [{"question": "Is Earrings greater than Rings?", "answer": "No"} for _ in range(num_generations)]
    )
    answers = [sample["answer"] for sample in samples]

    row = 8
    prompt_group_idx = row // num_generations
    source_idx = source_row_index(
        row,
        raw_count=len(samples),
        expanded_count=expanded_count,
        num_generations=num_generations,
    )

    assert answers[prompt_group_idx] == "2009"
    assert samples[source_idx]["question"] == "Is Earrings greater than Rings?"
    assert answers[source_idx] == "No"


def test_raw_prompt_row_maps_completion_row_to_prompt_group() -> None:
    num_generations = 8
    samples = [
        {"question": "Which year?", "answer": "2009"},
        {"question": "Is Earrings greater than Rings?", "answer": "No"},
    ]

    source_idx = source_row_index(
        8,
        raw_count=len(samples),
        expanded_count=len(samples) * num_generations,
        num_generations=num_generations,
    )

    assert samples[source_idx]["answer"] == "No"
