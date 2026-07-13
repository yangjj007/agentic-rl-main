from eval.output_behavior import summarize_output_behavior_counts


def test_summarize_output_behavior_counts_separates_reasoning_pathologies():
    counts = summarize_output_behavior_counts(
        [
            "Goal: Find max.\nObservation: 2 and 5.\nReasoning: 5 is larger.\nConclusion: max is 5.\nAnswer: 5",
            "Goal: Compare.\nObservation: inspect bars\nReasoning: calculate",
            "Goal:\nObservation.\nReasoning,\nConclusion,\nAnswer:.50 .",
            "Answer: 7",
        ]
    )

    assert counts == {
        "total": 4,
        "full_cot_template": 2,
        "partial_cot_template": 1,
        "goal_without_answer": 1,
        "empty_cot_skeleton": 1,
        "malformed_answer_section": 1,
    }


def test_summarize_output_behavior_counts_handles_empty_input():
    assert summarize_output_behavior_counts([]) == {
        "total": 0,
        "full_cot_template": 0,
        "partial_cot_template": 0,
        "goal_without_answer": 0,
        "empty_cot_skeleton": 0,
        "malformed_answer_section": 0,
    }
