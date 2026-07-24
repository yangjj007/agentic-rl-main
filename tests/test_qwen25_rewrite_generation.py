import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_dyme_qwen25_rewrites import (
    answer_in_conclusion,
    is_cached_complete,
    is_cached_complete_for_args,
    rewrite_item,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_completion(self, user_prompt, system_prompt=None, max_tokens=1024):
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        if len(self.calls) == 1:
            return '[{"fact": "Natalia sold 48 clips in April and 24 clips in May."}]'
        return (
            "Goal: Find the total number of clips Natalia sold.\n"
            "Observation: Natalia sold 48 clips in April and 24 clips in May.\n"
            "Reasoning: Add the two monthly counts: 48 + 24 = 72.\n"
            "Conclusion: Natalia sold 72 clips altogether."
        )


def test_rewrite_item_uses_two_step_dyme_refine_and_records_metadata():
    item = {
        "question": "Natalia sold 48 clips in April and half as many in May. How many total?",
        "answer": "72",
        "hint": "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72.",
    }
    client = FakeClient()

    rewritten = rewrite_item(item, task="gsm8k", client=client, model_id="Qwen/Qwen2.5-14B-Instruct-AWQ")

    assert len(client.calls) == 2
    assert "extract all the visual elements" in client.calls[0]["user_prompt"]
    assert "convert my input into the target reasoning text" in client.calls[1]["user_prompt"]
    assert rewritten["hint"].startswith("Goal:")
    assert rewritten["answer"] == "72"
    assert rewritten["dyme_rewrite"]["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert rewritten["dyme_rewrite"]["status"] == "ok"
    assert rewritten["dyme_rewrite"]["task"] == "gsm8k"
    assert rewritten["dyme_rewrite"]["source_hint_chars"] > 0


def test_rewrite_item_falls_back_to_original_hint_on_helper_failure():
    class BrokenClient:
        def get_completion(self, *args, **kwargs):
            raise RuntimeError("helper down")

    item = {"question": "q", "answer": "a", "hint": "original rationale"}

    rewritten = rewrite_item(item, task="aokvqa", client=BrokenClient(), model_id="Qwen/Qwen2.5-14B-Instruct-AWQ")

    assert rewritten["hint"] == "original rationale"
    assert rewritten["dyme_rewrite"]["status"] == "fallback"
    assert "helper down" in rewritten["dyme_rewrite"]["error"]


def test_aokvqa_rewrite_uses_existing_visual_fact_as_context():
    class FakeAokClient:
        def __init__(self):
            self.calls = []

        def get_completion(self, user_prompt, system_prompt=None, max_tokens=1024):
            self.calls.append(user_prompt)
            return (
                "Goal: Answer the visual commonsense question.\n"
                "Observation: The image shows luggage near a person on a street.\n"
                "Reasoning: Luggage near a street supports waiting for a cab.\n"
                "Conclusion: The answer is cab."
            )

    item = {
        "question": "What is the man by the bags awaiting?",
        "answer": "cab",
        "choices": ["skateboarder", "train", "delivery", "cab"],
        "visual_fact": '{"description": "A man stands by luggage on a street."}',
        "hint": "The luggage and street setting suggest he is waiting for a cab.",
    }
    client = FakeAokClient()

    rewritten = rewrite_item(item, task="aokvqa", client=client, model_id="Qwen/Qwen2.5-14B-Instruct-AWQ")

    assert len(client.calls) == 1
    assert "extract all the visual elements" not in client.calls[0]
    assert "convert my input into the target reasoning text" in client.calls[0]
    assert "A man stands by luggage" in client.calls[0]
    assert rewritten["dyme_rewrite"]["structured_context_source"] == "aokvqa_visual_fact_direct"
    assert rewritten["dyme_rewrite"]["status"] == "ok"


def test_retry_fallback_marks_cached_fallback_as_pending():
    ok_row = {"dyme_rewrite": {"status": "ok"}}
    fallback_row = {"dyme_rewrite": {"status": "fallback"}}

    assert is_cached_complete(ok_row, retry_fallback=False) is True
    assert is_cached_complete(fallback_row, retry_fallback=False) is True
    assert is_cached_complete(ok_row, retry_fallback=True) is True
    assert is_cached_complete(fallback_row, retry_fallback=True) is False


def test_gsm8k_answer_mismatch_can_mark_cached_row_pending():
    ok_matching = {
        "answer": "720",
        "hint": "Goal: q\nObservation: f\nReasoning: r\nConclusion: Carrie has $720 left.",
        "dyme_rewrite": {"status": "ok"},
    }
    ok_mismatch = {
        "answer": "720",
        "hint": "Goal: q\nObservation: f\nReasoning: r\nConclusion: Carrie has $320 left.",
        "dyme_rewrite": {"status": "ok"},
    }
    word_number = {
        "answer": "30",
        "hint": "Goal: q\nObservation: f\nReasoning: r\nConclusion: Thirty percent are present.",
        "dyme_rewrite": {"status": "ok"},
    }
    negated_answer = {
        "answer": "650",
        "hint": "Goal: q\nObservation: f\nReasoning: r\nConclusion: The answer is $325, not $650.",
        "dyme_rewrite": {"status": "ok"},
    }

    assert answer_in_conclusion(ok_matching["hint"], "720") is True
    assert answer_in_conclusion(ok_mismatch["hint"], "720") is False
    assert answer_in_conclusion(word_number["hint"], "30") is True
    assert answer_in_conclusion(negated_answer["hint"], "650") is False
    assert is_cached_complete_for_args(
        ok_matching,
        task="gsm8k",
        retry_fallback=True,
        retry_answer_mismatch=True,
    ) is True
    assert is_cached_complete_for_args(
        ok_mismatch,
        task="gsm8k",
        retry_fallback=True,
        retry_answer_mismatch=True,
    ) is False
