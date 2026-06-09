import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.privileged import build_privileged_context


def test_text_provider():
    sample = {"hint": "Rep=67", "answer": "Answer: 131"}
    suffix, _ = build_privileged_context(sample, ["text"])
    assert "Rep=67" in suffix
    assert "131" in suffix


def test_hybrid_provider():
    sample = {"hint": "step", "visual_fact": "bar=3", "answer": "Answer: 3"}
    suffix, _ = build_privileged_context(sample, ["hybrid"])
    assert "Visual Facts" in suffix
    assert "Reference" in suffix


if __name__ == "__main__":
    test_text_provider()
    test_hybrid_provider()
    print("Privileged provider tests passed.")
