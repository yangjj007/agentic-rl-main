import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.chart.evaluator import (
    eval_teacher_probe_chart,
    parse_teacher_probe_answer,
)


def test_teacher_probe_parser_accepts_explicit_answer_line():
    parsed = parse_teacher_probe_answer("Reasoning...\nAnswer: 70")

    assert parsed.answer == "70"
    assert parsed.has_answer_flag is True
    assert parsed.parse_failed is False


def test_teacher_probe_parser_accepts_bare_short_answer():
    parsed = parse_teacher_probe_answer("70")

    assert parsed.answer == "70"
    assert parsed.has_answer_flag is False
    assert parsed.parse_failed is False


def test_teacher_probe_parser_accepts_common_answer_phrase():
    parsed = parse_teacher_probe_answer("The answer is 70.")

    assert parsed.answer == "70"
    assert parsed.has_answer_flag is False
    assert parsed.parse_failed is False


def test_teacher_probe_parser_marks_missing_answer_as_parse_failure():
    parsed = parse_teacher_probe_answer("I need more information from the chart.")

    assert parsed.answer == ""
    assert parsed.has_answer_flag is False
    assert parsed.parse_failed is True


def test_teacher_probe_chart_eval_uses_parsed_answer():
    score, parsed = eval_teacher_probe_chart(
        "The answer is 70.",
        "Answer: 70",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "70"
    assert parsed.parse_failed is False
