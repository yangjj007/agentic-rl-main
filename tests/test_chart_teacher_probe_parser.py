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


def test_teacher_probe_chart_eval_accepts_chart_label_footnote_markers():
    score, parsed = eval_teacher_probe_chart(
        "Answer: Domestic",
        "Domestic*",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "Domestic"


def test_teacher_probe_chart_eval_accepts_parenthesized_chart_labels():
    score, parsed = eval_teacher_probe_chart(
        "Answer: Total Non Market Work (Men)",
        "Total Non Market Work (Men)",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "Total Non Market Work Men"


def test_teacher_probe_chart_eval_accepts_list_answer_surface_forms():
    score, parsed = eval_teacher_probe_chart(
        "Answer: Latvia and Australia",
        "[Latvia, Australia]",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "Latvia and Australia"


def test_teacher_probe_chart_eval_rejects_partial_list_answer():
    score, parsed = eval_teacher_probe_chart(
        "Answer: 15.28%",
        "[71.96, 15.28]",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 0.0
    assert parsed.answer == "15.28%"


def test_teacher_probe_chart_eval_accepts_numeric_answer_with_unit_context():
    score, parsed = eval_teacher_probe_chart(
        "Answer: 3 TWh (2000)",
        "3",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "3 TWh 2000"


def test_teacher_probe_chart_eval_accepts_simple_trend_inflection():
    score, parsed = eval_teacher_probe_chart(
        "Answer: Decreased",
        "decreasing",
        max_relative_change=0.05,
        answer_flag="answer:",
    )

    assert score == 1.0
    assert parsed.answer == "Decreased"
