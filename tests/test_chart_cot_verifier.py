from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_utils.chart_cot_verifier import (
    numbers_equivalent,
    parse_chart_cot,
    parse_deplot_table,
    parse_number,
    verify_chart_cot_trajectory,
    verify_conclusion_answer_consistency,
    verify_grounded_claims,
    verify_reasoning,
)


def test_parse_complete_chart_cot_with_escaped_newlines_and_final_answer() -> None:
    parsed = parse_chart_cot(
        "Goal: Find the minimum.\\n"
        "Observation: 2019: 70 and 2020: 77.\\n"
        "Reasoning: 70 is smaller than 77.\\n"
        "Conclusion: The minimum is 70.\\n"
        "Answer: draft\\nAnswer: 70"
    )

    assert parsed.goal == "Find the minimum."
    assert parsed.observation == "2019: 70 and 2020: 77."
    assert parsed.reasoning == "70 is smaller than 77."
    assert parsed.conclusion == "The minimum is 70."
    assert parsed.answer == "70"
    assert parsed.structure_valid is False
    assert parsed.duplicate_sections == ("answer",)


def test_parse_chart_cot_reports_missing_and_empty_sections() -> None:
    parsed = parse_chart_cot(
        "Goal: Find the minimum.\n"
        "Observation:\n"
        "Conclusion: The minimum is 70.\n"
        "Answer: 70"
    )

    assert parsed.missing_sections == ("reasoning",)
    assert parsed.empty_sections == ("observation",)
    assert parsed.structure_valid is False


def test_parse_json_encoded_deplot_and_normalize_row_width() -> None:
    raw = json.dumps(
        {
            "source": "google/deplot",
            "parsed_table": "Year | Revenue | Profit\n2019 | 1,200 | 39%\n2020 | 900",
        }
    )

    table = parse_deplot_table(raw)

    assert table is not None
    assert table.columns == ("Year", "Revenue", "Profit")
    assert table.rows == (("2019", "1,200", "39%"), ("2020", "900", ""))


def test_parse_deplot_rejects_placeholder() -> None:
    raw = json.dumps(
        {
            "source": "deplot_placeholder",
            "parsed_table": "Year | Value\n2020 | 70",
        }
    )

    assert parse_deplot_table(raw) is None


def test_numeric_normalization_is_percent_aware() -> None:
    assert parse_number("1,200").value == 1200
    assert parse_number("-3.5%").is_percent is True
    assert numbers_equivalent(parse_number("0.39"), parse_number("39%")) is True
    assert numbers_equivalent(parse_number("0.39"), parse_number("39")) is False
    assert numbers_equivalent(parse_number("70"), parse_number("70.0")) is True


def test_grounding_supports_and_contradicts_explicit_label_value_claims() -> None:
    table = parse_deplot_table("Year | Value\n2019 | 70\n2020 | 77")
    assert table is not None

    supported = verify_grounded_claims("Observation: 2019: 70.", table)
    contradicted = verify_grounded_claims("Observation: In 2019, the value is 71.", table)

    assert [(claim.label, claim.value, claim.status) for claim in supported] == [
        ("2019", "70", "supported")
    ]
    assert any(claim.value == "71" and claim.status == "contradicted" for claim in contradicted)


def test_grounding_keeps_bare_derived_number_unknown() -> None:
    table = parse_deplot_table("Year | Value\n2019 | 70\n2020 | 77")
    assert table is not None

    claims = verify_grounded_claims("Reasoning: The difference is 7.", table)

    assert len(claims) == 1
    assert claims[0].value == "7"
    assert claims[0].status == "unknown"


def test_grounding_does_not_bind_derived_result_to_nearby_year_or_series() -> None:
    table = parse_deplot_table(
        "Year | Keep same | Cut back\n2012 | 53 | 30\n"
        "Category | Will be eliminated | Here to stay\nValue | 43 | 49"
    )
    assert table is not None

    year_sum = verify_grounded_claims(
        "Conclusion: The sum of Cut back and Keep same in 2012 is 83.", table
    )
    difference = verify_grounded_claims(
        "Conclusion: The difference between Will be eliminated and Here to stay is 6.", table
    )

    assert not any(claim.status == "contradicted" for claim in year_sum)
    assert not any(claim.status == "contradicted" for claim in difference)


def test_grounding_aggregates_repeated_row_labels_across_series() -> None:
    table = parse_deplot_table(
        "Year | Entity | Value\n2004 | Armenia | 0.6831\n2004 | Singapore | 0.5147"
    )
    assert table is not None

    claims = verify_grounded_claims(
        "Observation: Armenia has 2004: 0.6831 and Singapore has 2004: 0.5147.", table
    )

    assert not any(claim.status == "contradicted" for claim in claims)
    assert sum(claim.status == "supported" for claim in claims) >= 2


def test_grounding_keeps_unmatched_multi_series_column_claim_unknown() -> None:
    table = parse_deplot_table(
        "Entity | 2004\nArmenia | 0.6831\nSingapore |"
    )
    assert table is not None

    claims = verify_grounded_claims(
        "Observation: Singapore has 2004: 0.5147.", table
    )

    assert not any(claim.status == "contradicted" for claim in claims)
    assert any(claim.value == "0.5147" and claim.status == "unknown" for claim in claims)


def test_grounding_tolerates_small_deplot_rounding_difference() -> None:
    table = parse_deplot_table("Year | Value\n2017 | 22.29")
    assert table is not None

    claims = verify_grounded_claims("Observation: 2017: 22.26.", table)

    assert any(claim.value == "22.26" and claim.status == "supported" for claim in claims)


def test_conclusion_answer_consistency_handles_text_and_numeric_values() -> None:
    text = verify_conclusion_answer_consistency("The peak occurs in 2019.", "2019")
    percent = verify_conclusion_answer_consistency("The result is 39%.", "0.39")
    mismatch = verify_conclusion_answer_consistency("The minimum is 70.", "72")

    assert text.status == "consistent"
    assert percent.status == "consistent"
    assert mismatch.status == "inconsistent"


def test_consistency_normalizes_percent_punctuation_and_keeps_implicit_yes_no_unknown() -> None:
    percent = verify_conclusion_answer_consistency("The difference is 17.36%.", "17.36")
    decimal_percent = verify_conclusion_answer_consistency(
        "The percentage is 0.37.", "37%"
    )
    label = verify_conclusion_answer_consistency(
        'The blue graph represents "NET Excellent/good".', "NET Excellent/ good"
    )
    yes_no = verify_conclusion_answer_consistency(
        "The sum is greater than 100.", "Yes"
    )
    explicit_no = verify_conclusion_answer_consistency(
        "The sum is not more than XMRig.", "No"
    )
    incorrect_no = verify_conclusion_answer_consistency(
        "The statement is incorrect.", "No"
    )

    assert percent.status == "consistent"
    assert decimal_percent.status == "consistent"
    assert label.status == "consistent"
    assert yes_no.status == "unknown"
    assert explicit_no.status == "consistent"
    assert incorrect_no.status == "consistent"


def test_consistency_recognizes_high_confidence_yes_no_conclusion_polarity() -> None:
    same_yes = verify_conclusion_answer_consistency(
        "The value of Directly Operated Store is the same as Wholesale.", "Yes"
    )
    indeed_yes = verify_conclusion_answer_consistency(
        "The value of the Don't know segment is indeed 7%.", "Yes"
    )
    differs_no = verify_conclusion_answer_consistency(
        "The value for Germany differs from the value for France.", "No"
    )
    unequal_yes = verify_conclusion_answer_consistency(
        "The two values are unequal.", "Yes"
    )

    assert same_yes.status == "consistent"
    assert indeed_yes.status == "consistent"
    assert differs_no.status == "consistent"
    assert unequal_yes.status == "inconsistent"


def test_consistency_finds_answer_before_trailing_dates_and_accepts_rounded_result_and_lists() -> None:
    dated = verify_conclusion_answer_consistency(
        "The price moved by 1.93 points from Mar 21 to May 21.", "1.93"
    )
    rounded = verify_conclusion_answer_consistency(
        "The ratio is approximately 11.83.", "11.83333333"
    )
    labels = verify_conclusion_answer_consistency(
        "The two countries are Turkey and MEDIAN.", "[Turkey, MEDIAN]"
    )

    assert dated.status == "consistent"
    assert rounded.status == "consistent"
    assert labels.status == "consistent"


def test_reasoning_checks_extrema_count_and_unknown_conservatively() -> None:
    table = parse_deplot_table("Year | Value\n2019 | 70\n2020 | 77\n2021 | 40")
    assert table is not None

    maximum = verify_reasoning("Comparing the values, the maximum is 77.", table)
    bad_minimum = verify_reasoning("The minimum value is 70.", table)
    count = verify_reasoning("Two values are above 50, so the count is 2.", table)
    unknown = verify_reasoning("The chart clearly supports the conclusion.", table)

    assert maximum[0].status == "valid"
    assert bad_minimum[0].status == "invalid"
    assert count[0].status == "valid"
    assert unknown[0].status == "unknown"


def test_reasoning_does_not_treat_argmax_label_or_difference_as_extrema_result() -> None:
    table = parse_deplot_table("Year | Value\n2015 | 80\n2017 | 93")
    assert table is not None

    argmax = verify_reasoning("The highest value occurs in 2017.", table)
    difference = verify_reasoning(
        "Subtract the smallest share 20.8 from the largest share 99.8 to get 79.", table
    )

    assert argmax[0].status == "unknown"
    assert difference[0].status == "unknown"


def test_reasoning_does_not_classify_sum_above_threshold_as_count() -> None:
    table = parse_deplot_table("Country | Value\nJapan | 24\nEstonia | 42.8")
    assert table is not None

    result = verify_reasoning(
        "Summing the values above 20 gives 24 + 42.8 = 66.8.", table
    )

    assert result[0].status == "unknown"


def test_reasoning_extrema_is_unknown_for_multi_series_table() -> None:
    table = parse_deplot_table(
        "Year | Blue | Orange\n2019 | 13.16 | 0.65\n2020 | 20.0 | 5.0"
    )
    assert table is not None

    result = verify_reasoning("The smallest percentage value is 13.16%.", table)

    assert result[0].status == "unknown"


def test_quality_classifier_assigns_q3_q2_q1_and_q0() -> None:
    table = "Year | Value\n2019 | 70\n2020 | 77"
    q3 = verify_chart_cot_trajectory(
        "Goal: Find the maximum.\n"
        "Observation: 2019: 70 and 2020: 77.\n"
        "Reasoning: The maximum is 77.\n"
        "Conclusion: The maximum is 77.\n"
        "Answer: 77",
        table,
        answer_correct=True,
    )
    q2 = verify_chart_cot_trajectory(
        "Goal: Find the answer.\n"
        "Observation: Inspect the chart.\n"
        "Reasoning: Use the visual evidence.\n"
        "Conclusion: Therefore this is the result.\n"
        "Answer: 77",
        table,
        answer_correct=True,
    )
    q1 = verify_chart_cot_trajectory(
        "Reasoning: The maximum is 77.\nAnswer: 77",
        table,
        answer_correct=True,
    )
    q0 = verify_chart_cot_trajectory(
        "Goal: Find the maximum.\n"
        "Observation: 2019: 71.\n"
        "Reasoning: The maximum is 77.\n"
        "Conclusion: The maximum is 70.\n"
        "Answer: 77",
        table,
        answer_correct=True,
    )

    assert q3.quality == "Q3"
    assert q2.quality == "Q2"
    assert q1.quality == "Q1"
    assert q0.quality == "Q0"
    assert "grounding_contradiction" in q0.reason_codes
    assert "conclusion_answer_inconsistent" in q0.reason_codes


def test_quality_grounding_uses_observation_not_derived_conclusion() -> None:
    result = verify_chart_cot_trajectory(
        "Goal: Find the sum.\n"
        "Observation: 2012: 53 and another value is 30.\n"
        "Reasoning: The sum of 53 and 30 is 83.\n"
        "Conclusion: The sum in 2012 is 83.\n"
        "Answer: 83",
        "Year | First | Second\n2012 | 53 | 30",
        answer_correct=True,
    )

    assert result.quality == "Q3"
    assert "grounding_contradiction" not in result.reason_codes
