from __future__ import annotations

from contextlib import contextmanager
import json
import signal

from opsd_utils.evidence_harness import (
    EvidenceAction,
    EvidenceCandidate,
    HarnessAttempt,
    HarnessDecision,
    HarnessStatus,
    ValidationResult,
)
from opsd_utils.evidence_harness.chartqa import (
    build_chartqa_arithmetic_recovery_suffix,
    build_chartqa_executable_deplot_recovery_suffix,
    build_chartqa_target_phrase_recovery_suffix,
    build_chartqa_candidate,
    build_visual_base_suffix,
    build_visual_deplot_suffix,
    build_visual_recovery_suffix,
    decide_after_parallel_attempts,
    decide_after_recovery,
    is_chartqa_visual_quarantine_question,
    validate_chartqa_candidate,
)


def _deplot(table: str) -> str:
    return json.dumps({"source": "google/deplot", "parsed_table": table})


@contextmanager
def _deadline(seconds: float):
    def _raise_timeout(_signum, _frame):
        raise TimeoutError("deadline exceeded")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_contracts_serialize_without_reference_fields() -> None:
    candidate = EvidenceCandidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        answer="42",
        raw_output="Reasoning: 40 + 2 = 42\nAnswer: 42",
    )
    validation = ValidationResult(
        validator_id="chartqa",
        status="PASS",
        reason_code="cross_attempt_agreement",
        supporting_refs=("base", "deplot"),
    )
    attempt = HarnessAttempt(candidate=candidate, validation=validation, cost=1.0)
    decision = HarnessDecision(
        status=HarnessStatus.ACCEPTED,
        selected_attempt_id="base",
        reason_code="cross_attempt_agreement",
        remaining_budget=1,
    )

    payload = {
        "attempt": attempt.to_dict(),
        "decision": decision.to_dict(),
    }

    assert payload["attempt"]["candidate"]["action"] == "visual_base"
    assert payload["decision"]["status"] == "accepted"
    assert "reference" not in json.dumps(payload).lower()


def test_chartqa_candidate_extracts_last_answer_line() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 41\nI checked again.\nAnswer: 42",
    )

    assert candidate.answer == "42"
    assert candidate.parse_failed is False


def test_chartqa_candidate_marks_missing_answer_as_parse_failure() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="The chart appears to show forty two.",
    )

    assert candidate.answer is None
    assert candidate.parse_failed is True


def test_chartqa_candidate_accepts_single_line_short_answer_fallback() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="64.86",
    )

    assert candidate.answer == "64.86"
    assert candidate.parse_failed is False


def test_deplot_validator_finds_direct_cell_support() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="The lowest value is visible in 2019.\nAnswer: 70",
    )

    result = validate_chartqa_candidate(
        candidate,
        _deplot("Year | Value\n2018 | 72\n2019 | 70\n2020 | 77"),
    )

    assert result.status == "PASS"
    assert result.reason_code == "deplot_direct_support"
    assert "70" in result.supporting_refs


def test_deplot_validator_finds_derived_arithmetic_support() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Subtract the two chart values: 72 - 30 = 42.\nAnswer: 42",
    )

    result = validate_chartqa_candidate(
        candidate,
        _deplot("Year | A | B\n2020 | 72 | 30"),
    )

    assert result.status == "PASS"
    assert result.reason_code == "deplot_derived_support"
    assert result.deterministic_value == "42"


def test_deplot_validator_does_not_stall_on_long_operator_noise() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 42\n" + ("1 + " * 5000),
    )

    with _deadline(0.5):
        result = validate_chartqa_candidate(
            candidate,
            _deplot("Year | A | B\n2020 | 72 | 30"),
        )

    assert result.status == "UNKNOWN"


def test_chartqa_arithmetic_suffix_is_gold_hidden_and_requires_equation() -> None:
    suffix = build_chartqa_arithmetic_recovery_suffix(
        _deplot("Year | A | B\n2020 | 72 | 30")
    )

    assert "Operands:" in suffix
    assert "Operation:" in suffix
    assert "Equation:" in suffix
    assert "Answer:" in suffix
    assert "72" in suffix
    assert "Reference Answer" not in suffix
    assert "Verified Hint" not in suffix
    assert "####" not in suffix


def test_chartqa_target_phrase_suffix_recovers_question_matched_rows() -> None:
    suffix = build_chartqa_target_phrase_recovery_suffix(
        "What's the percentage value of U.S. adults who have heard a lot about facial recognition technology?",
        _deplot("Response | Percent\nA little | 61\nA lot | 25\nNothing at all | 14"),
    )

    assert "[Recovered Candidate Evidence]" in suffix
    assert "A lot | 25" in suffix
    assert suffix.index("A lot | 25") < suffix.index("[Visual Facts - DePlot]")
    assert "Reference Answer" not in suffix
    assert "Verified Hint" not in suffix


def test_chartqa_target_phrase_suffix_recovers_threshold_rows() -> None:
    suffix = build_chartqa_target_phrase_recovery_suffix(
        "Which year and gender has a value above 800?",
        _deplot(
            "Year | Between men | Between women\n"
            "2017 | 620 | 755\n"
            "2018 | 675 | 820\n"
            "2019 | 690 | 790"
        ),
    )

    assert "Threshold rows: > 800" in suffix
    assert "2018 | 675 | 820" in suffix
    assert "2017 | 620 | 755" not in suffix.split("[Recovered Candidate Evidence]", 1)[1].split("[Visual Facts - DePlot]", 1)[0]


def test_chartqa_executable_deplot_recovery_computes_threshold_sum() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "What is the sum of the bars which is above 200 ?",
        _deplot(
            "Characteristic | Number of drugs and vaccines\n"
            "Preclinical | 707\n"
            "Public Clinical | 98\n"
            "Phase II Clinical | 216\n"
            "Registered | 4"
        ),
    )

    assert "[Executable DePlot Recovery]" in suffix
    assert "Operation: threshold_sum" in suffix
    assert "Preclinical | Number of drugs and vaccines = 707" in suffix
    assert "Phase II Clinical | Number of drugs and vaccines = 216" in suffix
    assert "Candidate answer: 923" in suffix
    assert "Reference Answer" not in suffix


def test_chartqa_executable_deplot_recovery_computes_threshold_count() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "How many countries have less than 200 million euros?",
        _deplot(
            "Characteristic | Brand value in million euros\n"
            "England | 8578\n"
            "Netherlands | 198\n"
            "Scotland | 110\n"
            "Portugal | 114\n"
            "Russia | 100"
        ),
    )

    assert "Operation: threshold_count" in suffix
    assert "Candidate answer: 4" in suffix


def test_chartqa_executable_deplot_recovery_maps_threshold_cell_to_row_and_column() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "Which year and gender has a value above 800?",
        _deplot(
            "Characteristic | Between men | Between women\n"
            "2017 | 620 | 755\n"
            "2018 | 682 | 820\n"
            "2019 | 675 | 744"
        ),
    )

    assert "Operation: threshold_label_lookup" in suffix
    assert "2018 | Between women = 820" in suffix
    assert "Candidate answer: [2018, Between women]" in suffix


def test_chartqa_executable_deplot_recovery_counts_exact_percent_matches() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "How many countries has a share of 3%?",
        _deplot(
            "Characteristic | Distribution of exports\n"
            "Canada | 25%\n"
            "Nigeria | 3%\n"
            "Peru | 3%\n"
            "Philippines | 3%\n"
            "Rest of world | 9%"
        ),
    )

    assert "Operation: exact_value_count" in suffix
    assert "Candidate answer: 3" in suffix


def test_chartqa_executable_deplot_recovery_subtracts_named_label_from_threshold_sum() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "The sum of revenue shares of countries below 30% minus Europe revenue share equals to what?",
        _deplot(
            "Characteristic | Revenues share\n"
            "Europe* | 43%\n"
            "Italy | 28%\n"
            "Other countries | 23%\n"
            "North America | 6%"
        ),
    )

    assert "Operation: threshold_sum_minus_label" in suffix
    assert "Subtracted cell: Europe* | Revenues share = 43" in suffix
    assert "Candidate answer: 14" in suffix


def test_chartqa_executable_deplot_recovery_finds_label_for_exact_value_in_year() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "In 2015 which category recorded 14.12 %",
        _deplot(
            "Characteristic | 0-14 years | 15-64 years | 65 years and older\n"
            "2016 | 14.17% | 66.92% | 18.91%\n"
            "2015 | 14.12% | 67.04% | 18.84%"
        ),
    )

    assert "Operation: exact_value_label_lookup" in suffix
    assert "2015 | 0-14 years = 14.12" in suffix
    assert "Candidate answer: 0-14 years" in suffix


def test_chartqa_executable_deplot_recovery_finds_row_from_value_signature() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "Data set 67%,62%,47%,40%, find its category?",
        _deplot(
            "Characteristic | Pain | Disease | Burden | Family burden\n"
            "All adults | 62% | 56% | 38% | 32%\n"
            "White Catholic | 67% | 62% | 47% | 40%\n"
            "Black Protestant | 42% | 34% | 26% | 21%"
        ),
    )

    assert "Operation: value_signature_lookup" in suffix
    assert "Candidate answer: White Catholic" in suffix


def test_chartqa_executable_deplot_recovery_computes_median_of_all_values() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "What is the median of all the bars?",
        _deplot(
            "Country | Health expenditure per person, 1995\n"
            "Japan | 1533.52\n"
            "Slovenia | 972.27\n"
            "Jamaica | 261.43\n"
            "Benin | 47.83"
        ),
    )

    assert "Operation: median_all_values" in suffix
    assert "Candidate answer: 616.85" in suffix


def test_chartqa_executable_deplot_recovery_finds_pair_sum_labels() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "Which two regions consists of a total of 1574.5?",
        _deplot(
            "Characteristic | Net sales in million U.S. dollars\n"
            "North America | 2483.9\n"
            "Western Europe | 945.9\n"
            "Asia-Pacific | 628.6\n"
            "Rest of World* | 294.7"
        ),
    )

    assert "Operation: pair_sum_label_lookup" in suffix
    assert "Candidate answer: [Western Europe, Asia-Pacific]" in suffix


def test_chartqa_executable_deplot_recovery_finds_same_value_pair_for_year() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "In which two quarters the average daily rate of hotels is the same in 2016?",
        _deplot(
            "Characteristic | 2016 | 2017\n"
            "Q4 | 158 | -\n"
            "Q3 | 148 | -\n"
            "Q2 | 148 | -\n"
            "Q1 | 178 | 28"
        ),
    )

    assert "Operation: same_value_pair_lookup" in suffix
    assert "Candidate answer: [Q2, Q3]" in suffix


def test_chartqa_executable_deplot_recovery_counts_column_comparisons() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        'How many countries "increase" value is maximum than its "Do not Increase" value?',
        _deplot(
            "Entity | Do not increase | Increase\n"
            "France | 58 | 33.0\n"
            "Italy | 46 | 50.0\n"
            "Germany | 42 | 51.0"
        ),
    )

    assert "Operation: column_comparison_count" in suffix
    assert "Candidate answer: 2" in suffix


def test_chartqa_executable_deplot_recovery_finds_max_consecutive_change_label() -> None:
    suffix = build_chartqa_executable_deplot_recovery_suffix(
        "Which year experienced the most drastic change in the number of registered players?",
        _deplot(
            "Characteristic | Number of players\n"
            "2015/16 | 4968\n"
            "2014/15 | 4851\n"
            "2013/14 | 8355\n"
            "2012/13 | 7255"
        ),
    )

    assert "Operation: max_consecutive_change" in suffix
    assert "Candidate answer: 2014/15" in suffix


def test_chartqa_arithmetic_quarantine_detects_visual_geometry_questions() -> None:
    assert is_chartqa_visual_quarantine_question(
        "Where do the blue and orange bars intersect?"
    )
    assert is_chartqa_visual_quarantine_question("What is the y-axis tick interval?")
    assert is_chartqa_visual_quarantine_question("What value does the grey slice represent?")
    assert not is_chartqa_visual_quarantine_question(
        "What is the sum of the bars above 200?"
    )


def test_deplot_validator_returns_unknown_for_unsupported_answer() -> None:
    candidate = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 999",
    )

    result = validate_chartqa_candidate(
        candidate,
        _deplot("Year | Value\n2019 | 70\n2020 | 72"),
    )

    assert result.status == "UNKNOWN"
    assert result.reason_code == "deplot_support_unresolved"


def test_decision_accepts_cross_attempt_agreement() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 70",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 70",
    )

    decision = decide_after_parallel_attempts(base, deplot, max_attempts=3)

    assert decision.status is HarnessStatus.ACCEPTED
    assert decision.selected_attempt_id == "deplot"
    assert decision.reason_code == "cross_attempt_agreement"
    assert decision.remaining_budget == 1


def test_decision_requests_recovery_on_disagreement() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 70",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 72",
    )

    decision = decide_after_parallel_attempts(base, deplot, max_attempts=3)

    assert decision.status is HarnessStatus.ACTIVE
    assert decision.next_action is EvidenceAction.VISUAL_RECOVERY
    assert decision.reason_code == "candidate_conflict"
    assert decision.remaining_budget == 1


def test_decision_requests_recovery_when_one_attempt_does_not_parse() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="No final line",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 72",
    )

    decision = decide_after_parallel_attempts(base, deplot, max_attempts=3)

    assert decision.status is HarnessStatus.ACTIVE
    assert decision.next_action is EvidenceAction.VISUAL_RECOVERY
    assert decision.reason_code == "candidate_parse_failure"


def test_recovery_decision_accepts_parseable_answer() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 70",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 72",
    )
    recovery = build_chartqa_candidate(
        attempt_id="recovery",
        action=EvidenceAction.VISUAL_RECOVERY,
        output="After checking the image, the answer is 70.\nAnswer: 70",
    )

    decision = decide_after_recovery(recovery, base=base, deplot=deplot, max_attempts=3)

    assert decision.status is HarnessStatus.ACCEPTED
    assert decision.selected_attempt_id == "recovery"
    assert decision.reason_code == "recovery_confirms_visual"
    assert decision.remaining_budget == 0


def test_recovery_decision_abstains_when_third_answer_has_no_support() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 12",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 8",
    )
    recovery = build_chartqa_candidate(
        attempt_id="recovery",
        action=EvidenceAction.VISUAL_RECOVERY,
        output="Answer: 9",
    )

    decision = decide_after_recovery(recovery, base=base, deplot=deplot, max_attempts=3)

    assert decision.status is HarnessStatus.ABSTAINED
    assert decision.selected_attempt_id is None
    assert decision.reason_code == "recovery_does_not_confirm_visual"


def test_recovery_decision_abstains_when_only_deplot_is_confirmed() -> None:
    base = build_chartqa_candidate(
        attempt_id="base",
        action=EvidenceAction.VISUAL_BASE,
        output="Answer: 12",
    )
    deplot = build_chartqa_candidate(
        attempt_id="deplot",
        action=EvidenceAction.ATTACH_DEPLOT,
        output="Answer: 8",
    )
    recovery = build_chartqa_candidate(
        attempt_id="recovery",
        action=EvidenceAction.VISUAL_RECOVERY,
        output="Answer: 8",
    )
    validation = ValidationResult(
        validator_id="chartqa_deplot",
        status="PASS",
        reason_code="deplot_direct_support",
    )

    decision = decide_after_recovery(
        recovery,
        base=base,
        deplot=deplot,
        validation=validation,
        max_attempts=3,
    )

    assert decision.status is HarnessStatus.ABSTAINED
    assert decision.selected_attempt_id is None
    assert decision.reason_code == "recovery_does_not_confirm_visual"


def test_recovery_decision_abstains_on_parse_failure() -> None:
    recovery = build_chartqa_candidate(
        attempt_id="recovery",
        action=EvidenceAction.VISUAL_RECOVERY,
        output="I cannot determine it.",
    )

    decision = decide_after_recovery(recovery, max_attempts=3)

    assert decision.status is HarnessStatus.ABSTAINED
    assert decision.selected_attempt_id is None
    assert decision.reason_code == "recovery_parse_failure"


def test_visual_base_prompt_keeps_image_and_excludes_deplot() -> None:
    suffix = build_visual_base_suffix()

    assert "full chart image" in suffix.lower()
    assert "DePlot" not in suffix
    assert "Answer:" in suffix


def test_visual_deplot_prompt_keeps_image_and_treats_table_as_fallible() -> None:
    suffix = build_visual_deplot_suffix(_deplot("Year | Value\n2019 | 70"))

    assert "full chart image" in suffix.lower()
    assert "DePlot" in suffix
    assert "fallible" in suffix.lower()
    assert "2019 | 70" in suffix


def test_recovery_prompt_contains_drafts_but_no_gold_fields() -> None:
    suffix = build_visual_recovery_suffix(
        deplot_value=_deplot("Year | Value\n2019 | 70"),
        base_output="Image draft says 70.",
        deplot_output="Table draft says 72.",
    )

    assert "full chart image" in suffix.lower()
    assert "Image draft says 70" in suffix
    assert "Table draft says 72" in suffix
    assert "2019 | 70" in suffix
    assert "Reference Answer" not in suffix
    assert "Verified Hint" not in suffix
    assert "oracle" not in suffix.lower()
    assert "correctness" not in suffix.lower()
