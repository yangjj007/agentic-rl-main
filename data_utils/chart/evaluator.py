from dataclasses import dataclass
import re
import string
from typing import Optional


def _normalize_string(s):
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(unnormalized_string: str) -> str:
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


class RelaxedCorrectness:
    """Relaxed correctness metrics.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."
    """

    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str) -> tuple[float | None, bool]:
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                value = float(text.rstrip("%"))
                return value, is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(char.isdigit() for char in text):
                return _normalize_string(text)
            else:
                return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def calculate_relative_change(prediction: float, target: float) -> float:
            return abs(prediction - target) / max(abs(target), 1e-10)

        def _compare_numeric_values(
            prediction: float, target: float, max_relative_change: float
        ) -> float:
            relative_change = calculate_relative_change(prediction, target)
            return 1.0 if relative_change <= max_relative_change else 0.0

        def _compare_text_values(prediction: str, target: str) -> float:
            while prediction and prediction[-1] in string.punctuation:
                prediction = prediction[:-1]
            return 1.0 if prediction.lower() == target.lower() else 0.0

        def _to_decimal(value: float, is_percent: bool) -> float:
            return value / 100 if is_percent else value

        def _compare_numeric_with_percent(
            prediction: float,
            prediction_is_percent: bool,
            target: float,
            target_is_percent: bool,
            max_relative_change: float,
        ) -> float:
            # Compare as-is
            value = _compare_numeric_values(prediction, target, max_relative_change)

            # If not equal and one is percent, try other comparisons
            if value != 1.0 and (prediction_is_percent or target_is_percent):
                value = max(
                    value,
                    _compare_numeric_values(
                        _to_decimal(prediction, prediction_is_percent),
                        target,
                        max_relative_change,
                    ),
                    _compare_numeric_values(
                        prediction,
                        _to_decimal(target, target_is_percent),
                        max_relative_change,
                    ),
                )
            return value

        prediction = _preprocess_text(prediction)
        prediction_float, prediction_is_percent = _to_float(prediction)

        value_list = []
        for target in targets:
            target = _preprocess_text(target)
            target_float, target_is_percent = _to_float(target)

            if prediction_float is not None and target_float is not None:
                # Compare as numeric values
                value = _compare_numeric_with_percent(
                    prediction_float,
                    prediction_is_percent,
                    target_float,
                    target_is_percent,
                    max_relative_change,
                )
            elif _is_letter(target) and len(prediction) > 0:
                # Compare as multiple choice options: take first letter from prediction
                value = 1.0 if prediction[0].lower() == target.lower() else 0.0
            else:
                # Compare as text values
                value = _compare_text_values(prediction, target)

            value_list.append(value)

        return max(value_list)

    def score(self, model_answer: str, reference_answer: str | list[str], max_relative_change=0.05) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        return self._relaxed_correctness(model_answer, reference_answer, max_relative_change)


class ExplicitPromptRelaxedCorrectness(RelaxedCorrectness):
    """Relaxed correctness for explicit prompt."""

    @property
    def name(self) -> str:
        return "explicit_prompt_relaxed_correctness"

    def _get_final_answer(self, generation: str) -> str:
        def _find_last_occurrence(pattern: str, string: str):
            return string.rfind(pattern)

        # Strip extraneous markdown around the answer:
        generation = re.sub(r"([aA]nswer)\**:\**", "\\1:", generation)

        final_answer_index = _find_last_occurrence("answer:", generation.lower())

        if final_answer_index != -1:
            # Find the start of the answer (after "final answer:")
            start_index = final_answer_index + len("answer:")

            # Split the remaining text into lines
            lines = generation[start_index:].split("\n")

            # Find the first non-empty line
            final_answer = next((line.strip() for line in lines if line.strip()), "")

            # Remove any markdown formatting
            final_answer = re.sub(r"[*_\[\]\(\)]", "", final_answer)

            return final_answer
        else:
            return ""

    def score(self, model_answer: str, reference_answer: str | list[str], max_relative_change=0.05) -> float:
        parsed_model_answer = self._get_final_answer(model_answer)
        if not parsed_model_answer:
            # Parsing failed.
            return 0.0
        return super().score(parsed_model_answer, reference_answer, max_relative_change)

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def eval_one_chart(
    model_answer: str,
    reference_answer: str | list[str],
    max_relative_change: float = 0.05,
    answer_flag = 'answer:'
) -> float:
    model_answer = model_answer.strip()
    reference_answer = reference_answer.strip()
    reference_answer = reference_answer.lower().replace(answer_flag, '')
    if answer_flag not in model_answer.lower():
        # If the answer is not in the model answer, we can use the relaxed correctness.
        return relaxed_correctness(model_answer, reference_answer, max_relative_change)
    """Evaluate one chart."""
    # return relaxed_correctness(model_answer, reference_answer)
    evaluator = ExplicitPromptRelaxedCorrectness()
    return evaluator.score(model_answer, reference_answer, max_relative_change)


@dataclass(frozen=True)
class TeacherProbeAnswer:
    answer: str
    has_answer_flag: bool
    parse_failed: bool


def _clean_probe_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[*_\[\]\(\)]", "", text).strip()
    text = _normalize_string(text)
    return _remove_end_punctuation(text).strip()


def _strip_probe_answer_flag(text: str, answer_flag: str) -> str:
    text = str(text or "").strip()
    flag = (answer_flag or "answer:").strip().lower()
    if flag and text.lower().startswith(flag):
        return text[len(flag) :].strip()
    return text


def _normalize_teacher_probe_score_text(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_probe_answer(text)).strip()


_TREND_CANONICAL = {
    "decrease": "decreasing",
    "decreased": "decreasing",
    "decreases": "decreasing",
    "decline": "decreasing",
    "declined": "decreasing",
    "declines": "decreasing",
    "fall": "decreasing",
    "falls": "decreasing",
    "fell": "decreasing",
    "increase": "increasing",
    "increased": "increasing",
    "increases": "increasing",
    "rise": "increasing",
    "rises": "increasing",
    "rising": "increasing",
    "grew": "increasing",
    "growth": "increasing",
}


def _teacher_probe_score_variants(text: str) -> list[str]:
    normalized = _normalize_teacher_probe_score_text(text)
    variants = [normalized] if normalized else []
    if not normalized:
        return variants

    lower = normalized.lower()
    trend = _TREND_CANONICAL.get(lower)
    if trend and trend not in [item.lower() for item in variants]:
        variants.append(trend)

    numeric_prefix = re.match(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?%?)\b", normalized)
    if numeric_prefix and re.search(r"[A-Za-z]", normalized[numeric_prefix.end() :]):
        value = numeric_prefix.group(1).replace(",", "")
        if value and value not in variants:
            variants.append(value)
    return variants


def _reference_list_items(
    reference_answer: str | list[str],
    *,
    answer_flag: str,
) -> list[str] | None:
    if isinstance(reference_answer, list):
        items = [
            _normalize_teacher_probe_score_text(
                _strip_probe_answer_flag(str(ref), answer_flag)
            )
            for ref in reference_answer
        ]
        items = [item for item in items if item]
        return items if len(items) > 1 else None

    raw = _strip_probe_answer_flag(str(reference_answer), answer_flag).strip()
    if raw.startswith("[") and raw.endswith("]"):
        items = [
            _normalize_teacher_probe_score_text(part)
            for part in raw[1:-1].split(",")
            if part.strip()
        ]
        return items if len(items) > 1 else None
    return None


def _prediction_list_items(prediction: str) -> list[str]:
    normalized = _normalize_teacher_probe_score_text(prediction)
    comma_form = re.sub(r"(?i)\s+\band\b\s+", ",", normalized)
    items = [
        _normalize_teacher_probe_score_text(part)
        for part in comma_form.split(",")
        if part.strip()
    ]
    return items if len(items) > 1 else [normalized] if normalized else []


def _score_teacher_probe_list_answer(
    prediction: str,
    targets: list[str],
    max_relative_change: float,
) -> float:
    predictions = _prediction_list_items(prediction)
    if len(predictions) != len(targets):
        return 0.0

    evaluator = RelaxedCorrectness()
    unused_prediction_indices = set(range(len(predictions)))
    for target in targets:
        matched_idx = None
        for idx in sorted(unused_prediction_indices):
            if evaluator.score(predictions[idx], target, max_relative_change) == 1.0:
                matched_idx = idx
                break
        if matched_idx is None:
            return 0.0
        unused_prediction_indices.remove(matched_idx)
    return 1.0


def parse_teacher_probe_answer(
    model_answer: str,
    *,
    answer_flag: str = "answer:",
) -> TeacherProbeAnswer:
    """Extract the short final answer from a teacher probe generation."""
    text = str(model_answer or "").strip()
    if not text:
        return TeacherProbeAnswer("", False, True)

    flag = (answer_flag or "answer:").strip().lower()
    lower = text.lower()
    flag_idx = lower.rfind(flag)
    if flag_idx >= 0:
        after = text[flag_idx + len(flag) :]
        lines = [line.strip() for line in after.splitlines() if line.strip()]
        answer = _clean_probe_answer(lines[0] if lines else "")
        return TeacherProbeAnswer(answer, True, not bool(answer))

    phrase_match = re.search(
        r"(?is)\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*([^\n.]+%?)",
        text,
    )
    if phrase_match:
        answer = _clean_probe_answer(phrase_match.group(1))
        return TeacherProbeAnswer(answer, False, not bool(answer))

    one_line = " ".join(text.split())
    answer = _clean_probe_answer(one_line)
    lower_answer = answer.lower()
    word_count = len(re.findall(r"[A-Za-z]+", answer))
    looks_like_failure = bool(
        re.search(
            r"\b(need|information|chart|cannot|can't|unable|unknown|insufficient)\b",
            lower_answer,
        )
    )
    if (
        answer
        and len(answer) <= 64
        and "\n" not in text
        and word_count <= 4
        and not looks_like_failure
    ):
        return TeacherProbeAnswer(answer, False, False)

    return TeacherProbeAnswer("", False, True)


def eval_teacher_probe_chart(
    model_answer: str,
    reference_answer: str | list[str],
    max_relative_change: float = 0.05,
    *,
    answer_flag: str = "answer:",
) -> tuple[float, TeacherProbeAnswer]:
    parsed = parse_teacher_probe_answer(model_answer, answer_flag=answer_flag)
    if parsed.parse_failed:
        return 0.0, parsed

    list_targets = _reference_list_items(reference_answer, answer_flag=answer_flag)
    if list_targets is not None:
        return (
            _score_teacher_probe_list_answer(
                parsed.answer,
                list_targets,
                max_relative_change,
            ),
            parsed,
        )

    if isinstance(reference_answer, list):
        references = [
            _normalize_teacher_probe_score_text(
                _strip_probe_answer_flag(str(ref), answer_flag)
            )
            for ref in reference_answer
        ]
    else:
        references = _normalize_teacher_probe_score_text(
            _strip_probe_answer_flag(str(reference_answer), answer_flag)
        )

    evaluator = RelaxedCorrectness()
    prediction_variants = _teacher_probe_score_variants(parsed.answer)
    if isinstance(references, list):
        reference_variants: list[str] = []
        for ref in references:
            reference_variants.extend(_teacher_probe_score_variants(ref))
        references_for_score = reference_variants or references
    else:
        references_for_score = _teacher_probe_score_variants(references) or references
    score = max(
        (
            evaluator.score(prediction, references_for_score, max_relative_change)
            for prediction in prediction_variants
        ),
        default=0.0,
    )
    return float(score), parsed

if __name__ == "__main__":
    # Example usage
    model_answer = "The reasoning above leads to the following answer: 0.009"
    score = eval_one_chart('2009', '2010', 0.05)
    print(f"Score: {score}")
