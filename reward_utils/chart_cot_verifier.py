"""Deterministic verification helpers for structured ChartQA trajectories."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


_SECTION_NAMES = ("goal", "observation", "reasoning", "conclusion", "answer")
_SECTION_RE = re.compile(
    r"(?is)(?:^|\n)[ \t]*(goal|observation|reasoning|conclusion|answer)[ \t]*:[ \t]*"
    r"(.*?)(?=(?:\n\s*)(?:goal|observation|reasoning|conclusion|answer)\s*:|\Z)"
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


@dataclass(frozen=True)
class ParsedChartCoT:
    goal: str = ""
    observation: str = ""
    reasoning: str = ""
    conclusion: str = ""
    answer: str = ""
    structure_valid: bool = False
    missing_sections: tuple[str, ...] = ()
    empty_sections: tuple[str, ...] = ()
    duplicate_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedNumber:
    value: Decimal
    is_percent: bool = False
    raw: str = ""


@dataclass(frozen=True)
class ChartTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def cells(self) -> tuple[str, ...]:
        return self.columns + tuple(cell for row in self.rows for cell in row)


@dataclass(frozen=True)
class GroundedClaim:
    sentence: str
    label: str
    value: str
    status: str


@dataclass(frozen=True)
class ConsistencyResult:
    status: str
    conclusion_value: str = ""
    answer_value: str = ""


@dataclass(frozen=True)
class ReasoningCheck:
    kind: str
    status: str
    stated_value: str = ""
    expected_value: str = ""


@dataclass(frozen=True)
class ChartCoTVerification:
    quality: str
    answer_correct: bool
    parsed: ParsedChartCoT
    deplot_available: bool
    grounded_claims: tuple[GroundedClaim, ...]
    reasoning_checks: tuple[ReasoningCheck, ...]
    conclusion_answer: ConsistencyResult
    reason_codes: tuple[str, ...]
    verification_error: bool = False


def _decode_newlines(text: Any) -> str:
    return str(text or "").replace("\\r\\n", "\n").replace("\\n", "\n")


def parse_chart_cot(text: str) -> ParsedChartCoT:
    normalized = _decode_newlines(text)
    values: dict[str, list[str]] = {name: [] for name in _SECTION_NAMES}
    for match in _SECTION_RE.finditer(normalized):
        values[match.group(1).lower()].append(match.group(2).strip())

    missing = tuple(name for name in _SECTION_NAMES if not values[name])
    empty = tuple(name for name in _SECTION_NAMES if values[name] and not values[name][-1])
    duplicates = tuple(name for name in _SECTION_NAMES if len(values[name]) > 1)
    selected = {name: (values[name][-1] if values[name] else "") for name in _SECTION_NAMES}
    valid = not missing and not empty and not duplicates
    return ParsedChartCoT(
        **selected,
        structure_valid=valid,
        missing_sections=missing,
        empty_sections=empty,
        duplicate_sections=duplicates,
    )


def parse_number(value: Any) -> NormalizedNumber | None:
    raw = str(value or "").strip()
    match = _NUMBER_RE.fullmatch(raw)
    if match is None:
        return None
    is_percent = raw.endswith("%")
    numeric = raw[:-1] if is_percent else raw
    numeric = numeric.replace(",", "")
    try:
        return NormalizedNumber(value=Decimal(numeric), is_percent=is_percent, raw=raw)
    except InvalidOperation:
        return None


def numbers_equivalent(
    left: NormalizedNumber | None,
    right: NormalizedNumber | None,
    *,
    tolerance: Decimal = Decimal("0.0001"),
) -> bool:
    if left is None or right is None:
        return False
    left_value = left.value
    right_value = right.value
    scale = max(abs(left_value), abs(right_value), Decimal(1))
    if abs(left_value - right_value) <= tolerance * scale:
        return True
    if left.is_percent and not right.is_percent:
        left_value = left_value / Decimal(100)
    elif right.is_percent and not left.is_percent:
        right_value = right_value / Decimal(100)
    scale = max(abs(left_value), abs(right_value), Decimal(1))
    return abs(left_value - right_value) <= tolerance * scale


def _deplot_payload(raw: Any) -> tuple[str, str]:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return "", ""
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return "", text
        if not isinstance(decoded, dict):
            return "", text
        payload = decoded
    else:
        return "", ""
    source = str(payload.get("source") or "")
    table = payload.get("parsed_table") or ""
    return source, str(table)


def parse_deplot_table(raw: Any) -> ChartTable | None:
    source, table_text = _deplot_payload(raw)
    if source == "deplot_placeholder" or not table_text.strip():
        return None
    lines = [line.strip() for line in _decode_newlines(table_text).splitlines() if line.strip()]
    if not lines:
        return None
    columns = tuple(cell.strip() for cell in lines[0].split("|"))
    if not columns or not any(columns):
        return None
    width = len(columns)
    rows: list[tuple[str, ...]] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        elif len(cells) > width:
            cells = cells[:width]
        rows.append(tuple(cells))
    return ChartTable(columns=columns, rows=tuple(rows))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text or "") if part.strip()]


def _label_value_map(table: ChartTable) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for row in table.rows:
        if row and row[0].strip():
            collected.setdefault(row[0].strip(), []).extend(cell for cell in row[1:] if cell.strip())
    for column_idx, column in enumerate(table.columns[1:], start=1):
        if column.strip():
            collected.setdefault(column.strip(), []).extend(
                row[column_idx] for row in table.rows if column_idx < len(row) and row[column_idx].strip()
            )
    return {
        label: tuple(dict.fromkeys(values))
        for label, values in collected.items()
    }


def _number_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _NUMBER_RE.finditer(text or "")]


def verify_grounded_claims(text: str, table: ChartTable | None) -> tuple[GroundedClaim, ...]:
    if table is None:
        return tuple(
            GroundedClaim(sentence=sentence, label="", value=value, status="unknown")
            for sentence in _sentences(text)
            for value in _number_tokens(sentence)
        )

    label_values = _label_value_map(table)
    row_counts = Counter(row[0].strip() for row in table.rows if row and row[0].strip())
    ambiguous_labels = {
        label for label, count in row_counts.items() if count > 1
    }
    if len(table.rows) > 1:
        ambiguous_labels.update(column.strip() for column in table.columns[1:] if column.strip())
    ambiguous_labels.update(
        label for label, values in label_values.items() if len(values) != 1
    )
    claims: list[GroundedClaim] = []
    consumed_spans: set[tuple[str, int, int]] = set()
    for sentence in _sentences(text):
        derived_sentence = bool(
            re.search(
                r"(?i)\b(sum|difference|average|total|count|subtract|adding|added|divid|ratio)\b",
                sentence,
            )
        )
        if not derived_sentence:
            ordered_labels = sorted(
                label_values.items(),
                key=lambda item: (item[0] in ambiguous_labels, -len(item[0])),
            )
            for label, expected_values in ordered_labels:
                label_pattern = rf"(?<!\w){re.escape(label)}(?!\w)"
                binding_patterns = (
                    re.compile(rf"(?i){label_pattern}[ \t]*[:=][ \t]*({_NUMBER_RE.pattern})"),
                    re.compile(
                        rf"(?i){label_pattern}[^.;:]{{0,24}}?\b(?:is|was|equals?)\b[ \t]*({_NUMBER_RE.pattern})"
                    ),
                )
                bindings = [
                    match
                    for pattern in binding_patterns
                    for match in pattern.finditer(sentence)
                ]
                for binding in bindings:
                    claimed = binding.group(1)
                    start, end = binding.span(1)
                    if (sentence, start, end) in consumed_spans:
                        continue
                    supported = any(
                        numbers_equivalent(
                            parse_number(claimed),
                            parse_number(expected),
                            tolerance=Decimal("0.005"),
                        )
                        for expected in expected_values
                    )
                    status = (
                        "supported"
                        if supported
                        else "unknown"
                        if label in ambiguous_labels
                        else "contradicted"
                    )
                    claims.append(
                        GroundedClaim(
                            sentence=sentence,
                            label=label,
                            value=claimed,
                            status=status,
                        )
                    )
                    consumed_spans.add((sentence, start, end))

        for match in _NUMBER_RE.finditer(sentence):
            span = (sentence, match.start(), match.end())
            if span in consumed_spans:
                continue
            token = match.group(0)
            if any(token == label for label in label_values):
                continue
            claims.append(
                GroundedClaim(sentence=sentence, label="", value=token, status="unknown")
            )
    return tuple(claims)


def _normalize_text_answer(text: str) -> str:
    value = re.sub(r"(?i)^\s*(?:final\s+)?answer\s*:\s*", "", str(text or ""))
    value = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", value.lower())
    value = re.sub(r"\s*([/\\-])\s*", r"\1", value)
    value = re.sub(r"[\[\](),\"']+", " ", value)
    value = re.sub(r"\band\b", " ", value)
    return " ".join(value.split())


def verify_conclusion_answer_consistency(conclusion: str, answer: str) -> ConsistencyResult:
    answer_value = _normalize_text_answer(answer)
    if not answer_value:
        return ConsistencyResult(status="unknown")

    if answer_value in {"yes", "no"}:
        normalized_conclusion = _normalize_text_answer(conclusion)
        negative = bool(
            re.search(
                r"\b(?:not|incorrect|false|isn't|aren't|doesn't|didn't|cannot|can't|"
                r"different|differs?|unequal)\b",
                normalized_conclusion,
            )
        )
        if negative:
            return ConsistencyResult(
                status="consistent" if answer_value == "no" else "inconsistent",
                conclusion_value="no",
                answer_value=answer_value,
            )
        explicit = re.search(r"(?i)\b(yes|no)\b", conclusion or "")
        if explicit is not None:
            stated = explicit.group(1).lower()
        elif re.search(
            r"\b(?:the same|same as|equal(?:s| to)?|identical|matches?|indeed)\b",
            normalized_conclusion,
        ):
            stated = "yes"
        else:
            return ConsistencyResult(status="unknown", answer_value=answer_value)
        return ConsistencyResult(
            status="consistent" if stated == answer_value else "inconsistent",
            conclusion_value=stated,
            answer_value=answer_value,
        )

    conclusion_numbers = _number_tokens(conclusion)
    answer_numbers = _number_tokens(str(answer or ""))
    if answer_numbers:
        answer_token = answer_numbers[-1]
        if not conclusion_numbers:
            return ConsistencyResult(status="unknown", answer_value=answer_token)
        matching = next(
            (
                token
                for token in conclusion_numbers
                if numbers_equivalent(
                    parse_number(token),
                    parse_number(answer_token),
                    tolerance=Decimal("0.001"),
                )
            ),
            None,
        )
        if matching is not None:
            return ConsistencyResult(
                status="consistent",
                conclusion_value=matching,
                answer_value=answer_token,
            )
        if len(conclusion_numbers) > 1:
            return ConsistencyResult(status="unknown", answer_value=answer_token)
        conclusion_token = conclusion_numbers[0]
        return ConsistencyResult(
            status="inconsistent",
            conclusion_value=conclusion_token,
            answer_value=answer_token,
        )

    normalized_conclusion = _normalize_text_answer(conclusion)
    if not normalized_conclusion:
        return ConsistencyResult(status="unknown", answer_value=answer_value)
    contains = re.search(rf"(?<!\w){re.escape(answer_value)}(?!\w)", normalized_conclusion)
    return ConsistencyResult(
        status="consistent" if contains else "inconsistent",
        conclusion_value=answer_value if contains else normalized_conclusion,
        answer_value=answer_value,
    )


def _table_numeric_values(table: ChartTable | None) -> list[NormalizedNumber]:
    if table is None:
        return []
    values: list[NormalizedNumber] = []
    for row in table.rows:
        for cell in row[1:]:
            number = parse_number(cell)
            if number is not None:
                values.append(number)
    return values


def _number_text(number: Decimal) -> str:
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def verify_reasoning(reasoning: str, table: ChartTable | None) -> tuple[ReasoningCheck, ...]:
    text = reasoning or ""
    low = text.lower()
    table_values = _table_numeric_values(table)

    extrema_match = re.search(
        rf"(?i)\b(maximum|highest|largest|peak|minimum|lowest|smallest)\b"
        rf"[^.;:]{{0,20}}?\b(?:is|equals?)\b[ \t]*({_NUMBER_RE.pattern})",
        text,
    )
    if extrema_match and table_values and table is not None and len(table.columns) == 2:
        keyword = extrema_match.group(1).lower()
        extrema_kind = "maximum" if keyword in {"maximum", "highest", "largest", "peak"} else "minimum"
        stated = extrema_match.group(2)
        if stated:
            expected_num = (
                max(number.value for number in table_values)
                if extrema_kind == "maximum"
                else min(number.value for number in table_values)
            )
            expected = _number_text(expected_num)
            return (
                ReasoningCheck(
                    kind=extrema_kind,
                    status=(
                        "valid"
                        if numbers_equivalent(parse_number(stated), parse_number(expected))
                        else "invalid"
                    ),
                    stated_value=stated,
                    expected_value=expected,
                ),
            )

    threshold_match = re.search(
        r"(?i)\b(above|over|greater than|below|under|less than)\s+"
        r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)",
        text,
    )
    if threshold_match and table_values and re.search(r"(?i)\b(count|how many|number of)\b", text):
        threshold = parse_number(threshold_match.group(2))
        mentioned = _number_tokens(text)
        if threshold is not None and len(mentioned) >= 2:
            stated = mentioned[-1]
            if threshold_match.group(1).lower() in {"above", "over", "greater than"}:
                count = sum(number.value > threshold.value for number in table_values)
            else:
                count = sum(number.value < threshold.value for number in table_values)
            expected = str(count)
            return (
                ReasoningCheck(
                    kind="count_threshold",
                    status=(
                        "valid"
                        if numbers_equivalent(parse_number(stated), parse_number(expected))
                        else "invalid"
                    ),
                    stated_value=stated,
                    expected_value=expected,
                ),
            )

    arithmetic_patterns = (
        (
            "difference",
            re.compile(
                r"(?i)difference(?:\s+between)?\s+([-+]?\d+(?:\.\d+)?)\s+and\s+"
                r"([-+]?\d+(?:\.\d+)?).*?(?:is|equals?)\s+([-+]?\d+(?:\.\d+)?)"
            ),
            lambda a, b: abs(a - b),
        ),
        (
            "sum",
            re.compile(
                r"(?i)sum(?:\s+of)?\s+([-+]?\d+(?:\.\d+)?)\s+and\s+"
                r"([-+]?\d+(?:\.\d+)?).*?(?:is|equals?)\s+([-+]?\d+(?:\.\d+)?)"
            ),
            lambda a, b: a + b,
        ),
    )
    for kind, pattern, operation in arithmetic_patterns:
        match = pattern.search(text)
        if match:
            left, right, stated = (Decimal(match.group(i)) for i in (1, 2, 3))
            expected_num = operation(left, right)
            return (
                ReasoningCheck(
                    kind=kind,
                    status="valid" if stated == expected_num else "invalid",
                    stated_value=_number_text(stated),
                    expected_value=_number_text(expected_num),
                ),
            )

    comparison = re.search(
        r"(?i)([-+]?\d+(?:\.\d+)?)\s+is\s+"
        r"(greater than|less than|equal to)\s+([-+]?\d+(?:\.\d+)?)",
        text,
    )
    if comparison:
        left = Decimal(comparison.group(1))
        right = Decimal(comparison.group(3))
        relation = comparison.group(2).lower()
        valid = {
            "greater than": left > right,
            "less than": left < right,
            "equal to": left == right,
        }[relation]
        return (ReasoningCheck(kind="comparison", status="valid" if valid else "invalid"),)

    return (ReasoningCheck(kind="unparsed", status="unknown"),)


def verify_chart_cot_trajectory(
    response: str,
    deplot: Any,
    *,
    answer_correct: bool,
) -> ChartCoTVerification:
    parsed = parse_chart_cot(response)
    table = parse_deplot_table(deplot)
    claims = verify_grounded_claims(parsed.observation, table)
    reasoning_checks = verify_reasoning(parsed.reasoning, table)
    consistency = verify_conclusion_answer_consistency(parsed.conclusion, parsed.answer)

    reason_codes: list[str] = []
    if not answer_correct:
        reason_codes.append("answer_incorrect")
    if not parsed.structure_valid:
        reason_codes.append("structure_invalid")
    if any(claim.status == "contradicted" for claim in claims):
        reason_codes.append("grounding_contradiction")
    if any(check.status == "invalid" for check in reasoning_checks):
        reason_codes.append("reasoning_invalid")
    if consistency.status == "inconsistent":
        reason_codes.append("conclusion_answer_inconsistent")
    elif consistency.status == "unknown":
        reason_codes.append("conclusion_answer_unknown")

    if any(
        code in reason_codes
        for code in (
            "grounding_contradiction",
            "reasoning_invalid",
            "conclusion_answer_inconsistent",
        )
    ):
        quality = "Q0"
    elif not answer_correct or not parsed.structure_valid:
        quality = "Q1"
    elif consistency.status != "consistent":
        quality = "Q2"
    else:
        quality = "Q3"

    return ChartCoTVerification(
        quality=quality,
        answer_correct=bool(answer_correct),
        parsed=parsed,
        deplot_available=table is not None,
        grounded_claims=claims,
        reasoning_checks=reasoning_checks,
        conclusion_answer=consistency,
        reason_codes=tuple(reason_codes),
    )


def verifier_error_result(response: str, *, answer_correct: bool) -> ChartCoTVerification:
    try:
        parsed = parse_chart_cot(response)
    except Exception:
        parsed = ParsedChartCoT()
    return ChartCoTVerification(
        quality="Q2",
        answer_correct=bool(answer_correct),
        parsed=parsed,
        deplot_available=False,
        grounded_claims=(),
        reasoning_checks=(ReasoningCheck(kind="verifier_error", status="unknown"),),
        conclusion_answer=ConsistencyResult(status="unknown"),
        reason_codes=("verifier_error",),
        verification_error=True,
    )


def normalize_reasoning_template(reasoning: str, table: ChartTable | None = None) -> str:
    text = str(reasoning or "").lower()
    labels: set[str] = set()
    if table is not None:
        labels.update(column.strip().lower() for column in table.columns if column.strip())
        labels.update(row[0].strip().lower() for row in table.rows if row and row[0].strip())
    for label in sorted(labels, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(label)}(?!\w)", "<LABEL>", text)
    text = _NUMBER_RE.sub("<NUM>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_template_diversity(templates: list[str], *, top_k: int = 20) -> dict[str, Any]:
    total = len(templates)
    counts = Counter(templates)
    dominant = max(counts.values(), default=0)
    return {
        "count": total,
        "unique_count": len(counts),
        "unique_template_rate": len(counts) / max(total, 1),
        "dominant_template_rate": dominant / max(total, 1),
        "top_templates": [
            {"template": template, "count": count, "rate": count / max(total, 1)}
            for template, count in counts.most_common(top_k)
        ],
    }
