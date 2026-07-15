"""Gold-hidden ChartQA evidence parsing, validation, and routing."""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

from data_utils.chart.deplot_pipeline import format_deplot_for_teacher
from opsd_utils.evidence_harness.contracts import (
    EvidenceAction,
    EvidenceCandidate,
    HarnessDecision,
    HarnessStatus,
    ValidationResult,
)


_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.*?)\s*$")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_EQUATION_EXPR_TAIL_RE = re.compile(r"[-+()\d.\s*/]+$")
_EQUATION_RESULT_HEAD_RE = re.compile(r"\s*(?P<result>[-+]?\d+(?:\.\d+)?)")
_MAX_DERIVATION_LINES = 80
_MAX_EQUATION_LINE_CHARS = 512
_BARE_ANSWER_REJECT_WORDS = {
    "answer",
    "cannot",
    "determine",
    "final",
    "line",
    "maybe",
    "unsure",
}
_VISUAL_QUARANTINE_TERMS = (
    "blue",
    "green",
    "orange",
    "red",
    "grey",
    "gray",
    "pink",
    "purple",
    "yellow",
    "intersect",
    "intersection",
    "cross",
    "y-axis",
    "x-axis",
    "axis tick",
    "tick interval",
    "slice",
    "pie",
)
_TARGET_EVIDENCE_STOPWORDS = {
    "a",
    "about",
    "all",
    "and",
    "are",
    "bar",
    "bars",
    "by",
    "chart",
    "color",
    "does",
    "for",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "line",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "to",
    "value",
    "was",
    "what",
    "whats",
    "which",
    "who",
    "with",
}


def _clean_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_answer(value: Any) -> str:
    text = _clean_answer(value).lower().rstrip(".")
    numeric = _as_number(text)
    if numeric is not None:
        return _format_number(numeric)
    return text


def _as_number(value: Any) -> float | None:
    text = _clean_answer(value).replace(",", "").rstrip("%")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _single_line_short_answer(lines: list[str]) -> str | None:
    if len(lines) != 1:
        return None
    text = _clean_answer(lines[0])
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(text.split()) > 5 or len(text) > 80:
        return None
    if text.endswith((".", "!", "?")) and _as_number(text) is None:
        return None
    if any(word in _BARE_ANSWER_REJECT_WORDS for word in words):
        return None
    return text or None


def build_chartqa_candidate(
    *,
    attempt_id: str,
    action: EvidenceAction,
    output: str,
) -> EvidenceCandidate:
    raw_output = str(output or "")
    matches = _ANSWER_LINE_RE.findall(raw_output)
    answer = _clean_answer(matches[-1]) if matches else None
    if answer is None:
        nonempty = [line.strip() for line in raw_output.splitlines() if line.strip()]
        answer = _single_line_short_answer(nonempty)
    if answer == "":
        answer = None
    return EvidenceCandidate(
        attempt_id=attempt_id,
        action=action,
        answer=answer,
        raw_output=raw_output,
        parse_failed=answer is None,
        unresolved_refs=("final_answer",) if answer is None else (),
    )


def _parsed_table(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("parsed_table") or "").strip()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            return str(payload.get("parsed_table") or "").strip()
    return ""


def _table_cells(value: Any) -> tuple[str, ...]:
    table = _parsed_table(value)
    cells: list[str] = []
    for line in table.splitlines():
        for cell in line.split("|"):
            cleaned = _clean_answer(cell)
            if cleaned:
                cells.append(cleaned)
    return tuple(cells)


def _target_tokens(value: Any) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", str(value or "").lower())
        if token and token not in _TARGET_EVIDENCE_STOPWORDS and len(token) > 1
    }
    return tokens


def _table_lines(table: str) -> list[str]:
    return [_clean_answer(line) for line in str(table or "").splitlines() if _clean_answer(line)]


def _split_table_line(line: str) -> list[str]:
    return [_clean_answer(cell) for cell in str(line or "").split("|") if _clean_answer(cell)]


def _table_header_and_rows(table: str) -> tuple[list[str], list[list[str]]]:
    lines = [_split_table_line(line) for line in _table_lines(table)]
    lines = [cells for cells in lines if cells]
    if not lines:
        return [], []
    header = lines[0]
    return header, lines[1:]


def _iter_numeric_table_cells(table: str) -> list[dict[str, Any]]:
    header, rows = _table_header_and_rows(table)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        row_label = row[0]
        for cell_idx, cell in enumerate(row[1:], start=1):
            column_label = header[cell_idx] if cell_idx < len(header) else f"column_{cell_idx + 1}"
            for number in _NUMBER_RE.findall(cell):
                value = _as_number(number)
                if value is None:
                    continue
                records.append(
                    {
                        "row_label": row_label,
                        "column_label": column_label,
                        "cell": cell,
                        "value": value,
                        "line": " | ".join(row),
                    }
                )
    return records


def _threshold_condition(question: str):
    match = re.search(
        r"(?i)\b(above|over|greater than|more than|at least|below|under|less than|at most)\s+([-+]?\d+(?:\.\d+)?)",
        str(question or ""),
    )
    if not match:
        return None
    operator = match.group(1).lower()
    threshold = float(match.group(2))

    def passes(value: float) -> bool:
        if operator in {"above", "over", "greater than", "more than"}:
            return value > threshold
        if operator in {"at least"}:
            return value >= threshold
        if operator in {"below", "under", "less than"}:
            return value < threshold
        return value <= threshold

    symbol = ">" if operator in {"above", "over", "greater than", "more than"} else "<"
    if operator == "at least":
        symbol = ">="
    elif operator == "at most":
        symbol = "<="
    return symbol, threshold, passes


def _question_matched_rows(question: str, table: str, *, max_rows: int = 5) -> list[str]:
    question_tokens = _target_tokens(question)
    if not question_tokens:
        return []
    scored: list[tuple[int, int, str]] = []
    question_lower = str(question or "").lower()
    for idx, line in enumerate(_table_lines(table)):
        cells = [_clean_answer(cell) for cell in line.split("|") if _clean_answer(cell)]
        row_tokens = _target_tokens(" ".join(cells))
        overlap = len(question_tokens & row_tokens)
        phrase_bonus = sum(
            2
            for cell in cells
            if len(cell) > 2 and re.search(r"[A-Za-z]", cell) and cell.lower() in question_lower
        )
        score = overlap + phrase_bonus
        if score > 0:
            scored.append((score, -idx, line))
    scored.sort(reverse=True)
    return [line for _score, _neg_idx, line in scored[:max_rows]]


def _threshold_candidate_rows(question: str, table: str, *, max_rows: int = 8) -> tuple[str, list[str]] | None:
    condition = _threshold_condition(question)
    if condition is None:
        return None
    symbol, threshold, passes = condition

    rows: list[str] = []
    for line in _table_lines(table):
        values: list[float | None] = []
        cells = [_clean_answer(cell) for cell in line.split("|") if _clean_answer(cell)]
        for cell_idx, cell in enumerate(cells):
            for number in _NUMBER_RE.findall(cell):
                value = _as_number(number)
                if cell_idx == 0 and value is not None and 1800 <= value <= 2100:
                    continue
                values.append(value)
        if any(value is not None and passes(value) for value in values):
            rows.append(line)
            if len(rows) >= max_rows:
                break
    return f"{symbol} {_format_number(threshold)}", rows


def _recovered_target_evidence(question: str, table: str) -> str:
    sections: list[str] = []
    threshold = _threshold_candidate_rows(question, table)
    if threshold is not None:
        label, rows = threshold
        if rows:
            sections.append("Threshold rows: " + label + "\n" + "\n".join(rows))
    matched_rows = _question_matched_rows(question, table)
    if matched_rows:
        sections.append("Question-matched rows:\n" + "\n".join(matched_rows))
    return "\n\n".join(sections).strip()


def _target_phrase_from_row(row: str) -> str:
    header_like = {
        "characteristic",
        "country",
        "entity",
        "percent",
        "percentage",
        "response",
        "value",
        "year",
    }
    for cell in (_clean_answer(part) for part in str(row or "").split("|")):
        if not cell:
            continue
        if cell.lower() in header_like:
            continue
        if _as_number(cell) is not None:
            continue
        if re.search(r"[A-Za-z]", cell):
            return cell
    return ""


def build_chartqa_target_phrase_response_prefix(question: str, deplot_value: Any) -> str:
    raw_table = _parsed_table(deplot_value)
    for row in _question_matched_rows(question, raw_table, max_rows=3):
        phrase = _target_phrase_from_row(row)
        if phrase:
            return f"Target phrase: {phrase}\nEvidence:"
    return "Target phrase:"


def _format_numeric_cell(record: dict[str, Any]) -> str:
    return (
        f"{record['row_label']} | {record['column_label']} = "
        f"{_format_number(float(record['value']))}"
    )


def _question_numeric_targets(question: str) -> list[float]:
    values: list[float] = []
    for raw in _NUMBER_RE.findall(str(question or "")):
        value = _as_number(raw)
        if value is None:
            continue
        if 1800 <= value <= 2100:
            continue
        values.append(value)
    return values


def _near_equal(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-6)


def _best_named_record(
    phrase: str,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target_tokens = _target_tokens(phrase)
    if not target_tokens:
        return None
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        label_tokens = _target_tokens(
            f"{record['row_label']} {record['column_label']}"
        )
        overlap = len(target_tokens & label_tokens)
        if overlap:
            scored.append((overlap, -idx, record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2] if scored else None


def _minus_record(question: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    match = re.search(r"(?i)\bminus\b(.+?)(?:\s+equals?\b|\?|$)", str(question or ""))
    if not match:
        return None
    return _best_named_record(match.group(1), records)


def _year_targets(question: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", str(question or "")))


def _row_numeric_values(row: list[str]) -> list[float]:
    values: list[float] = []
    for cell in row[1:]:
        for number in _NUMBER_RE.findall(cell):
            value = _as_number(number)
            if value is not None:
                values.append(value)
    return values


def _executable_deplot_recovery_block(question: str, table: str) -> str:
    question_lower = str(question or "").lower()
    header, rows = _table_header_and_rows(table)
    records = _iter_numeric_table_cells(table)
    operation = "unresolved"
    matched: list[dict[str, Any]] = []
    candidate_answer = "unresolved"
    subtracted_record: dict[str, Any] | None = None

    condition = _threshold_condition(question)
    if condition is not None:
        _symbol, _threshold, passes = condition
        matched = [record for record in records if passes(float(record["value"]))]
        if matched and "sum" in question_lower:
            subtracted_record = _minus_record(question, records)
            if subtracted_record is not None:
                operation = "threshold_sum_minus_label"
                candidate_answer = _format_number(
                    sum(float(record["value"]) for record in matched)
                    - float(subtracted_record["value"])
                )
            else:
                operation = "threshold_sum"
                candidate_answer = _format_number(sum(float(record["value"]) for record in matched))
        elif matched and ("how many" in question_lower or "number of" in question_lower):
            operation = "threshold_count"
            candidate_answer = str(len(matched))
        elif matched and "which" in question_lower:
            operation = "threshold_label_lookup"
            if len(matched) == 1:
                record = matched[0]
                candidate_answer = f"[{record['row_label']}, {record['column_label']}]"
            else:
                candidate_answer = "; ".join(
                    f"[{record['row_label']}, {record['column_label']}]" for record in matched
                )
    elif "how many" in question_lower and "increase" in question_lower and "do not increase" in question_lower:
        increase_column = next(
            (
                column
                for column in header
                if "increase" in column.lower() and "do not" not in column.lower()
            ),
            None,
        )
        no_increase_column = next(
            (column for column in header if "do not increase" in column.lower()),
            None,
        )
        if increase_column and no_increase_column:
            by_row: dict[str, dict[str, dict[str, Any]]] = {}
            for record in records:
                by_row.setdefault(str(record["row_label"]), {})[
                    str(record["column_label"])
                ] = record
            matched = []
            for row_records in by_row.values():
                left = row_records.get(no_increase_column)
                right = row_records.get(increase_column)
                if left and right and float(right["value"]) > float(left["value"]):
                    matched.extend([left, right])
            operation = "column_comparison_count"
            candidate_answer = str(len(matched) // 2)
    elif "how many" in question_lower:
        targets = _question_numeric_targets(question)
        if targets:
            target = targets[-1]
            matched = [
                record for record in records if _near_equal(float(record["value"]), target)
            ]
            if matched:
                operation = "exact_value_count"
                candidate_answer = str(len(matched))
    elif "median" in question_lower and records:
        values = sorted(float(record["value"]) for record in records)
        midpoint = len(values) // 2
        if len(values) % 2:
            median = values[midpoint]
        else:
            median = (values[midpoint - 1] + values[midpoint]) / 2.0
        operation = "median_all_values"
        candidate_answer = _format_number(median)
        matched = records
    elif "which two" in question_lower and "total of" in question_lower:
        targets = _question_numeric_targets(question)
        if targets:
            target = targets[-1]
            for left_idx, left in enumerate(records):
                for right in records[left_idx + 1 :]:
                    if left["row_label"] == right["row_label"]:
                        continue
                    if _near_equal(float(left["value"]) + float(right["value"]), target):
                        operation = "pair_sum_label_lookup"
                        matched = [left, right]
                        candidate_answer = (
                            f"[{left['row_label']}, {right['row_label']}]"
                        )
                        break
                if candidate_answer != "unresolved":
                    break
    elif "which two" in question_lower and "same" in question_lower:
        year_targets = _year_targets(question)
        year_column = next(
            (year for year in year_targets if year in header),
            None,
        )
        if year_column is not None:
            same_value_groups: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                if record["column_label"] != year_column:
                    continue
                key = _format_number(float(record["value"]))
                same_value_groups.setdefault(key, []).append(record)
            for group in same_value_groups.values():
                if len(group) >= 2:
                    group = sorted(group, key=lambda record: str(record["row_label"]))
                    operation = "same_value_pair_lookup"
                    matched = group[:2]
                    candidate_answer = (
                        f"[{matched[0]['row_label']}, {matched[1]['row_label']}]"
                    )
                    break
    elif "most drastic change" in question_lower and records:
        first_column = header[1] if len(header) > 1 else None
        ordered = [
            record
            for record in records
            if first_column is None or record["column_label"] == first_column
        ]
        if len(ordered) >= 2:
            changes = [
                (
                    abs(float(ordered[idx]["value"]) - float(ordered[idx + 1]["value"])),
                    ordered[idx],
                    ordered[idx + 1],
                )
                for idx in range(len(ordered) - 1)
            ]
            _change, earlier, later = max(changes, key=lambda item: item[0])
            operation = "max_consecutive_change"
            matched = [earlier, later]
            candidate_answer = str(earlier["row_label"])
    elif records:
        targets = _question_numeric_targets(question)
        if len(targets) >= 2 and ("category" in question_lower or "find" in question_lower):
            for row in rows:
                values = _row_numeric_values(row)
                if all(any(_near_equal(value, target) for value in values) for target in targets):
                    operation = "value_signature_lookup"
                    candidate_answer = row[0]
                    matched = [
                        record for record in records if record["row_label"] == row[0]
                    ]
                    break
        elif targets:
            target = targets[-1]
            matched = [
                record for record in records if _near_equal(float(record["value"]), target)
            ]
            if matched and (
                "category" in question_lower
                or "what is that" in question_lower
                or _year_targets(question)
            ):
                operation = "exact_value_label_lookup"
                candidate = matched[0]
                candidate_answer = str(candidate["column_label"])

    lines = [
        "[Executable DePlot Recovery]",
        f"Operation: {operation}",
    ]
    if matched:
        lines.append("Matched cells:")
        lines.extend(f"- {_format_numeric_cell(record)}" for record in matched[:12])
    else:
        lines.append("Matched cells: none")
    if subtracted_record is not None:
        lines.append(f"Subtracted cell: {_format_numeric_cell(subtracted_record)}")
    lines.append(f"Candidate answer: {candidate_answer}")
    return "\n".join(lines)


def build_chartqa_executable_deplot_recovery_suffix(question: str, deplot_value: Any) -> str:
    raw_table = _parsed_table(deplot_value)
    table = format_deplot_for_teacher(deplot_value) or raw_table
    recovery_block = _executable_deplot_recovery_block(question, raw_table)
    return (
        "Solve the ChartQA question using the DePlot table and the executable DePlot "
        "recovery block below. This is a table-only recovery operator: the block is "
        "computed only from the visible DePlot table, not from any reference answer or "
        "hidden hint. If the "
        "operation and matched cells answer the question, return the Candidate answer as "
        "the final short answer. If the block is unresolved or not applicable, "
        "use the DePlot table to repair it. Do not use any hidden hint or "
        "reference answer.\n\n"
        "Return only the single short answer after the provided Answer: prefix.\n\n"
        f"{recovery_block}\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )


def build_chartqa_executable_deplot_response_prefix(question: str, deplot_value: Any) -> str:
    raw_table = _parsed_table(deplot_value)
    recovery_block = _executable_deplot_recovery_block(question, raw_table)
    match = re.search(r"(?m)^Candidate answer:\s*(.*?)\s*$", recovery_block)
    candidate = _clean_answer(match.group(1)) if match else ""
    if candidate and candidate.lower() != "unresolved":
        return f"Answer: {candidate}"
    return "Answer:"


def build_chartqa_scale_unit_recovery_suffix(
    *,
    deplot_value: Any,
    attempts: list[dict[str, Any]],
) -> str:
    table = format_deplot_for_teacher(deplot_value) or _parsed_table(deplot_value)
    attempt_lines: list[str] = []
    for idx, attempt in enumerate(attempts[-6:], start=1):
        action = _clean_answer(attempt.get("action") or f"attempt_{idx}")
        output = _clean_answer(attempt.get("teacher_output") or "")
        if len(output) > 500:
            output = output[:500].rstrip() + " ..."
        if output:
            attempt_lines.append(f"{idx}. {action}: {output}")
    prior_block = "\n".join(attempt_lines) or "No prior usable attempt."
    return (
        "Repair only answer-surface scale or unit mistakes from the prior teacher attempts. "
        "Do not redo broad chart reasoning. Check whether a prior numeric answer is supported "
        "but missing a percent sign, using the wrong percent-vs-decimal scale, or carrying an "
        "unwanted label. If the chart/table describes percentages, shares, or rates and a prior "
        "answer is a bare percent value such as 8.86, prefer returning it with a percent sign "
        "as 8.86%. If no scale or unit repair is justified, return the best short answer from "
        "the prior attempts. Do not use any hidden hint or reference answer.\n\n"
        "Return only the single repaired answer after the provided Answer: prefix.\n\n"
        f"[Prior Teacher Attempts]\n{prior_block}\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )


def _safe_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        left = _safe_arithmetic(node.left)
        right = _safe_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero")
        return left / right
    raise ValueError("unsupported arithmetic expression")


def _iter_equation_candidates(output: str):
    for line in str(output or "").splitlines()[:_MAX_DERIVATION_LINES]:
        if "=" not in line or len(line) > _MAX_EQUATION_LINE_CHARS:
            continue
        parts = line.split("=")
        for idx in range(1, len(parts)):
            expression_match = _EQUATION_EXPR_TAIL_RE.search(parts[idx - 1])
            result_match = _EQUATION_RESULT_HEAD_RE.match(parts[idx])
            if not expression_match or not result_match:
                continue
            expression = expression_match.group(0).strip()
            if len(_NUMBER_RE.findall(expression)) < 2:
                continue
            yield expression, result_match.group("result")


def _derived_value(output: str, cells: tuple[str, ...]) -> str | None:
    cell_numbers = {
        _format_number(number)
        for number in (_as_number(cell) for cell in cells)
        if number is not None
    }
    for expression, result in _iter_equation_candidates(output):
        operands = {
            _format_number(float(number))
            for number in _NUMBER_RE.findall(expression)
        }
        if operands and not operands.issubset(cell_numbers):
            continue
        try:
            computed = _safe_arithmetic(ast.parse(expression, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        stated = _as_number(result)
        if stated is None or not math.isclose(computed, stated, rel_tol=1e-6, abs_tol=1e-6):
            continue
        return _format_number(computed)
    return None


def validate_chartqa_candidate(
    candidate: EvidenceCandidate,
    deplot_value: Any,
) -> ValidationResult:
    if candidate.parse_failed or candidate.answer is None:
        return ValidationResult(
            validator_id="chartqa_deplot",
            status="FAIL",
            reason_code="answer_parse_failure",
        )

    cells = _table_cells(deplot_value)
    answer_norm = _normalized_answer(candidate.answer)
    for cell in cells:
        if _normalized_answer(cell) == answer_norm:
            return ValidationResult(
                validator_id="chartqa_deplot",
                status="PASS",
                reason_code="deplot_direct_support",
                supporting_refs=(cell,),
                deterministic_value=answer_norm,
            )

    derived = _derived_value(candidate.raw_output, cells)
    if derived is not None and derived == answer_norm:
        return ValidationResult(
            validator_id="chartqa_deplot",
            status="PASS",
            reason_code="deplot_derived_support",
            supporting_refs=tuple(cells),
            deterministic_value=derived,
        )

    return ValidationResult(
        validator_id="chartqa_deplot",
        status="UNKNOWN",
        reason_code="deplot_support_unresolved",
    )


def decide_after_parallel_attempts(
    base: EvidenceCandidate,
    deplot: EvidenceCandidate,
    *,
    max_attempts: int = 3,
) -> HarnessDecision:
    remaining = max(0, int(max_attempts) - 2)
    if not base.parse_failed and not deplot.parse_failed:
        if _normalized_answer(base.answer) == _normalized_answer(deplot.answer):
            return HarnessDecision(
                status=HarnessStatus.ACCEPTED,
                selected_attempt_id=deplot.attempt_id,
                reason_code="cross_attempt_agreement",
                remaining_budget=remaining,
            )
        reason = "candidate_conflict"
    else:
        reason = "candidate_parse_failure"

    if remaining > 0:
        return HarnessDecision(
            status=HarnessStatus.ACTIVE,
            next_action=EvidenceAction.VISUAL_RECOVERY,
            reason_code=reason,
            remaining_budget=remaining,
        )
    return HarnessDecision(
        status=HarnessStatus.BUDGET_EXHAUSTED,
        reason_code=reason,
        remaining_budget=0,
    )


def decide_after_recovery(
    recovery: EvidenceCandidate,
    *,
    base: EvidenceCandidate | None = None,
    deplot: EvidenceCandidate | None = None,
    validation: ValidationResult | None = None,
    max_attempts: int = 3,
) -> HarnessDecision:
    remaining = max(0, int(max_attempts) - 3)
    if recovery.parse_failed or recovery.answer is None:
        return HarnessDecision(
            status=HarnessStatus.ABSTAINED,
            reason_code="recovery_parse_failure",
            remaining_budget=remaining,
        )

    recovery_answer = _normalized_answer(recovery.answer)
    base_answer = None
    if base is not None and not base.parse_failed and base.answer is not None:
        base_answer = _normalized_answer(base.answer)
    if base_answer is not None and recovery_answer == base_answer:
        return HarnessDecision(
            status=HarnessStatus.ACCEPTED,
            selected_attempt_id=recovery.attempt_id,
            reason_code="recovery_confirms_visual",
            remaining_budget=remaining,
        )
    return HarnessDecision(
        status=HarnessStatus.ABSTAINED,
        reason_code=(
            "recovery_does_not_confirm_visual"
            if deplot is not None and not deplot.parse_failed
            else "recovery_unverified"
        ),
        remaining_budget=remaining,
    )


def build_visual_base_suffix() -> str:
    return (
        "Solve the question by inspecting the full chart image. Identify the relevant "
        "labels, series, and values, perform any necessary calculation, and cross-check "
        "the requested unit. Keep the reasoning concise. The final non-empty line must "
        "be exactly: Answer: <single short answer>"
    )


def build_visual_deplot_suffix(deplot_value: Any) -> str:
    table = format_deplot_for_teacher(deplot_value) or _parsed_table(deplot_value)
    return (
        "Solve the question using the full chart image and the auxiliary DePlot table "
        "below. DePlot is fallible OCR evidence: verify ambiguous labels, colors, series, "
        "and values against the full chart image. Check row/column orientation and show "
        "any necessary arithmetic. The final non-empty line must be exactly: "
        "Answer: <single short answer>\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )


def build_visual_recovery_suffix(
    *,
    deplot_value: Any,
    base_output: str,
    deplot_output: str,
) -> str:
    table = format_deplot_for_teacher(deplot_value) or _parsed_table(deplot_value)
    return (
        "Resolve the disagreement or missing answer between two teacher attempts. Inspect "
        "the full chart image again and use the DePlot table only as fallible auxiliary "
        "evidence. Check legend/color grounding, row/column alignment, selected values, "
        "and arithmetic. Do not copy either draft without verifying it. Keep the reasoning "
        "concise. The final non-empty line must be exactly: Answer: <single short answer>\n\n"
        f"[Visual Attempt]\n{str(base_output).strip()}\n\n"
        f"[Visual + DePlot Attempt]\n{str(deplot_output).strip()}\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )


def is_chartqa_visual_quarantine_question(question: str) -> bool:
    text = str(question or "").lower()
    return any(term in text for term in _VISUAL_QUARANTINE_TERMS)


def build_chartqa_arithmetic_recovery_suffix(deplot_value: Any) -> str:
    table = format_deplot_for_teacher(deplot_value) or _parsed_table(deplot_value)
    return (
        "Solve the ChartQA question using the full chart image and the DePlot table below. "
        "DePlot is fallible OCR evidence, but if the required values are present in the "
        "table, make the calculation explicit so it can be checked. Do not use any hidden "
        "hint or reference answer. If the question depends mainly on colors, legend "
        "grounding, axis tick spacing, or visual intersections, inspect the image and avoid "
        "inventing unsupported table values.\n\n"
        "Return exactly these lines:\n"
        "Operands: <values copied from chart/DePlot cells>\n"
        "Operation: <lookup|sum|difference|ratio|average|median|count|extreme|other>\n"
        "Equation: <arithmetic expression using copied numeric operands, or none>\n"
        "Answer: <single short answer>\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )


def build_chartqa_target_phrase_recovery_suffix(question: str, deplot_value: Any) -> str:
    raw_table = _parsed_table(deplot_value)
    table = format_deplot_for_teacher(deplot_value) or raw_table
    recovered_evidence = _recovered_target_evidence(question, raw_table)
    recovered_block = (
        f"\n\n[Recovered Candidate Evidence]\n{recovered_evidence}"
        if recovered_evidence
        else ""
    )
    return (
        "Solve the ChartQA question by grounding the exact requested target before reading "
        "a value. First identify the Target phrase from the question: the entity, category, "
        "legend/color, series, time point, or threshold condition being asked about. Do not "
        "answer with the largest, most salient, or previously mentioned chart item unless it "
        "matches that exact target. For color or legend questions, inspect the full image to "
        "map the legend/color to the right label before using the DePlot table. For threshold "
        "questions, return all requested labels that satisfy the condition. For percentage "
        "questions, return the displayed numeric value for the requested target and include "
        "a percent sign if the chart axis or slice is in percent. Do not use any hidden hint "
        "or reference answer.\n\n"
        "Return exactly these lines:\n"
        "Target phrase: <literal target from the question>\n"
        "Evidence: <matched chart/DePlot label and value>\n"
        "Answer: <single short answer>\n\n"
        "Prefer [Recovered Candidate Evidence] when it matches the question, then cross-check "
        "against the full DePlot table and image."
        f"{recovered_block}\n\n"
        f"[Visual Facts - DePlot]\n{table}"
    )
