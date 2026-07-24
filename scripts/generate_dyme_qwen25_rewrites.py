#!/usr/bin/env python
"""Generate DyME-style Qwen2.5 rewritten training hints.

The script mirrors DyME's prerefine flow:
1. Convert the existing hint/context into structured facts with ``prompt_ic``.
2. Ask Qwen2.5 to rewrite the hint into a structured reasoning target.

It is resumable through a JSONL cache keyed by row index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.aokvqa.prompts import prompt_refine as aok_prompt_refine
from data_utils.chart.prompts import prompt_refine as chart_prompt_refine
from data_utils.commom_util import prompt_ic

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"

TEMPLATES = {
    "chart": """Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]
""",
    "gsm8k": """Goal: [State the main question to be answered in one simple sentence.]
Observation: [List the key numbers and relationships from the problem statement.]
Reasoning: [Show the step-by-step calculation process. Each step should be a clear mathematical operation.]
Conclusion: [State the final answer clearly.]
""",
    "aokvqa": """Goal: [State the visual/common-sense question to answer.]
Observation: [List the key visual facts and relevant context.]
Reasoning: [Explain how the facts eliminate alternatives or support the answer.]
Conclusion: [State the answer clearly.]
""",
}

SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS_NUMBER_WORDS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}

SYSTEM_PROMPTS = {
    "chart": "You are a seasoned professional in the field of chart analysis.",
    "gsm8k": (
        "You are a seasoned professional in mathematics. Return only the requested "
        "rewritten reasoning text, without extra commentary."
    ),
    "aokvqa": (
        "You are a seasoned professional in visual commonsense reasoning. Return only "
        "the requested rewritten reasoning text, without extra commentary."
    ),
}


def _task_prompt_refine(task: str) -> str:
    if task == "aokvqa":
        return aok_prompt_refine
    return chart_prompt_refine


def _int_to_words(value: int) -> str:
    if value < 20:
        return SMALL_NUMBER_WORDS[value]
    if value < 100:
        tens = value // 10 * 10
        remainder = value % 10
        return TENS_NUMBER_WORDS[tens] if remainder == 0 else f"{TENS_NUMBER_WORDS[tens]} {SMALL_NUMBER_WORDS[remainder]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        prefix = f"{SMALL_NUMBER_WORDS[hundreds]} hundred"
        return prefix if remainder == 0 else f"{prefix} {_int_to_words(remainder)}"
    if value < 1_000_000:
        thousands = value // 1000
        remainder = value % 1000
        prefix = f"{_int_to_words(thousands)} thousand"
        return prefix if remainder == 0 else f"{prefix} {_int_to_words(remainder)}"
    return str(value)


def _number_word_variants(answer: str) -> set[str]:
    if not re.fullmatch(r"\d+", answer):
        return set()
    value = int(answer)
    cardinal = _int_to_words(value)
    variants = {cardinal, cardinal.replace(" hundred ", " hundred and ")}
    if value in ORDINAL_WORDS:
        variants.add(ORDINAL_WORDS[value])
    elif 20 < value < 100 and value % 10:
        tens = value // 10 * 10
        ones = value % 10
        variants.add(f"{TENS_NUMBER_WORDS[tens]} {ORDINAL_WORDS[ones]}")
    return variants


def _extract_conclusion(text: str) -> str:
    if "Conclusion:" not in text:
        return text
    return text.rsplit("Conclusion:", 1)[-1]


def _decimal_or_none(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").strip().rstrip("%")).normalize()
    except (InvalidOperation, ValueError):
        return None


def answer_in_conclusion(text: str, answer: Any) -> bool:
    answer_text = str(answer or "").strip()
    if not answer_text:
        return True
    conclusion = _extract_conclusion(str(text or ""))
    normalized_answer = answer_text.replace(",", "").strip()
    normalized_conclusion = conclusion.replace(",", "")
    if re.search(rf"\bnot\s+\$?{re.escape(normalized_answer)}\b", normalized_conclusion, flags=re.IGNORECASE):
        return False
    answer_decimal = _decimal_or_none(normalized_answer)
    if answer_decimal is not None:
        for number in re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", conclusion):
            number_decimal = _decimal_or_none(number)
            if number_decimal is not None and number_decimal == answer_decimal:
                return True
    elif re.search(rf"\b{re.escape(normalized_answer.lower())}\b", conclusion.lower()):
        return True

    lowered = re.sub(r"[-_]+", " ", conclusion.lower())
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _number_word_variants(normalized_answer))


def needs_answer_retry(item: dict[str, Any] | None, *, task: str) -> bool:
    if item is None or task != "gsm8k":
        return False
    if item.get("dyme_rewrite", {}).get("status") != "ok":
        return False
    return not answer_in_conclusion(str(item.get("hint") or ""), item.get("answer"))


def _answer_repair_prompt(
    *,
    question: str,
    answer: str,
    source_context: str,
    flawed_rewrite: str,
    template: str,
) -> str:
    return f"""Rewrite the reasoning target for a GSM8K training example.

Requirements:
- Use the problem statement and original solution as the factual source.
- Correct any arithmetic or logic error in the flawed rewrite.
- The final Conclusion must explicitly include the exact reference answer: {answer}
- Return only the rewritten target text in the template format.
- Do not add notes, caveats, or extra commentary.

Question:
{question}

Reference answer:
{answer}

Original solution:
{source_context}

Flawed rewrite:
{flawed_rewrite}

Template:
{template}

Rewritten target:
"""


def _source_context(item: dict[str, Any], task: str) -> str:
    if task == "aokvqa":
        visual_fact = str(item.get("visual_fact") or "").strip()
        hint = str(item.get("hint") or "").strip()
        choices = item.get("choices")
        parts = []
        if visual_fact:
            parts.append(f"Visual facts:\n{visual_fact}")
        if choices:
            parts.append(f"Choices: {choices}")
        if hint:
            parts.append(f"Original rationale:\n{hint}")
        return "\n\n".join(parts).strip() or hint
    return str(item.get("hint") or "").strip()


def _is_valid_rewrite(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return bool(stripped) and "Goal:" in stripped and "Conclusion:" in stripped


def _fallback(item: dict[str, Any], *, task: str, model_id: str, started_at: float, error: str) -> dict[str, Any]:
    out = deepcopy(item)
    hint = str(item.get("hint") or "").strip()
    out["hint"] = hint
    out["dyme_rewrite"] = {
        "model": model_id,
        "task": task,
        "status": "fallback",
        "source_hint_chars": len(hint),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "error": error[:500],
    }
    return out


def rewrite_item(
    item: dict[str, Any],
    *,
    task: str,
    client: Any,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Rewrite one training item using DyME's two-step prerefine flow."""
    started_at = time.time()
    source = _source_context(item, task)
    if not source:
        return _fallback(item, task=task, model_id=model_id, started_at=started_at, error="empty source hint/context")

    try:
        if task == "aokvqa":
            ic_text = source
            structured_context_source = "aokvqa_visual_fact_direct"
        else:
            ic_text = client.get_completion(
                prompt_ic % source,
                system_prompt=SYSTEM_PROMPTS.get(task),
                max_tokens=5000,
            )
            structured_context_source = "prompt_ic"
            if not isinstance(ic_text, str) or not ic_text.strip():
                raise RuntimeError("empty structured context response")

        refiner_prompt = _task_prompt_refine(task) % (
            ic_text.strip(),
            item.get("question", ""),
            item.get("answer", ""),
            TEMPLATES[task],
        )
        rewritten_hint = client.get_completion(
            refiner_prompt,
            system_prompt=SYSTEM_PROMPTS.get(task),
            max_tokens=1000,
        )
        if not _is_valid_rewrite(rewritten_hint):
            raise RuntimeError(f"invalid rewritten hint: {str(rewritten_hint)[:160]!r}")
        answer_consistency_repaired = False
        if task == "gsm8k" and not answer_in_conclusion(rewritten_hint, item.get("answer")):
            repaired_hint = client.get_completion(
                _answer_repair_prompt(
                    question=str(item.get("question", "")),
                    answer=str(item.get("answer", "")),
                    source_context=source,
                    flawed_rewrite=rewritten_hint.strip(),
                    template=TEMPLATES[task],
                ),
                system_prompt=SYSTEM_PROMPTS.get(task),
                max_tokens=1000,
            )
            if not _is_valid_rewrite(repaired_hint) or not answer_in_conclusion(repaired_hint, item.get("answer")):
                raise RuntimeError(f"answer-inconsistent rewritten hint: {str(rewritten_hint)[:160]!r}")
            rewritten_hint = repaired_hint
            answer_consistency_repaired = True

        out = deepcopy(item)
        out["hint"] = rewritten_hint.strip()
        out["dyme_rewrite"] = {
            "model": model_id,
            "task": task,
            "status": "ok",
            "source_hint_chars": len(source),
            "structured_context_source": structured_context_source,
            "structured_context_chars": len(ic_text.strip()),
            "rewritten_hint_chars": len(out["hint"]),
            "answer_consistency_repaired": answer_consistency_repaired,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        return out
    except Exception as exc:
        return _fallback(item, task=task, model_id=model_id, started_at=started_at, error=str(exc))


def _client_config_for_worker(args: argparse.Namespace, worker_id: int) -> dict[str, Any]:
    port = args.init_port + (worker_id % args.num_server)
    return {
        "client_type": "openai",
        "api_key": args.api_key,
        "api_base": args.api_base % str(port) if "%s" in args.api_base else args.api_base,
        "timeout": args.timeout,
        "model_id": args.model_id,
        "init_port": args.init_port,
        "num_server": args.num_server,
    }


def _load_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cached[int(row["index"])] = row["item"]
    return cached


def is_cached_complete(item: dict[str, Any] | None, *, retry_fallback: bool) -> bool:
    if item is None:
        return False
    if not retry_fallback:
        return True
    return item.get("dyme_rewrite", {}).get("status") != "fallback"


def is_cached_complete_for_args(
    item: dict[str, Any] | None,
    *,
    task: str,
    retry_fallback: bool,
    retry_answer_mismatch: bool,
) -> bool:
    if not is_cached_complete(item, retry_fallback=retry_fallback):
        return False
    if retry_answer_mismatch and needs_answer_retry(item, task=task):
        return False
    return True


def _write_cache_record(path: Path, lock: Lock, index: int, item: dict[str, Any]) -> None:
    record = {"index": index, "item": item}
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _rewrite_index(args: argparse.Namespace, index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    from client_utils.openai_api import OpenAIClient

    client = OpenAIClient(_client_config_for_worker(args, index), max_retries=args.max_retries)
    return index, rewrite_item(item, task=args.task, client=client, model_id=args.model_id)


def generate_rewrites(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache or f"{args.output}.cache.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    cached = _load_cache(cache_path) if args.resume else {}
    rewritten: list[dict[str, Any] | None] = [
        cached.get(i)
        if is_cached_complete_for_args(
            cached.get(i),
            task=args.task,
            retry_fallback=args.retry_fallback,
            retry_answer_mismatch=args.retry_answer_mismatch,
        )
        else None
        for i in range(len(rows))
    ]
    pending = [(i, row) for i, row in enumerate(rows) if rewritten[i] is None]
    lock = Lock()

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_rewrite_index, args, i, row) for i, row in pending]
            completed = 0
            for future in as_completed(futures):
                index, item = future.result()
                rewritten[index] = item
                _write_cache_record(cache_path, lock, index, item)
                completed += 1
                if completed % args.log_every == 0 or completed == len(pending):
                    ok = sum(1 for row in rewritten if row and row.get("dyme_rewrite", {}).get("status") == "ok")
                    fallback = sum(
                        1 for row in rewritten if row and row.get("dyme_rewrite", {}).get("status") == "fallback"
                    )
                    print(
                        f"progress completed={completed}/{len(pending)} total={sum(row is not None for row in rewritten)}/{len(rows)} "
                        f"ok={ok} fallback={fallback}",
                        flush=True,
                    )

    final_rows = [row if row is not None else rows[i] for i, row in enumerate(rewritten)]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    ok = sum(1 for row in final_rows if row.get("dyme_rewrite", {}).get("status") == "ok")
    fallback = sum(1 for row in final_rows if row.get("dyme_rewrite", {}).get("status") == "fallback")
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "cache": str(cache_path),
        "task": args.task,
        "rows": len(final_rows),
        "ok": ok,
        "fallback": fallback,
        "model": args.model_id,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("chart", "gsm8k", "aokvqa"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-fallback", action="store_true")
    parser.add_argument("--retry-answer-mismatch", action="store_true")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--api-base", default="http://127.0.0.1:%s/v1")
    parser.add_argument("--init-port", type=int, default=23333)
    parser.add_argument("--num-server", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    generate_rewrites(parse_args(argv))


if __name__ == "__main__":
    main()
