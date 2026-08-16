import re
from typing import Any

from PIL import Image

from data_utils.chart.deplot_pipeline import format_deplot_for_teacher, is_deplot_placeholder
from data_utils.privileged_schema import parse_visual_fact
from opsd_utils.privileged.base import PrivilegedContextProvider
from opsd_utils.privileged.image_utils import heuristic_crop_from_visual_fact, load_rgb


DEFAULT_FORMAT_ONLY_HINT = (
    "Use the following structure in your response:\n"
    "Goal: ...\nObservation: ...\nReasoning: ...\nAnswer: ..."
)

CHARTQA_SHORT_ANSWER_HINT = (
    "Answer the chart question using only the chart image and provided visual evidence. "
    "Keep reasoning concise. Finish with exactly one final line:\n"
    "Answer: <short answer>"
)

# This is intentionally separate from ``CHARTQA_SHORT_ANSWER_HINT``. A
# teacher answer probe must be terse for reliable answer parsing, while a
# trajectory target needs an auditable chain of observations and reasoning.
# Neither prompt contains a gold answer or a gold rationale.
CHARTQA_STRUCTURED_TRAJECTORY_HINT = (
    "Write an evidence-grounded solution to the chart question. Use only the "
    "chart image and the provided DePlot table as evidence; do not assume any "
    "reference answer or hidden annotation. Do not transcribe the full table. "
    "Every numeric observation must be supported by the image or DePlot. In the "
    "Observation, cite at least two explicit `label: value` pairs from the DePlot "
    "table when it contains two or more rows; this evidence must be relevant to "
    "the requested comparison. "
    "Your response must contain exactly these five non-empty headings in this "
    "order:\n"
    "Goal: ...\n"
    "Observation: ...\n"
    "Reasoning: ...\n"
    "Conclusion: ...\n"
    "Answer: <short answer>"
)

# Used only for a no-gold retry after the deterministic verifier finds that a
# otherwise well-formed trajectory failed to ground its Observation in DePlot.
# It deliberately names no row, value, answer, hint, or reference annotation:
# the table already supplied by ``DeplotOnlyProvider`` remains the sole extra
# evidence.  The wording is intentionally concrete because some instruction-
# tuned teachers otherwise restate the final answer as an "observation".
CHARTQA_STRUCTURED_TRAJECTORY_EVIDENCE_RETRY_HINT = (
    "Regenerate a complete evidence-grounded solution to the chart question. "
    "Use only the chart image and the provided DePlot table; do not use or "
    "assume any reference answer, hint, or hidden annotation. This is a "
    "quality retry because an earlier draft did not cite table evidence. "
    "In Observation, copy at least two exact `label: value` facts from distinct "
    "rows or series in the DePlot table, then use those facts in Reasoning. "
    "Do not replace them with an abstract statement such as 'the lowest value "
    "is ...'. Do not transcribe the entire table. Your response must contain "
    "exactly these five non-empty headings in this order:\n"
    "Goal: ...\n"
    "Observation: ...\n"
    "Reasoning: ...\n"
    "Conclusion: ...\n"
    "Answer: <short answer>"
)


CHARTQA_ORACLE_HINT = (
    "You are running an ORACLE-HINT evidence-anchored ChartQA teacher probe.\n"
    "Use the chart image and DePlot table as visual evidence, but they are not "
    "the authority for the final answer.\n"
    "Hard priority order: Reference Answer > Verified Hint > DePlot consistency "
    "check > chart image.\n"
    "DePlot may contain OCR or table errors. Use it only to check or phrase the "
    "Observation when it supports the verified hint. If it conflicts, ignore it.\n"
    "Every output is invalid unless it contains exactly these five headings in "
    "this order: Goal:, Observation:, Reasoning:, Conclusion:, Answer:.\n"
    "Do not output a short answer only. Do not transcribe the chart or DePlot "
    "table. Do not include markdown tables.\n\n"
    "Example from training data:\n"
    "Goal: Find the lowest value of the red graph.\n"
    "Observation: The data for the 'Rep/Lean Rep' category across the years are: "
    "2018: 72, 2019: 70, and 2020: 77.\n"
    "Reasoning: Comparing the values, the minimum value is 70.\n"
    "Conclusion: The lowest value of the red graph is 70.\n"
    "Answer: 70\n\n"
    "Example from training data:\n"
    "Goal: Determine the number of years covered by the line graph.\n"
    "Observation: The data shows values for different years: 2008: 28, 2009: 91, "
    "2010: 97, 2011: 105, 2012: 115, 2013: 123, and 2014: 137.\n"
    "Reasoning: Counting the distinct years in the dataset provides the total "
    "number of years covered.\n"
    "Conclusion: The line graph covers 7 years.\n"
    "Answer: 7"
)

TEACHER_RESPONSE_PREFIX_MARKER = "[Teacher Response Prefix]"

_HINT_SECTION_RE = re.compile(
    r"(?is)(goal|observation|reasoning|conclusion)\s*:\s*(.*?)(?=(?:\n\s*)?(?:goal|observation|reasoning|conclusion)\s*:|$)"
)


def _extract_hint_sections(hint: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in _HINT_SECTION_RE.finditer(hint or ""):
        key = match.group(1).lower()
        text = " ".join(match.group(2).strip().split())
        if text:
            sections[key] = text
    return sections


def _clean_answer_text(answer: Any) -> str:
    text = str(answer or "").strip()
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    return text


def split_teacher_response_prefix(suffix: str) -> tuple[str, str]:
    """Split optional assistant prefill text from provider suffix."""
    text = str(suffix or "")
    if TEACHER_RESPONSE_PREFIX_MARKER not in text:
        return text, ""
    before, after = text.split(TEACHER_RESPONSE_PREFIX_MARKER, 1)
    return before.rstrip(), after.strip()


def _deplot_status(sample: dict[str, Any]) -> str:
    raw = sample.get("visual_fact_deplot")
    if not str(raw or "").strip():
        return "missing"
    if is_deplot_placeholder(raw):
        return "placeholder"
    if format_deplot_for_teacher(raw).strip():
        return "real"
    return "unknown"


def teacher_probe_evidence_status(
    sample: dict[str, Any],
    provider_names: list[str],
) -> dict[str, Any]:
    """Describe whether a teacher-probe candidate has clean extra evidence."""
    providers = set(provider_names or [])
    deplot_status = _deplot_status(sample)
    uses_deplot = bool({"visual_facts_deplot", "visual_facts"} & providers)
    visual_fact_used = bool("visual_facts" in providers) and bool(
        str(sample.get("visual_fact") or sample.get("visual_facts") or "").strip()
        or str(sample.get("visual_fact_hint") or "").strip()
    )
    crop_used = bool("crop" in providers) and bool(sample.get("image"))

    clean_evidence_present = bool(
        (uses_deplot and deplot_status == "real") or crop_used
    )
    evidence_present = clean_evidence_present or visual_fact_used
    return {
        "evidence_present": evidence_present,
        "clean_evidence_present": clean_evidence_present,
        "deplot_status": deplot_status,
        "deplot_real": deplot_status == "real",
        "deplot_placeholder": deplot_status == "placeholder",
        "visual_fact_used": visual_fact_used,
        "crop_used": crop_used,
    }


class FormatOnlyProvider(PrivilegedContextProvider):
    """Structure hint only — no gold answer or reference reasoning (anti-leakage)."""

    def __init__(self, hint_text: str | None = None):
        self._hint_text = (hint_text or DEFAULT_FORMAT_ONLY_HINT).strip()

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        return self._hint_text


class TextProvider(PrivilegedContextProvider):
    def __init__(self, include_gold: bool = True):
        self.include_gold = include_gold

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        if not self.include_gold:
            return ""
        parts = []
        hint = (sample.get("hint") or "").strip()
        answer = (sample.get("answer") or "").strip()
        if hint:
            parts.append(f"[Reference Reasoning]\n{hint}")
        if answer:
            parts.append(f"[Reference Answer]\n{answer}")
        return "\n\n".join(parts)


class OracleHintProvider(PrivilegedContextProvider):
    """Evidence-anchored oracle: keep image/DePlot and prefill the teacher format."""

    @staticmethod
    def _answer_line(answer: Any) -> str:
        text = _clean_answer_text(answer)
        return f"Answer: {text}" if text else ""

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        hint = str(sample.get("hint") or sample.get("visual_fact_hint") or sample.get("visual_fact") or "").strip()
        answer_text = _clean_answer_text(sample.get("answer"))
        sections = _extract_hint_sections(hint)
        goal = sections.get("goal") or "Answer the chart question using the verified hint."
        observation = sections.get("observation") or "Use the verified hint as the authoritative observation."
        reasoning = sections.get("reasoning") or "Follow the verified hint and check DePlot only as supporting evidence."
        conclusion = sections.get("conclusion") or (
            f"The reference answer is {answer_text}."
            if answer_text
            else "Use the verified reference answer."
        )

        response_prefix = "\n".join(
            [
                f"Goal: {goal}",
                f"Observation: {observation}",
                f"Reasoning: {reasoning}",
                f"Conclusion: {conclusion}",
                "Answer:",
            ]
        )

        parts = [
            "[Oracle-Hint Evidence Contract]",
            "Reference Answer is authoritative. Verified Hint is authoritative reasoning style.",
            "DePlot and image are supporting visual context only; never let DePlot override Reference Answer.",
            "Do not transcribe the DePlot table. Do not output a short answer only.",
            "Your answer must continue the assistant prefix and complete the final Answer line.",
        ]
        if hint:
            parts.append(f"[Verified Hint]\n{hint}")
        if answer_text:
            parts.append(f"[Reference Answer]\n{answer_text}")
            parts.append(
                "[Final Hard Rule]\n"
                "The final non-empty line must be exactly:\n"
                f"Answer: {answer_text}"
            )
        parts.append(
            "[Output Rules]\n"
            "Use exactly five headings in order: Goal:, Observation:, Reasoning:, Conclusion:, Answer:.\n"
            "Begin your next message with exactly: Goal:\n"
            "Do not transcribe the chart or DePlot table."
        )
        parts.append(f"{TEACHER_RESPONSE_PREFIX_MARKER}\n{response_prefix}")
        return "\n\n".join(parts)


class DeplotOnlyProvider(PrivilegedContextProvider):
    """F2 only: offline DePlot table from visual_fact_deplot — no hint/CoT (anti-leakage)."""

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        deplot_vf = sample.get("visual_fact_deplot")
        if deplot_vf and not is_deplot_placeholder(deplot_vf):
            text = format_deplot_for_teacher(deplot_vf)
            if text:
                return f"[Visual Facts - DePlot]\n{text}"
        return ""


class VisualFactsProvider(PrivilegedContextProvider):
    """B1: raw JSON visual facts; F1+F2 merge hint and deplot sources."""

    def _collect_visual_fact_parts(self, sample: dict[str, Any]) -> list[str]:
        parts: list[str] = []
        hint_vf = sample.get("visual_fact_hint")
        if hint_vf:
            text = parse_visual_fact(hint_vf)
            if text:
                parts.append(f"[Visual Facts - Hint]\n{text}")

        deplot_vf = sample.get("visual_fact_deplot")
        if deplot_vf and not is_deplot_placeholder(deplot_vf):
            text = format_deplot_for_teacher(deplot_vf)
            if text:
                parts.append(f"[Visual Facts - DePlot]\n{text}")

        primary = sample.get("visual_fact") or sample.get("visual_facts")
        if primary and not (hint_vf or deplot_vf):
            text = parse_visual_fact(primary)
            if text:
                parts.append(f"[Visual Facts]\n{text}")
        elif primary and (hint_vf or deplot_vf):
            text = parse_visual_fact(primary)
            if text:
                parts.append(f"[Visual Facts - Combined]\n{text}")

        return parts

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        parts = self._collect_visual_fact_parts(sample)
        return "\n\n".join(parts)


class CropProvider(PrivilegedContextProvider):
    """Returns evidence crop as second teacher image (dual-image path uses image_utils)."""

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        return ""

    def build_teacher_images(self, sample: dict[str, Any], crop_cfg: dict[str, Any] | None = None) -> list[Image.Image]:
        image = sample.get("image")
        if image is None:
            return []
        full = load_rgb(image)
        if full is None:
            return []
        crop, _, _ = heuristic_crop_from_visual_fact(full, sample, crop_cfg)
        return [crop]


class HybridProvider(PrivilegedContextProvider):
    def __init__(
        self,
        provider_names: list[str],
        crop_cfg: dict[str, Any] | None = None,
        *,
        text_include_gold: bool = True,
        format_only_hint: str | None = None,
    ):
        self._providers: list[PrivilegedContextProvider] = []
        self._crop_cfg = crop_cfg or {}
        for name in provider_names:
            if name == "text":
                self._providers.append(TextProvider(include_gold=text_include_gold))
            elif name == "format_only":
                self._providers.append(FormatOnlyProvider(format_only_hint))
            elif name == "oracle_hint":
                self._providers.append(OracleHintProvider())
            elif name == "visual_facts":
                self._providers.append(VisualFactsProvider())
            elif name == "visual_facts_deplot":
                self._providers.append(DeplotOnlyProvider())
            elif name == "crop":
                self._providers.append(CropProvider())

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        chunks = [p.build_teacher_suffix(sample) for p in self._providers]
        chunks = [c for c in chunks if c.strip()]
        return "\n\n".join(chunks)

    def build_teacher_images(self, sample: dict[str, Any]) -> list[Image.Image]:
        for p in self._providers:
            if isinstance(p, CropProvider):
                imgs = p.build_teacher_images(sample, self._crop_cfg)
                if imgs:
                    return imgs
        return []
