"""7B teacher Visual Checker for thinking reward."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from data_utils.chart.prompts import prompt_template, prompt_thinking_reward
from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.checker import RewardCalculatorLocal
from reward_utils.template_pool import TemplatePool, _comparison_prompt, is_valid_reasoning_template
from reward_utils.teacher_generate import (
    TeacherGenerateRequest,
    teacher_generate_batched_chunks,
    teacher_generate_one,
)
from reward_utils.visual_batch_ops import prefetch_ic_unique
from reward_utils.visual_ic import extract_visual_facts_teacher, ic_text_from_offline_sample


def build_image_primary_checker_prompt(
    *,
    question: str,
    answer: str,
    reasoning: str,
    student_answer: str = "",
    aux_evidence: str = "",
    aux_mode: str = "none",
    has_answer_flag: Optional[bool] = None,
) -> str:
    """Build the strict image-grounded checker prompt.

    The chart image is passed separately through the multimodal teacher request.
    Any textual extraction is opt-in and explicitly marked as noisy.
    """
    prompt = f"""You are a strict chart reasoning judge. The attached chart image is the only authoritative visual source.
Judge whether the student's reasoning is grounded in the image and supports the reference answer.
Return exactly one token: high, medium, or low. No explanation.

Question:
{question}

Reference answer:
{answer}

Student final answer:
{student_answer or "[missing]"}

Required final answer marker present:
{"unknown" if has_answer_flag is None else ("yes" if has_answer_flag else "no")}

Student reasoning:
{reasoning}
"""
    if aux_mode == "deplot_noisy" and aux_evidence:
        prompt += f"""
Optional noisy extracted text:
{aux_evidence}

This text may be incomplete or wrong. Ignore it whenever it conflicts with the image.
"""
    prompt += """
Rubric:
high = A complete reasoning chain uses visible chart evidence, cites or compares the relevant values/categories correctly, and explicitly supports both the student's final answer and the reference answer. Do not use high for terse answer fragments.
medium = The reasoning is mostly grounded in the image and supports the answer, but evidence is incomplete, vague, partially checked, the student's final answer is only partially supported, or the required final answer marker is missing.
low = The reasoning does not use visible chart evidence, fabricates or misreads chart data, is logically inconsistent with the answer, the student's final answer conflicts with the image/reference answer, is empty, lacks a clear reasoning chain, or only repeats/guesses an answer fragment.

Return one token only.
"""
    return prompt


_ANSWER_FRAGMENT_WORD_LIMIT = 10
_REASONING_MARKERS = (
    "because",
    "therefore",
    "reason",
    "shows",
    "shown",
    "chart",
    "graph",
    "compare",
    "compared",
    "higher",
    "lower",
    "largest",
    "smallest",
    "highest",
    "lowest",
    "increase",
    "decrease",
    "difference",
    "sum",
    "total",
    "average",
    "mode",
)


def _looks_like_answer_fragment(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    low = stripped.lower()
    words = re.findall(r"[a-zA-Z0-9]+", low)
    if len(words) > _ANSWER_FRAGMENT_WORD_LIMIT:
        return False
    if any(marker in low for marker in _REASONING_MARKERS):
        return False
    return True


def _split_response_parts(response: str, answer_flag: str) -> tuple[str, str, bool]:
    text = response or ""
    parts = re.split(f"(?i){re.escape(answer_flag)}", text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), True
    return text.strip(), "", False


def _postprocess_checker_label(
    *,
    score: float,
    label: str,
    reasoning: str,
    has_answer_flag: bool,
    student_answer_correct: Optional[bool] = None,
) -> tuple[float, str, str]:
    if _looks_like_answer_fragment(reasoning):
        if student_answer_correct is True:
            return 0.5, "medium", "correct_answer_fragment"
        if score > 0.0 or label != "low":
            return 0.0, "low", "answer_fragment"
        return score, label, ""
    if student_answer_correct is False and score > 0.5:
        return 0.5, "medium", "answer_incorrect_high_cap"
    if not has_answer_flag and score > 0.5:
        return 0.5, "medium", "missing_answer_flag_high_cap"
    return score, label, ""


def _score_from_label(text: str) -> tuple[float, str]:
    low = (text or "").strip().lower()
    if "high" in low:
        return 1.0, "high"
    if "medium" in low:
        return 0.5, "medium"
    if "low" in low:
        return 0.0, "low"
    return 0.0, "unknown"


def _has_image(image: Any) -> bool:
    if image is None:
        return False
    if isinstance(image, str) and not image.strip():
        return False
    return True


def _chart_system_prompt() -> str:
    return (
        "You are a seasoned professional in the field of chart analysis, demonstrating "
        "exceptional expertise and insight into complex chart data. Your output should be "
        "only judgement, without any additional text or explanation."
    )


@dataclass
class _CheckerJob:
    sample_idx: int
    response: str
    question: str
    answer: str
    hint: str
    thinking_part: str
    student_answer: str
    has_answer_flag: bool
    student_answer_correct: Optional[bool]


class TeacherVisualChecker(RewardCalculatorLocal):
    """Teacher-backed thinking reward; format/answer rewards stay local."""

    requires_sequential = True

    def __init__(
        self,
        RL_CONFIG,
        CLIENT_CONFIG,
        gpu_id=0,
        *,
        visual_config: Optional[dict] = None,
        template_pool: Optional[TemplatePool] = None,
        local_fallback: Optional[RewardCalculatorLocal] = None,
    ):
        super().__init__(RL_CONFIG, CLIENT_CONFIG, gpu_id=gpu_id)
        self.visual_config = visual_config or {}
        checker_cfg = self.visual_config.get("checker", {})
        self._enabled = bool(checker_cfg.get("enabled", True))
        self._max_per_batch = int(checker_cfg.get("max_per_batch", 0) or 0)
        self._grounding = str(checker_cfg.get("grounding", "image_primary") or "image_primary").lower()
        self._aux_evidence_mode = str(
            checker_cfg.get("aux_evidence", checker_cfg.get("aux", "none")) or "none"
        ).lower()
        self._ic_source = self.visual_config.get("ic_source", "teacher_image")
        self._max_ic_tokens = int(checker_cfg.get("max_ic_tokens", 768))
        self._max_score_tokens = int(checker_cfg.get("max_score_tokens", 16))
        self._max_template_tokens = int(checker_cfg.get("max_template_tokens", 512))
        self._teacher_batch_size = int(self.visual_config.get("teacher_batch_size", 4))
        self.template_pool = template_pool or TemplatePool(
            template_path=self.visual_config.get("template_pool", {}).get("path", "best_template.txt"),
            refresh_interval_sec=float(
                self.visual_config.get("template_pool", {}).get("refresh_interval_sec", 60)
            ),
        )
        self._local_fallback = local_fallback or RewardCalculatorLocal(RL_CONFIG, CLIENT_CONFIG, gpu_id)
        self._teacher_model = None
        self._processor = None
        self._recorder: Optional[VisualBatchRecorder] = None
        self._ic_cache: dict[tuple[str, str], str] = {}
        self._batch_samples: list[dict] = []
        self._batch_images: list[Any] = []
        self._batch_questions: list[str] = []
        self._teacher_budget_used = 0
        self._prefetch_ic = bool(self.visual_config.get("prefetch_ic", True))
        self._thinking_score_cache: dict[int, float] = {}

    def _uses_legacy_ic_prompt(self) -> bool:
        return self._grounding in ("ic_text", "legacy_ic", "text")

    def _uses_aux_evidence(self) -> bool:
        return self._aux_evidence_mode == "deplot_noisy"

    def bind_teacher(self, teacher_model, processor) -> None:
        self._teacher_model = teacher_model
        self._processor = processor

    def begin_generate_batch(
        self,
        *,
        samples: list[dict],
        images: list[Any],
        questions: list[str],
        global_step: int,
        output_dir: str,
    ) -> None:
        log_cfg = self.visual_config.get("logging", {})
        self._recorder = VisualBatchRecorder(
            global_step=global_step,
            output_dir=output_dir,
            log_cfg=log_cfg,
        )
        self._batch_samples = samples
        self._batch_images = images
        self._batch_questions = questions
        self._ic_cache = {}
        self._teacher_budget_used = 0
        self._thinking_score_cache = {}
        if self._prefetch_ic and (self._uses_legacy_ic_prompt() or self._uses_aux_evidence()):
            prefetch_ic_unique(
                teacher_model=self._teacher_model,
                processor=self._processor,
                samples=samples,
                images=images,
                questions=questions,
                ic_source=self._ic_source,
                max_new_tokens=self._max_ic_tokens,
                cache=self._ic_cache,
                recorder=self._recorder,
                teacher_batch_size=self._teacher_batch_size,
            )

    def end_generate_batch(self) -> dict[str, Any]:
        if self._recorder is None:
            return {}
        stats = self._recorder.finish()
        self._recorder = None
        self._batch_samples = []
        self._thinking_score_cache = {}
        return stats

    def _use_teacher_for_idx(self, idx: int) -> bool:
        if not self._enabled or self._teacher_model is None:
            return False
        if self._max_per_batch > 0 and self._teacher_budget_used >= self._max_per_batch:
            return False
        return True

    def _collect_checker_job(
        self,
        sample_idx: int,
        response: str,
        question: str,
        answer: str,
        hint: str,
        task: str,
    ) -> Optional[_CheckerJob]:
        thinking_part, student_answer, has_answer_flag = _split_response_parts(response, self.answer_flag)
        student_answer_correct: Optional[bool] = None
        if "chart" in task:
            student_answer_correct = self.get_answer_reward(response, answer, task) > 0.0
        if not thinking_part:
            score = 0.5 if student_answer_correct is True else 0.0
            label = "medium" if student_answer_correct is True else "low"
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=score,
                    label=label,
                    thinking_len=0,
                    student_answer_preview=student_answer[:120],
                    student_answer_correct=student_answer_correct,
                    has_answer_flag=has_answer_flag,
                    skipped_no_thinking=True,
                    postprocess_reason="correct_answer_only" if student_answer_correct is True else "",
                )
            self._thinking_score_cache[sample_idx] = score
            return None
        if not self._use_teacher_for_idx(sample_idx) or "chart" not in task:
            score = self._local_fallback.get_thinking_reward_prompt(response, question, answer, hint, task)
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=float(score or 0.0),
                    label="local",
                    thinking_len=len(thinking_part),
                    student_answer_preview=student_answer[:120],
                    student_answer_correct=student_answer_correct,
                    has_answer_flag=has_answer_flag,
                    local_fallback=True,
                )
            self._thinking_score_cache[sample_idx] = float(score or 0.0)
            return None
        return _CheckerJob(
            sample_idx=sample_idx,
            response=response,
            question=question,
            answer=answer,
            hint=hint,
            thinking_part=thinking_part,
            student_answer=student_answer,
            has_answer_flag=has_answer_flag,
            student_answer_correct=student_answer_correct,
        )

    def batch_score_thinking(
        self,
        jobs: list[_CheckerJob],
        task: str,
    ) -> dict[int, float]:
        if not jobs or "chart" not in task:
            return {}
        scores: dict[int, float] = {}
        requests: list[TeacherGenerateRequest] = []
        job_meta: list[_CheckerJob] = []
        ic_texts: list[str] = []
        for job in jobs:
            sample = self._batch_samples[job.sample_idx] if job.sample_idx < len(self._batch_samples) else {}
            image = self._batch_images[job.sample_idx] if job.sample_idx < len(self._batch_images) else None
            q_wo = self._batch_questions[job.sample_idx] if job.sample_idx < len(self._batch_questions) else job.question
            ic_text = ""
            aux_evidence = ""
            aux_used = False
            if self._uses_legacy_ic_prompt():
                ic_text, _ = extract_visual_facts_teacher(
                    teacher_model=self._teacher_model,
                    processor=self._processor,
                    sample=sample,
                    question=q_wo,
                    image=image,
                    ic_source=self._ic_source,
                    max_new_tokens=self._max_ic_tokens,
                    cache=self._ic_cache,
                    recorder=self._recorder,
                    sample_idx=job.sample_idx,
                )
                eval_prompt = prompt_thinking_reward % (ic_text, q_wo, job.answer, job.thinking_part)
            else:
                if not _has_image(image):
                    if self._recorder is not None:
                        self._recorder.record_checker(
                            sample_idx=job.sample_idx,
                            score=0.0,
                            label="image_missing",
                            thinking_len=len(job.thinking_part),
                            has_answer_flag=job.has_answer_flag,
                            thinking_preview=job.thinking_part[:400],
                            image_missing=True,
                            fallback_reason="checker_image_missing",
                        )
                    scores[job.sample_idx] = 0.0
                    continue
                if self._uses_aux_evidence():
                    aux_evidence, _ = ic_text_from_offline_sample(sample)
                    aux_used = bool(aux_evidence)
                eval_prompt = build_image_primary_checker_prompt(
                    question=q_wo,
                    answer=job.answer,
                    reasoning=job.thinking_part,
                    student_answer=job.student_answer,
                    aux_evidence=aux_evidence,
                    aux_mode=self._aux_evidence_mode,
                    has_answer_flag=job.has_answer_flag,
                )
            requests.append(
                TeacherGenerateRequest(
                    prompt_text=eval_prompt,
                    images=[image] if image is not None else [],
                    max_new_tokens=self._max_score_tokens,
                    repetition_penalty=1.05,
                )
            )
            job_meta.append(job)
            ic_texts.append(ic_text)
            job.aux_used = aux_used  # type: ignore[attr-defined]
            self._teacher_budget_used += 1

        if not requests:
            self._thinking_score_cache.update(scores)
            return scores

        raw_outputs, _ = teacher_generate_batched_chunks(
            self._teacher_model,
            self._processor,
            requests,
            chunk_size=self._teacher_batch_size,
            recorder=self._recorder,
            timing_kind="checker",
        )

        for job, raw_out, ic_text in zip(job_meta, raw_outputs, ic_texts):
            try:
                score, label = _score_from_label(raw_out)
                parse_failure = label == "unknown"
                score, label, postprocess_reason = _postprocess_checker_label(
                    score=score,
                    label=label,
                    reasoning=job.thinking_part,
                    has_answer_flag=job.has_answer_flag,
                    student_answer_correct=job.student_answer_correct,
                )
                template_extract_attempted = False
                if score >= 1.0:
                    template_extract_attempted = True
                    image = self._batch_images[job.sample_idx] if job.sample_idx < len(self._batch_images) else None
                    tpl_prompt = prompt_template % job.thinking_part
                    ext_template, _ = teacher_generate_one(
                        self._teacher_model,
                        self._processor,
                        tpl_prompt,
                        [image] if image is not None else [],
                        max_new_tokens=self._max_template_tokens,
                        do_sample=False,
                        recorder=self._recorder,
                        timing_kind="checker",
                    )
                    if is_valid_reasoning_template(ext_template):
                        written, cmp_label = self.template_pool.maybe_update(
                            ext_template,
                            lambda cur, new: self._compare_templates(cur, new, _chart_system_prompt()),
                        )
                        if self._recorder is not None:
                            self._recorder.record_pool(
                                msg="template_candidate",
                                sample_idx=job.sample_idx,
                                template_preview=ext_template[:400],
                                compare_result=cmp_label,
                                written=written,
                                pool_path=self.template_pool.template_path,
                            )
                    elif self._recorder is not None:
                        self._recorder.record_pool(
                            msg="invalid_template_candidate",
                            sample_idx=job.sample_idx,
                            template_preview=(ext_template or "")[:400],
                            compare_result="invalid",
                            written=False,
                            pool_path=self.template_pool.template_path,
                        )
                if self._recorder is not None:
                    self._recorder.record_checker(
                        sample_idx=job.sample_idx,
                        score=score,
                        label=label,
                        thinking_len=len(job.thinking_part),
                        student_answer_preview=job.student_answer[:120],
                        student_answer_correct=job.student_answer_correct,
                        has_answer_flag=job.has_answer_flag,
                        thinking_preview=job.thinking_part[:400],
                        ic_chars=len(ic_text),
                        grounding=self._grounding,
                        aux_evidence_used=bool(getattr(job, "aux_used", False)),
                        parse_failure=parse_failure,
                        raw_teacher_output=raw_out,
                        postprocess_reason=postprocess_reason,
                        template_extract_attempted=template_extract_attempted,
                    )
                scores[job.sample_idx] = score
            except Exception as exc:
                score = self._local_fallback.get_thinking_reward_prompt(
                    job.response, job.question, job.answer, job.hint, task
                )
                if self._recorder is not None:
                    self._recorder.record_checker(
                        sample_idx=job.sample_idx,
                        score=float(score or 0.0),
                        label="local",
                        thinking_len=len(job.thinking_part),
                        local_fallback=True,
                        error=str(exc)[:120],
                    )
                scores[job.sample_idx] = float(score or 0.0)
        self._thinking_score_cache.update(scores)
        return scores

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        sample_idx = getattr(self, "_current_sample_idx", 0)
        if sample_idx in self._thinking_score_cache:
            return self._thinking_score_cache[sample_idx]
        job = self._collect_checker_job(sample_idx, response, question, answer, hint, task)
        if job is None:
            return self._thinking_score_cache.get(sample_idx, 0.0)
        scores = self.batch_score_thinking([job], task)
        return scores.get(sample_idx, 0.0)

    def prepare_thinking_jobs(
        self,
        responses: list[str],
        prompts: list[str],
        answers: list[str],
        hints: list[str],
        task: str,
    ) -> list[_CheckerJob]:
        jobs: list[_CheckerJob] = []
        for i, (resp, prompt, ans, hint) in enumerate(zip(responses, prompts, answers, hints)):
            self._current_sample_idx = i
            job = self._collect_checker_job(i, resp, prompt, ans, hint, task)
            if job is not None:
                jobs.append(job)
        return jobs

    def _compare_templates(self, current: str, new: str, system_prompt: str) -> bool:
        prompt = _comparison_prompt(current, new)
        try:
            out, _ = teacher_generate_one(
                self._teacher_model,
                self._processor,
                prompt,
                [],
                max_new_tokens=30,
                do_sample=False,
                recorder=self._recorder,
                timing_kind="checker",
            )
            return out.strip().upper() == "YES"
        except Exception:
            return False

    def record_route_binding(self, **fields: Any) -> None:
        if self._recorder is not None:
            self._recorder.record_route(**fields)
