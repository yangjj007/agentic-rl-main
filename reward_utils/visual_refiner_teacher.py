"""7B teacher Visual Refiner for SFT ground-truth hints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from data_utils.chart.prompts import prompt_refine
from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.refiner import ContextRefinerLocal
from reward_utils.template_pool import TemplatePool
from reward_utils.teacher_generate import (
    TeacherGenerateRequest,
    teacher_generate_batched_chunks,
    teacher_generate_one,
)
from reward_utils.visual_batch_ops import prefetch_ic_unique
from reward_utils.visual_ic import extract_visual_facts_teacher


def _chart_system_prompt() -> str:
    return (
        "You are a seasoned professional in the field of chart analysis, demonstrating "
        "exceptional expertise and insight into complex chart data."
    )


@dataclass
class _RefinerJob:
    sample_idx: int
    question: str
    hint: str
    reference_answer: str


def build_no_gold_refiner_prompt(*, ic_text: str, question: str, template: str) -> str:
    """Ask for an evidence-only hint when the reference answer is unavailable.

    The old implementation only removed the literal ``Answer:`` prefix before
    inserting the answer in ``prompt_refine``.  That is still gold-answer
    exposure.  This prompt deliberately has no answer slot and tells the
    teacher not to emit a final answer; the normal online-SFT formatter appends
    the supervised answer later, outside the refiner.
    """
    return f"""Given an attached chart image and the extracted visual evidence below, write an evidence-grounded reasoning hint for the question.

Do not infer, state, or format a final answer. Do not use a final-answer heading.
Use the requested structure and only describe observations that are supported by the image or visual evidence.

<IC>:\n{ic_text}
<Q>:\n{question}
<T>:\n{template}
<Output>:\n"""


def _is_usable_refined_hint(text: str, *, include_gold: bool) -> bool:
    """Reject malformed/no-gold refiner output instead of training on it."""
    lowered = (text or "").lower()
    if not text or "goal:" not in lowered or "observation:" not in lowered:
        return False
    if not include_gold and "answer:" in lowered:
        return False
    return True


class TeacherVisualRefiner(ContextRefinerLocal):
    """Teacher-backed hint refinement; passthrough on failure."""

    requires_sequential = True

    def __init__(
        self,
        RL_CONFIG,
        CLIENT_CONFIG,
        gpu_id=0,
        *,
        visual_config: Optional[dict] = None,
        template_pool: Optional[TemplatePool] = None,
    ):
        super().__init__(RL_CONFIG, CLIENT_CONFIG, gpu_id=gpu_id)
        self.visual_config = visual_config or {}
        refiner_cfg = self.visual_config.get("refiner", {})
        self._enabled = bool(refiner_cfg.get("enabled", True))
        self._ic_source = self.visual_config.get("ic_source", "teacher_image")
        self._max_ic_tokens = int(refiner_cfg.get("max_ic_tokens", 768))
        self._max_refine_tokens = int(refiner_cfg.get("max_refine_tokens", 1000))
        self._skip_cold_start = bool(refiner_cfg.get("skip_cold_start", True))
        self._prefetch_ic = bool(self.visual_config.get("prefetch_ic", True))
        self._teacher_batch_size = int(self.visual_config.get("teacher_batch_size", 4))
        self.template_pool = template_pool or TemplatePool(
            template_path=self.visual_config.get("template_pool", {}).get("path", "best_template.txt"),
            refresh_interval_sec=float(
                self.visual_config.get("template_pool", {}).get("refresh_interval_sec", 60)
            ),
        )
        self._teacher_model = None
        self._processor = None
        self._recorder: Optional[VisualBatchRecorder] = None
        self._ic_cache: dict[tuple[str, str], str] = {}
        self._batch_samples: list[dict] = []
        self._batch_images: list[Any] = []
        self._skip_cold_start_active = False
        self._refine_result_cache: dict[int, str] = {}

    def bind_teacher(self, teacher_model, processor) -> None:
        self._teacher_model = teacher_model
        self._processor = processor

    def begin_generate_batch(
        self,
        *,
        samples: list[dict],
        images: list[Any],
        questions: Optional[list[str]] = None,
        global_step: int,
        output_dir: str,
        recorder: Optional[VisualBatchRecorder] = None,
        ic_cache: Optional[dict] = None,
        skip_cold_start: bool = False,
    ) -> None:
        if recorder is not None:
            self._recorder = recorder
        else:
            log_cfg = self.visual_config.get("logging", {})
            self._recorder = VisualBatchRecorder(
                global_step=global_step,
                output_dir=output_dir,
                log_cfg=log_cfg,
            )
        self._batch_samples = samples
        self._batch_images = images
        self._skip_cold_start_active = skip_cold_start
        self._refine_result_cache = {}
        if ic_cache is not None:
            self._ic_cache = ic_cache
        else:
            self._ic_cache = {}
        batch_questions = questions or [
            s.get("question_wo_prompt", s.get("question", "")) for s in samples
        ]
        if (
            not self._skip_cold_start_active
            and self._prefetch_ic
            and ic_cache is None
        ):
            prefetch_ic_unique(
                teacher_model=self._teacher_model,
                processor=self._processor,
                samples=samples,
                images=images,
                questions=batch_questions,
                ic_source=self._ic_source,
                max_new_tokens=self._max_ic_tokens,
                cache=self._ic_cache,
                recorder=self._recorder,
                teacher_batch_size=self._teacher_batch_size,
            )

    def end_generate_batch(self) -> None:
        self._batch_samples = []
        self._batch_images = []
        self._skip_cold_start_active = False
        self._refine_result_cache = {}

    def record_refiner_dedupe(
        self,
        *,
        sample_idx: int,
        result: str,
        hint: str,
        source_idx: int,
    ) -> None:
        if self._recorder is None:
            return
        in_len = len(hint or "")
        out_len = len(result or "")
        self._recorder.record_refiner(
            sample_idx=sample_idx,
            changed=result.strip() != hint.strip(),
            in_len=in_len,
            out_len=out_len,
            delta=out_len - in_len,
            dedupe_hit=True,
            dedupe_source_idx=source_idx,
            hint_after_preview=result[:400],
            passthrough=result.strip() == hint.strip(),
        )

    def _passthrough_record(self, sample_idx: int, hint: str, reason: str) -> str:
        in_len = len(hint or "")
        if self._recorder is not None:
            self._recorder.record_refiner(
                sample_idx=sample_idx,
                changed=False,
                in_len=in_len,
                out_len=in_len,
                delta=0,
                passthrough=True,
                reason=reason,
            )
        return hint

    def batch_refine_hints(self, jobs: list[_RefinerJob], task: str) -> dict[int, str]:
        results: dict[int, str] = {}
        if not jobs:
            return results

        if self._skip_cold_start_active:
            for job in jobs:
                results[job.sample_idx] = self._passthrough_record(
                    job.sample_idx, job.hint, "skip_cold_start"
                )
            return results

        if not self._enabled or self._teacher_model is None or "chart" not in task:
            for job in jobs:
                results[job.sample_idx] = self._passthrough_record(
                    job.sample_idx, job.hint, "disabled_or_no_teacher"
                )
            return results

        template = self.template_pool.get_template()
        include_gold = self.visual_config.get("refiner", {}).get("include_gold", False)
        requests: list[TeacherGenerateRequest] = []
        job_meta: list[_RefinerJob] = []
        ic_texts: list[str] = []

        for job in jobs:
            if not job.hint:
                results[job.sample_idx] = self._passthrough_record(job.sample_idx, job.hint, "empty_hint")
                continue

            sample = self._batch_samples[job.sample_idx] if job.sample_idx < len(self._batch_samples) else {}
            image = self._batch_images[job.sample_idx] if job.sample_idx < len(self._batch_images) else None
            ic_text, _ = extract_visual_facts_teacher(
                teacher_model=self._teacher_model,
                processor=self._processor,
                sample=sample,
                question=job.question,
                image=image,
                ic_source=self._ic_source,
                max_new_tokens=self._max_ic_tokens,
                cache=self._ic_cache,
                recorder=self._recorder,
                sample_idx=job.sample_idx,
            )
            if include_gold:
                eval_prompt = prompt_refine % (
                    ic_text,
                    job.question,
                    job.reference_answer,
                    template,
                )
            else:
                eval_prompt = build_no_gold_refiner_prompt(
                    ic_text=ic_text,
                    question=job.question,
                    template=template,
                )
            requests.append(
                TeacherGenerateRequest(
                    prompt_text=eval_prompt,
                    images=[image] if image is not None else [],
                    max_new_tokens=self._max_refine_tokens,
                    repetition_penalty=1.05,
                )
            )
            job_meta.append(job)
            ic_texts.append(ic_text)

        if not requests:
            return results

        raw_outputs, _ = teacher_generate_batched_chunks(
            self._teacher_model,
            self._processor,
            requests,
            chunk_size=self._teacher_batch_size,
            recorder=self._recorder,
            timing_kind="refiner",
        )

        for job, raw_out, ic_text in zip(job_meta, raw_outputs, ic_texts):
            in_len = len(job.hint or "")
            try:
                candidate = (raw_out or "").strip()
                valid_output = _is_usable_refined_hint(
                    candidate,
                    include_gold=bool(include_gold),
                )
                refined = candidate if valid_output else job.hint
                out_len = len(refined)
                changed = refined.strip() != job.hint.strip()
                if self._recorder is not None:
                    self._recorder.record_refiner(
                        sample_idx=job.sample_idx,
                        changed=changed,
                        in_len=in_len,
                        out_len=out_len,
                        delta=out_len - in_len,
                        template_source=self.template_pool.template_path,
                        ic_chars=len(ic_text),
                        hint_before_preview=job.hint[:400],
                        hint_after_preview=refined[:400],
                        has_goal="goal:" in refined.lower(),
                        has_observation="observation:" in refined.lower(),
                        has_answer="answer:" in refined.lower(),
                        no_gold_mode=not include_gold,
                        valid_output=valid_output,
                        passthrough=not valid_output,
                        reason="" if valid_output else "invalid_refiner_output",
                        raw_teacher_output=candidate[:400],
                    )
                results[job.sample_idx] = refined
            except Exception as exc:
                results[job.sample_idx] = self._passthrough_record(
                    job.sample_idx, job.hint, str(exc)[:120]
                )

        self._refine_result_cache.update(results)
        return results

    def refine_hint(self, question, hint: str, reference_answer: str, task: str, gpu_id=None):
        sample_idx = getattr(self, "_current_sample_idx", 0)
        if sample_idx in self._refine_result_cache:
            return self._refine_result_cache[sample_idx]

        if not hint:
            return self._passthrough_record(sample_idx, hint, "empty_hint")

        if self._skip_cold_start_active:
            return self._passthrough_record(sample_idx, hint, "skip_cold_start")

        if not self._enabled or self._teacher_model is None or "chart" not in task:
            return self._passthrough_record(sample_idx, hint, "disabled_or_no_teacher")

        results = self.batch_refine_hints(
            [
                _RefinerJob(
                    sample_idx=sample_idx,
                    question=question,
                    hint=hint,
                    reference_answer=reference_answer,
                )
            ],
            task,
        )
        return results.get(sample_idx, hint)
