"""7B teacher Visual Refiner for SFT ground-truth hints."""
from __future__ import annotations

from typing import Any, Optional

from data_utils.chart.prompts import prompt_refine
from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.refiner import ContextRefinerLocal
from reward_utils.template_pool import TemplatePool
from reward_utils.teacher_generate import teacher_generate_one
from reward_utils.visual_ic import extract_visual_facts_teacher


def _chart_system_prompt() -> str:
    return (
        "You are a seasoned professional in the field of chart analysis, demonstrating "
        "exceptional expertise and insight into complex chart data."
    )


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

    def bind_teacher(self, teacher_model, processor) -> None:
        self._teacher_model = teacher_model
        self._processor = processor

    def begin_generate_batch(
        self,
        *,
        samples: list[dict],
        images: list[Any],
        global_step: int,
        output_dir: str,
        recorder: Optional[VisualBatchRecorder] = None,
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
        self._ic_cache = {}

    def end_generate_batch(self) -> None:
        self._batch_samples = []
        self._batch_images = []

    def refine_hint(self, question, hint: str, reference_answer: str, task: str, gpu_id=None):
        sample_idx = getattr(self, "_current_sample_idx", 0)
        in_len = len(hint or "")
        if not hint:
            if self._recorder is not None:
                self._recorder.record_refiner(
                    sample_idx=sample_idx,
                    changed=False,
                    in_len=0,
                    out_len=0,
                    delta=0,
                    reason="empty_hint",
                    passthrough=True,
                )
            return hint

        if not self._enabled or self._teacher_model is None or "chart" not in task:
            if self._recorder is not None:
                self._recorder.record_refiner(
                    sample_idx=sample_idx,
                    changed=False,
                    in_len=in_len,
                    out_len=in_len,
                    delta=0,
                    passthrough=True,
                    reason="disabled_or_no_teacher",
                )
            return hint

        sample = self._batch_samples[sample_idx] if sample_idx < len(self._batch_samples) else {}
        image = self._batch_images[sample_idx] if sample_idx < len(self._batch_images) else None
        template = self.template_pool.get_template()
        ref_answer = reference_answer
        if self.visual_config.get("refiner", {}).get("include_gold") is False:
            ref_answer = reference_answer.lower().replace("answer:", "").strip()

        ic_text, _ = extract_visual_facts_teacher(
            teacher_model=self._teacher_model,
            processor=self._processor,
            sample=sample,
            question=question,
            image=image,
            ic_source=self._ic_source,
            max_new_tokens=self._max_ic_tokens,
            cache=self._ic_cache,
            recorder=self._recorder,
            sample_idx=sample_idx,
        )

        eval_prompt = prompt_refine % (ic_text, question, ref_answer, template)
        try:
            output, _ = teacher_generate_one(
                self._teacher_model,
                self._processor,
                eval_prompt,
                [],
                max_new_tokens=self._max_refine_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
            refined = (output or "").strip() or hint
            out_len = len(refined)
            changed = refined.strip() != hint.strip()
            if self._recorder is not None:
                self._recorder.record_refiner(
                    sample_idx=sample_idx,
                    changed=changed,
                    in_len=in_len,
                    out_len=out_len,
                    delta=out_len - in_len,
                    template_source=self.template_pool.template_path,
                    ic_chars=len(ic_text),
                    hint_before_preview=hint[:400],
                    hint_after_preview=refined[:400],
                    has_goal="goal:" in refined.lower(),
                    has_observation="observation:" in refined.lower(),
                    has_answer=self.visual_config.get("refiner", {}).get("include_gold", False)
                    and "answer" in refined.lower(),
                    passthrough=not changed,
                )
            return refined
        except Exception as exc:
            if self._recorder is not None:
                self._recorder.record_refiner(
                    sample_idx=sample_idx,
                    changed=False,
                    in_len=in_len,
                    out_len=in_len,
                    delta=0,
                    passthrough=True,
                    reason=str(exc)[:120],
                )
            return hint
