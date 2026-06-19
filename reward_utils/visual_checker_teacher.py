"""7B teacher Visual Checker for thinking reward."""
from __future__ import annotations

import re
from typing import Any, Optional

from data_utils.chart.prompts import prompt_template, prompt_thinking_reward
from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.checker import RewardCalculatorLocal
from reward_utils.template_pool import TemplatePool, _comparison_prompt
from reward_utils.teacher_generate import teacher_generate_one
from reward_utils.visual_ic import extract_visual_facts_teacher


def _score_from_label(text: str) -> tuple[float, str]:
    low = (text or "").strip().lower()
    if "high" in low:
        return 1.0, "high"
    if "medium" in low:
        return 0.5, "medium"
    if "low" in low:
        return 0.0, "low"
    return 0.0, "low"


def _chart_system_prompt() -> str:
    return (
        "You are a seasoned professional in the field of chart analysis, demonstrating "
        "exceptional expertise and insight into complex chart data. Your output should be "
        "only judgement, without any additional text or explanation."
    )


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
        self._ic_source = self.visual_config.get("ic_source", "teacher_image")
        self._max_ic_tokens = int(checker_cfg.get("max_ic_tokens", 768))
        self._max_score_tokens = int(checker_cfg.get("max_score_tokens", 16))
        self._max_template_tokens = int(checker_cfg.get("max_template_tokens", 512))
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

    def end_generate_batch(self) -> dict[str, Any]:
        if self._recorder is None:
            return {}
        stats = self._recorder.finish()
        self._recorder = None
        self._batch_samples = []
        return stats

    def _use_teacher_for_idx(self, idx: int) -> bool:
        if not self._enabled or self._teacher_model is None:
            return False
        if self._max_per_batch > 0 and self._teacher_budget_used >= self._max_per_batch:
            return False
        return True

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        sample_idx = getattr(self, "_current_sample_idx", 0)
        thinking_part = response.lower().split(self.answer_flag)[0].strip()
        has_answer_flag = self.answer_flag in response.lower()

        if not thinking_part:
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=0.0,
                    label="low",
                    thinking_len=0,
                    has_answer_flag=has_answer_flag,
                    skipped_no_thinking=True,
                )
            return 0.0

        if not self._use_teacher_for_idx(sample_idx) or "chart" not in task:
            score = self._local_fallback.get_thinking_reward_prompt(response, question, answer, hint, task)
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=float(score or 0.0),
                    label="local",
                    thinking_len=len(thinking_part),
                    has_answer_flag=has_answer_flag,
                    local_fallback=True,
                )
            return score

        sample = self._batch_samples[sample_idx] if sample_idx < len(self._batch_samples) else {}
        image = self._batch_images[sample_idx] if sample_idx < len(self._batch_images) else None
        q_wo = self._batch_questions[sample_idx] if sample_idx < len(self._batch_questions) else question

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
            sample_idx=sample_idx,
        )
        self._teacher_budget_used += 1

        system_prompt = _chart_system_prompt()
        eval_prompt = prompt_thinking_reward % (ic_text, q_wo, answer, thinking_part)
        try:
            raw_out, _ = teacher_generate_one(
                self._teacher_model,
                self._processor,
                eval_prompt,
                [],
                max_new_tokens=self._max_score_tokens,
                do_sample=False,
            )
            score, label = _score_from_label(raw_out)
            template_extract_attempted = False
            if score >= 1.0:
                template_extract_attempted = True
                tpl_prompt = prompt_template % thinking_part
                ext_template, _ = teacher_generate_one(
                    self._teacher_model,
                    self._processor,
                    tpl_prompt,
                    [],
                    max_new_tokens=self._max_template_tokens,
                    do_sample=False,
                )
                if ext_template and "none" not in ext_template.strip().lower():
                    written, cmp_label = self.template_pool.maybe_update(
                        ext_template,
                        lambda cur, new: self._compare_templates(cur, new, system_prompt),
                    )
                    if self._recorder is not None:
                        self._recorder.record_pool(
                            msg="template_candidate",
                            sample_idx=sample_idx,
                            template_preview=ext_template[:400],
                            compare_result=cmp_label,
                            written=written,
                            pool_path=self.template_pool.template_path,
                        )
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=score,
                    label=label,
                    thinking_len=len(thinking_part),
                    has_answer_flag=has_answer_flag,
                    thinking_preview=thinking_part[:400],
                    ic_chars=len(ic_text),
                    raw_teacher_output=raw_out,
                    template_extract_attempted=template_extract_attempted,
                )
            return score
        except Exception as exc:
            score = self._local_fallback.get_thinking_reward_prompt(response, question, answer, hint, task)
            if self._recorder is not None:
                self._recorder.record_checker(
                    sample_idx=sample_idx,
                    score=float(score or 0.0),
                    label="local",
                    thinking_len=len(thinking_part),
                    local_fallback=True,
                    error=str(exc)[:120],
                )
            return score

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
            )
            return out.strip().upper() == "YES"
        except Exception:
            return False

    def record_route_binding(self, **fields: Any) -> None:
        if self._recorder is not None:
            self._recorder.record_route(**fields)
