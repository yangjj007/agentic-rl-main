"""Training health monitor: degeneration alerts, rolling stats, cross-step correlation."""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

from opsd_utils import debug_log as opsd_debug

ALERT_GEN_CLIP_COLLAPSE = "GEN_CLIP_COLLAPSE"
ALERT_GEN_REPEAT_DEGEN = "GEN_REPEAT_DEGEN"
ALERT_OPT_GRAD_SPIKE = "OPT_GRAD_SPIKE"
ALERT_OPT_NAN_INF = "OPT_NAN_INF"
ALERT_RL_ZERO_SIGNAL = "RL_ZERO_SIGNAL"
ALERT_REWARD_FORMAT_HACK = "REWARD_FORMAT_HACK"
ALERT_DATA_EMPTY_VF = "DATA_EMPTY_VF"
ALERT_LOGIT_MODE_COLLAPSE = "LOGIT_MODE_COLLAPSE"
ALERT_ANSWER_TOKEN_DRIFT = "ANSWER_TOKEN_DRIFT"
ALERT_CLIP_FALSE_HEALTHY = "CLIP_FALSE_HEALTHY"
ALERT_OPSD_LEAKAGE_PATTERN = "OPSD_LEAKAGE_PATTERN"
ALERT_OPSD_ON_CORRECT = "OPSD_ON_CORRECT"
ALERT_VISUAL_IC_FAIL_HIGH = "VISUAL_IC_FAIL_HIGH"
ALERT_VISUAL_CHECKER_ALL_LOW = "VISUAL_CHECKER_ALL_LOW"
ALERT_VISUAL_REFINER_NOOP = "VISUAL_REFINER_NOOP"
ALERT_VISUAL_POOL_STALE = "VISUAL_POOL_STALE"
ALERT_TEMPLATE_SKELETON_COLLAPSE = "TEMPLATE_SKELETON_COLLAPSE"
ALERT_TEMPLATE_ANSWER_COLLAPSE = "TEMPLATE_ANSWER_COLLAPSE"
ALERT_TEMPLATE_PARTIAL_DRIFT = "TEMPLATE_PARTIAL_DRIFT"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _rolling_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


class TrainingHealthMonitor:
    """Collect per-step signals, emit layered [OPSD-HEALTH] logs, expose metrics keys."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.window = max(2, int(cfg.get("window", 20)))
        self.log_on_generate = bool(cfg.get("log_on_generate", True))
        self.log_every_step = bool(cfg.get("log_every_step", True))
        self.log_detail_bundle = bool(cfg.get("log_detail_bundle", True))
        self.log_alerts_immediately = bool(cfg.get("log_alerts_immediately", True))
        self.metrics_every_step = bool(cfg.get("metrics_every_step", True))

        self._history: deque[dict[str, Any]] = deque(maxlen=self.window)
        self._step_fields: dict[str, Any] = {}
        self._step_alerts: list[str] = []
        self._p_greedy_history: deque[float] = deque(maxlen=5)
        self._p_answer_history: deque[float] = deque(maxlen=5)
        self._eos_history: deque[float] = deque(maxlen=5)
        self._checker_all_low_streak: int = 0
        self._last_step: Optional[int] = None

    def reset_step(self, step: int) -> None:
        self._step_fields = {"global_step": step}
        self._step_alerts = []
        self._last_step = step

    def _emit_alert(self, step: int, code: str, **fields: Any) -> None:
        if code not in self._step_alerts:
            self._step_alerts.append(code)
        if self.log_alerts_immediately and opsd_debug.should_log_health_alerts_immediately():
            opsd_debug.log_health("ALERT", code, global_step=step, **fields)

    def _check_generate_alerts(self, step: int, stats: dict[str, Any], logits: dict[str, Any]) -> list[str]:
        clipped = _safe_float(stats.get("clipped_rate"))
        eos_rate = _safe_float(stats.get("eos_terminated_rate"))
        degenerate_rate = _safe_float(stats.get("degenerate_rate"))
        repeat_loop = int(stats.get("repeat_loop_count", 0) or 0)
        p_greedy = _safe_float(logits.get("p_greedy_first"))
        p_eos = _safe_float(logits.get("p_eos_first"))
        p_answer = _safe_float(logits.get("p_answer_first"))
        full_cot_template_rate = _safe_float(stats.get("full_cot_template_rate"))
        partial_cot_template_rate = _safe_float(stats.get("partial_cot_template_rate"))
        goal_without_answer_rate = _safe_float(stats.get("goal_without_answer_rate"))
        empty_cot_skeleton_rate = _safe_float(stats.get("empty_cot_skeleton_rate"))
        malformed_answer_section_rate = _safe_float(
            stats.get("malformed_answer_section_rate")
        )

        if clipped > 0.8 and degenerate_rate < 0.05:
            self._emit_alert(
                step,
                ALERT_CLIP_FALSE_HEALTHY,
                clipped_rate=clipped,
                degenerate_rate=degenerate_rate,
                hint="high clip with low degenerate_rate often masks Answer-only collapse",
            )
        if clipped > 0.7 and eos_rate < 0.3:
            self._emit_alert(
                step,
                ALERT_GEN_CLIP_COLLAPSE,
                clipped_rate=clipped,
                eos_rate=eos_rate,
                hint="raise repetition_penalty, lower temperature, or shorten max_completion_length",
            )
        if degenerate_rate > 0.5 or repeat_loop > 0:
            self._emit_alert(
                step,
                ALERT_GEN_REPEAT_DEGEN,
                degenerate_rate=degenerate_rate,
                repeat_loop_count=repeat_loop,
            )
        if full_cot_template_rate > 0.8 and empty_cot_skeleton_rate > 0.2:
            self._emit_alert(
                step,
                ALERT_TEMPLATE_SKELETON_COLLAPSE,
                full_cot_template_rate=full_cot_template_rate,
                empty_cot_skeleton_rate=empty_cot_skeleton_rate,
                hint="structured reasoning has collapsed into mostly empty fixed headings",
            )
        if full_cot_template_rate > 0.8 and malformed_answer_section_rate > 0.2:
            self._emit_alert(
                step,
                ALERT_TEMPLATE_ANSWER_COLLAPSE,
                full_cot_template_rate=full_cot_template_rate,
                malformed_answer_section_rate=malformed_answer_section_rate,
                hint="fixed reasoning headings co-occur with malformed Answer sections",
            )
        if partial_cot_template_rate > 0.6 and goal_without_answer_rate > 0.6:
            self._emit_alert(
                step,
                ALERT_TEMPLATE_PARTIAL_DRIFT,
                partial_cot_template_rate=partial_cot_template_rate,
                goal_without_answer_rate=goal_without_answer_rate,
                hint="Goal-style headings are spreading before valid Answer sections emerge",
            )

        if p_greedy > 0:
            self._p_greedy_history.append(p_greedy)
            self._eos_history.append(eos_rate)
            if (
                len(self._p_greedy_history) >= 3
                and all(p > 0.99 for p in list(self._p_greedy_history)[-3:])
                and len(self._eos_history) >= 2
                and self._eos_history[-1] < self._eos_history[-2] - 0.1
            ):
                self._emit_alert(
                    step,
                    ALERT_LOGIT_MODE_COLLAPSE,
                    p_greedy_first=p_greedy,
                    p_eos_first=p_eos,
                    eos_rate=eos_rate,
                    hint="first token collapsed to Goal: template; EOS probability near zero",
                )
        if p_answer > 0:
            self._p_answer_history.append(p_answer)
            if len(self._p_answer_history) >= 3 and all(
                p < 0.5 for p in list(self._p_answer_history)[-3:]
            ):
                self._emit_alert(
                    step,
                    ALERT_ANSWER_TOKEN_DRIFT,
                    p_answer_first=p_answer,
                    hint="first-token Answer probability low for 3 consecutive generate batches",
                )
        return list(self._step_alerts)

    def record_generate(
        self,
        step: int,
        stats: dict[str, Any],
        logits_stats: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        if not self.enabled:
            return []
        logits_stats = logits_stats or {}
        self._step_fields.update(
            {
                "degenerate_rate": stats.get("degenerate_rate"),
                "clipped_rate": stats.get("clipped_rate"),
                "eos_terminated_rate": stats.get("eos_terminated_rate"),
                "repeat_loop_count": stats.get("repeat_loop_count"),
                "char_repeat_count": stats.get("char_repeat_count", 0),
                "p_greedy_first": logits_stats.get("p_greedy_first"),
                "p_eos_first": logits_stats.get("p_eos_first"),
                "p_answer_first": logits_stats.get("p_answer_first"),
                "entropy_first": logits_stats.get("entropy_first"),
                "degenerate_rate_format": stats.get("degenerate_rate_format"),
                "degenerate_rate_repeat": stats.get("degenerate_rate_repeat"),
                "format_without_thinking_rate": stats.get("format_without_thinking_rate"),
                "full_cot_template_rate": stats.get("full_cot_template_rate"),
                "partial_cot_template_rate": stats.get("partial_cot_template_rate"),
                "goal_without_answer_rate": stats.get("goal_without_answer_rate"),
                "empty_cot_skeleton_rate": stats.get("empty_cot_skeleton_rate"),
                "malformed_answer_section_rate": stats.get("malformed_answer_section_rate"),
            }
        )
        alerts = self._check_generate_alerts(step, stats, logits_stats)
        alert_str = ",".join(alerts) if alerts else "none"

        if self.log_on_generate and opsd_debug.should_log_health_on_generate():
            opsd_debug.log_health(
                "generate",
                "batch health",
                global_step=step,
                degenerate_rate=stats.get("degenerate_rate"),
                clipped_rate=stats.get("clipped_rate"),
                eos_rate=stats.get("eos_terminated_rate"),
                repeat_loop_count=stats.get("repeat_loop_count"),
                char_repeat_count=stats.get("char_repeat_count", 0),
                p_greedy=logits_stats.get("p_greedy_first"),
                p_eos=logits_stats.get("p_eos_first"),
                alerts=alert_str,
            )
        return alerts

    def record_visual(self, step: int, fields: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._step_fields.update(fields)
        ic_ok_rate = _safe_float(fields.get("visual/ic_ok_rate"), default=1.0)
        ic_calls = _safe_float(fields.get("visual/ic_calls"), default=0.0)
        ic_fail_count = _safe_float(fields.get("visual/ic_fail_count"), default=0.0)
        if (ic_calls > 0.0 or ic_fail_count > 0.0) and ic_ok_rate < 0.8:
            self._emit_alert(
                step,
                ALERT_VISUAL_IC_FAIL_HIGH,
                ic_ok_rate=ic_ok_rate,
            )
        checker_high = int(fields.get("visual/checker_high", 0) or 0)
        checker_medium = int(fields.get("visual/checker_medium", 0) or 0)
        if checker_high + checker_medium == 0:
            self._checker_all_low_streak += 1
        else:
            self._checker_all_low_streak = 0
        if self._checker_all_low_streak >= 3:
            self._emit_alert(
                step,
                ALERT_VISUAL_CHECKER_ALL_LOW,
                streak=self._checker_all_low_streak,
            )
        refiner_changed_rate = _safe_float(fields.get("visual/refiner_changed_rate"))
        sft_ratio = _safe_float(self._step_fields.get("sft_replaced_ratio"))
        if refiner_changed_rate < 0.1 and sft_ratio > 0.5:
            self._emit_alert(
                step,
                ALERT_VISUAL_REFINER_NOOP,
                refiner_changed_rate=refiner_changed_rate,
                sft_replaced_ratio=sft_ratio,
            )
        pool_updates = int(fields.get("visual/pool_updates", 0) or 0)
        if step > 500 and pool_updates == 0:
            self._emit_alert(
                step,
                ALERT_VISUAL_POOL_STALE,
                pool_updates=pool_updates,
            )

    def record_data(self, step: int, fields: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._step_fields.update(fields)
        vf_empty = _safe_float(fields.get("visual_fact_empty_rate"))
        if vf_empty > 0.5:
            self._emit_alert(
                step,
                ALERT_DATA_EMPTY_VF,
                visual_fact_empty_rate=vf_empty,
                hint="rebuild train_medium_vf_full.json with visual_fact hints",
            )

    def record_routing(self, step: int, fields: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._step_fields.update(fields)
        format_mean = _safe_float(fields.get("format_mean"))
        acc_mean = _safe_float(fields.get("accuracy_mean"))
        degenerate_rate = _safe_float(self._step_fields.get("degenerate_rate"))
        if format_mean > 0.7 and acc_mean < 0.05 and degenerate_rate > 0.4:
            self._emit_alert(
                step,
                ALERT_REWARD_FORMAT_HACK,
                format_mean=format_mean,
                accuracy_mean=acc_mean,
                degenerate_rate=degenerate_rate,
            )
        opsd_on_correct = _safe_float(fields.get("opsd_on_correct_rate"))
        if opsd_on_correct > 0.01:
            self._emit_alert(
                step,
                ALERT_OPSD_ON_CORRECT,
                opsd_on_correct_rate=opsd_on_correct,
            )
        leakage_skip = int(fields.get("opsd_skipped_leakage", 0) or 0)
        if leakage_skip > 0:
            self._emit_alert(
                step,
                ALERT_OPSD_LEAKAGE_PATTERN,
                opsd_skipped_leakage=leakage_skip,
            )

    def record_loss(self, step: int, fields: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._step_fields.update(fields)
        loss_val = fields.get("combined_loss_scalar", fields.get("grpo_loss_scalar"))
        if loss_val is not None and not math.isfinite(_safe_float(loss_val, default=float("nan"))):
            self._emit_alert(step, ALERT_OPT_NAN_INF, loss=loss_val)

        adv_abs = _safe_float(fields.get("advantages_abs_mean"))
        zero_grpo = _safe_float(fields.get("grpo_zero_loss_rate"))
        if adv_abs < 1e-6 and zero_grpo > 0.8:
            self._emit_alert(
                step,
                ALERT_RL_ZERO_SIGNAL,
                advantages_abs_mean=adv_abs,
                grpo_zero_loss_rate=zero_grpo,
            )

    def record_optimizer(self, step: int, grad_norm: Optional[float], lr: Optional[float]) -> None:
        if not self.enabled:
            return
        gn = _safe_float(grad_norm) if grad_norm is not None else None
        if gn is not None:
            self._step_fields["grad_norm"] = gn
            hist = [h.get("grad_norm") for h in self._history if h.get("grad_norm") is not None]
            if len(hist) >= 3:
                mean, std = _rolling_mean_std([float(x) for x in hist])
                if std > 1e-8 and gn > mean + 3 * std:
                    self._emit_alert(
                        step,
                        ALERT_OPT_GRAD_SPIKE,
                        grad_norm=gn,
                        rolling_mean=mean,
                        rolling_std=std,
                    )
        if lr is not None:
            self._step_fields["learning_rate"] = lr

    def correlate(self) -> dict[str, Any]:
        """Cross-step deltas and root-cause hints from rolling history."""
        hints: list[str] = []
        out: dict[str, Any] = {"root_cause_hints": hints}

        if len(self._history) < 2:
            out["root_cause_hints"] = ["insufficient history for correlation"]
            return out

        prev = self._history[-1]
        prev2 = self._history[-2] if len(self._history) >= 2 else prev

        for key in ("grad_norm", "clipped_rate", "eos_terminated_rate", "p_greedy_first", "degenerate_rate"):
            cur_v = self._step_fields.get(key)
            old_v = prev.get(key)
            if cur_v is not None and old_v is not None:
                out[f"delta_{key}"] = _safe_float(cur_v) - _safe_float(old_v)

        gn_prev = prev.get("grad_norm")
        clip_cur = self._step_fields.get("clipped_rate")
        if gn_prev is not None and clip_cur is not None and _safe_float(clip_cur) > 0.7:
            hints.append("high clip rate may follow recent gradient update (check delta_grad_norm)")

        p_prev = prev2.get("p_greedy_first")
        p_cur = self._step_fields.get("p_greedy_first")
        eos_prev = prev2.get("eos_terminated_rate")
        eos_cur = self._step_fields.get("eos_terminated_rate")
        if (
            p_prev is not None
            and p_cur is not None
            and _safe_float(p_cur) > 0.99
            and eos_prev is not None
            and eos_cur is not None
            and _safe_float(eos_prev) > 0.5
            and _safe_float(eos_cur) < 0.2
        ):
            hints.append("after gradient step: p_greedy rose to ~1.0 and eos_rate collapsed")

        if ALERT_RL_ZERO_SIGNAL in self._step_alerts and ALERT_GEN_REPEAT_DEGEN in self._step_alerts:
            hints.append("RL zero signal co-occurs with repetition degeneration")

        if not hints:
            hints.append("none")
        out["root_cause_hints"] = hints
        return out

    def maybe_log_detail_bundle(self, step: int) -> None:
        if not self.enabled or not self.log_detail_bundle:
            return
        if not opsd_debug.should_log_detail(step):
            return
        opsd_debug.log_health_detail_banner(step, "TRAINING HEALTH BUNDLE")
        corr = self.correlate()
        hist_keys = (
            "degenerate_rate",
            "clipped_rate",
            "eos_terminated_rate",
            "grad_norm",
            "p_greedy_first",
            "grpo_zero_loss_rate",
            "sft_replaced_ratio",
        )
        rolling: dict[str, Any] = {}
        for key in hist_keys:
            vals = [_safe_float(h[key]) for h in self._history if h.get(key) is not None]
            if vals:
                mean, std = _rolling_mean_std(vals)
                rolling[f"{key}_mean"] = mean
                rolling[f"{key}_std"] = std

        snapshot_fields = {k: v for k, v in self._step_fields.items() if k != "global_step"}
        opsd_debug.log_health_detail(
            "health",
            "step snapshot",
            global_step=step,
            alerts=self._step_alerts or ["none"],
            **snapshot_fields,
            **rolling,
        )
        opsd_debug.log_health_detail(
            "health",
            "cross-step correlation",
            global_step=step,
            **corr,
        )

    def finish_step(self, step: int) -> dict[str, float]:
        """L2 step summary log + metrics keys for Trainer.log()."""
        snapshot = dict(self._step_fields)
        snapshot["alert_count"] = len(self._step_alerts)
        snapshot["alerts"] = list(self._step_alerts)
        self._history.append(snapshot)

        if self.log_every_step and opsd_debug.should_log_health_every_step():
            corr = self.correlate()
            opsd_debug.log_health(
                "step",
                "step summary",
                global_step=step,
                grad_norm=snapshot.get("grad_norm"),
                lr=snapshot.get("learning_rate"),
                sft_replaced_ratio=snapshot.get("sft_replaced_ratio"),
                grpo_zero_loss_rate=snapshot.get("grpo_zero_loss_rate"),
                degenerate_rate=snapshot.get("degenerate_rate"),
                clipped_rate=snapshot.get("clipped_rate"),
                eos_rate=snapshot.get("eos_terminated_rate"),
                alert_count=len(self._step_alerts),
                hints=corr.get("root_cause_hints"),
            )

        self.maybe_log_detail_bundle(step)

        if not self.metrics_every_step:
            return {}

        metrics: dict[str, float] = {}
        mapping = {
            "completions/degenerate_rate": "degenerate_rate",
            "completions/eos_rate": "eos_terminated_rate",
            "completions/repeat_loop_count": "repeat_loop_count",
            "completions/full_cot_template_rate": "full_cot_template_rate",
            "completions/partial_cot_template_rate": "partial_cot_template_rate",
            "completions/goal_without_answer_rate": "goal_without_answer_rate",
            "completions/empty_cot_skeleton_rate": "empty_cot_skeleton_rate",
            "completions/malformed_answer_section_rate": "malformed_answer_section_rate",
            "routing/sft_replaced_ratio": "sft_replaced_ratio",
            "routing/opsd_skipped_degenerate": "opsd_skipped_degenerate",
            "routing/opsd_skipped_leakage": "opsd_skipped_leakage",
            "routing/opsd_on_correct_rate": "opsd_on_correct_rate",
            "routing/grpo_on_correct_rate": "grpo_on_correct_rate",
            "routing/opd_teacher_call_rate": "opd_teacher_call_rate",
            "routing/grpo_route_rate": "grpo_route_rate",
            "routing/opd_route_rate": "opd_route_rate",
            "routing/sft_route_rate": "sft_route_rate",
            "routing/total_completion_count": "total_completion_count",
            "routing/wrong_completion_count": "wrong_completion_count",
            "routing/probe_candidate_count": "probe_candidate_count",
            "routing/teacher_correct_count": "teacher_correct_count",
            "routing/opd_route_count": "opd_route_count",
            "routing/sft_route_count": "sft_route_count",
            "routing/grpo_route_count": "grpo_route_count",
            "routing/teacher_probe_candidate_rate": "teacher_probe_candidate_rate",
            "routing/teacher_probe_correct_rate": "teacher_probe_correct_rate",
            "routing/teacher_probe_wrong_rate": "teacher_probe_wrong_rate",
            "routing/teacher_probe_skipped_no_evidence_rate": "teacher_probe_skipped_no_evidence_rate",
            "routing/teacher_probe_skipped_budget_rate": "teacher_probe_skipped_budget_rate",
            "routing/teacher_probe_candidate_accuracy": "teacher_probe_candidate_accuracy",
            "routing/teacher_probe_probed_accuracy": "teacher_probe_probed_accuracy",
            "routing/teacher_probe_evidence_present_rate": "teacher_probe_evidence_present_rate",
            "routing/teacher_probe_deplot_placeholder_rate": "teacher_probe_deplot_placeholder_rate",
            "routing/teacher_probe_deplot_real_rate": "teacher_probe_deplot_real_rate",
            "routing/teacher_probe_visual_fact_used_rate": "teacher_probe_visual_fact_used_rate",
            "routing/teacher_probe_answer_flag_rate": "teacher_probe_answer_flag_rate",
            "routing/teacher_probe_parse_fail_rate": "teacher_probe_parse_fail_rate",
            "routing/teacher_probe_gold_suffix_rate": "teacher_probe_gold_suffix_rate",
            "routing/degenerate_hard_override_rate": "degenerate_hard_override_rate",
            "routing/clipped_hard_override_rate": "clipped_hard_override_rate",
            "routing/teacher_correct_overridden_rate": "teacher_correct_overridden_rate",
            "routing/signal_aware_sft_rate": "signal_aware_sft_rate",
            "routing/utility_candidate_rate": "utility_candidate_rate",
            "routing/utility_grpo_mean": "utility_grpo_mean",
            "routing/utility_opd_mean": "utility_opd_mean",
            "routing/utility_sft_mean": "utility_sft_mean",
            "routing/utility_margin_mean": "utility_margin_mean",
            "routing/utility_reroute_grpo_rate": "utility_reroute_grpo_rate",
            "routing/utility_reroute_opd_rate": "utility_reroute_opd_rate",
            "routing/utility_reroute_sft_rate": "utility_reroute_sft_rate",
            "routing/utility_teacher_traj_removed_rate": "utility_teacher_traj_removed_rate",
            "routing/utility_switch_rate": "utility_switch_rate",
            "routing/utility_blocked_switch_rate": "utility_blocked_switch_rate",
            "routing/utility_stable_hold_rate": "utility_stable_hold_rate",
            "routing/utility_invalid_current_switch_rate": "utility_invalid_current_switch_rate",
            "routing/utility_switch_gain_mean": "utility_switch_gain_mean",
            "routing/utility_ema_grpo_mean": "utility_ema_grpo_mean",
            "routing/utility_ema_opd_mean": "utility_ema_opd_mean",
            "routing/utility_ema_sft_mean": "utility_ema_sft_mean",
            "routing/utility_mode_stable_state_count": "utility_mode_stable_state_count",
            "routing/opd_route_cap_rate": "opd_route_cap_rate",
            "routing/opd_route_cap_prompt_rate": "opd_route_cap_prompt_rate",
            "routing/opd_route_cap_kept_rate": "opd_route_cap_kept_rate",
            "routing/opd_route_cap_teacher_traj_removed_rate": "opd_route_cap_teacher_traj_removed_rate",
            "routing/opd_route_cap_grpo_rate": "opd_route_cap_grpo_rate",
            "routing/opd_route_cap_skip_rate": "opd_route_cap_skip_rate",
            "routing/skip_route_rate": "skip_route_rate",
            "sampling/effective_enabled": "effective_sampling_enabled",
            "sampling/effective_mixed_update_rate": "effective_sampling_mixed_update_rate",
            "sampling/effective_all_wrong_update_rate": "effective_sampling_all_wrong_update_rate",
            "sampling/effective_all_correct_update_rate": "effective_sampling_all_correct_update_rate",
            "sampling/effective_missing_index_rate": "effective_sampling_missing_index_rate",
            "filter/effective_group_enabled": "effective_group_filter_enabled",
            "filter/effective_group_filtered_rate": "effective_group_filtered_rate",
            "filter/effective_group_all_wrong_filtered_rate": "effective_group_all_wrong_filtered_rate",
            "filter/effective_group_all_correct_filtered_rate": "effective_group_all_correct_filtered_rate",
            "filter/effective_group_kept_all_wrong_rate": "effective_group_kept_all_wrong_rate",
            "filter/effective_group_teacher_traj_removed_rate": "effective_group_teacher_traj_removed_rate",
            "loss/positive_replay": "positive_replay_loss",
            "loss/positive_replay_weight": "positive_replay_weight",
            "replay/positive_available": "positive_replay_available",
            "replay/positive_skipped_rate": "positive_replay_skipped_rate",
            "replay/positive_batch_size": "positive_replay_batch_size",
            "replay/positive_tokens": "positive_replay_tokens",
            "loss/rollout_replay": "rollout_replay_loss",
            "loss/rollout_replay_weight": "rollout_replay_weight",
            "replay/rollout_available": "rollout_replay_available",
            "replay/rollout_skipped_rate": "rollout_replay_skipped_rate",
            "replay/rollout_batch_size": "rollout_replay_batch_size",
            "replay/rollout_tokens": "rollout_replay_tokens",
            "replay/rollout_advantage_mean": "rollout_replay_advantage_mean",
            "replay/rollout_buffer_size": "rollout_replay_buffer_size",
            "replay/rollout_added": "rollout_replay_added",
            "replay/rollout_skipped_not_positive": "rollout_replay_skipped_not_positive",
            "replay/rollout_skipped_low_advantage": "rollout_replay_skipped_low_advantage",
            "routing/teacher_sft_repair_rate": "teacher_sft_repair_rate",
            "routing/teacher_sft_repair_all_wrong_rate": "teacher_sft_repair_all_wrong_rate",
            "routing/teacher_sft_repair_slot_utilization": "teacher_sft_repair_slot_utilization",
            "routing/teacher_correct_to_opd_rate": "teacher_correct_to_opd_rate",
            "routing/teacher_correct_to_sft_repair_rate": "teacher_correct_to_sft_repair_rate",
            "repair/repaired_prompt_to_mixed_rate": "repaired_prompt_to_mixed_rate",
            "repair/repaired_prompt_still_all_wrong_rate": "repaired_prompt_still_all_wrong_rate",
            "repair/teacher_sft_target_student_short_rate": "teacher_sft_target_student_short_rate",
            "repair/teacher_sft_target_answer_only_rate": "teacher_sft_target_answer_only_rate",
            "repair/teacher_sft_target_full_hint_format_rate": "teacher_sft_target_full_hint_format_rate",
            "repair/teacher_sft_target_exact_answer_line_rate": "teacher_sft_target_exact_answer_line_rate",
            "leakage/teacher_sft_privileged_tag_rate": "teacher_sft_privileged_tag_rate",
            "teacher_probe/generated_tokens_mean": "teacher_probe_generated_tokens_mean",
            "teacher_probe/generated_tokens_p95": "teacher_probe_generated_tokens_p95",
            "teacher_probe/clipped_rate": "teacher_probe_clipped_rate",
            "reward/perception_mean": "perception_reward_mean",
            "reward/perception_skipped_rate": "perception_reward_skipped_rate",
            "reward/perception_judge_parse_fail_rate": "perception_judge_parse_fail_rate",
            "reward/diagnostic_deplot_overlap_mean": "diagnostic_deplot_overlap_mean",
            "teacher/privileged_suffix_has_gold_rate": "privileged_suffix_has_gold_rate",
            "teacher/visual_fact_empty_rate": "visual_fact_empty_rate",
            "teacher/suffix_len_mean": "teacher_suffix_len_mean",
            "signal/grpo_zero_loss_rate": "grpo_zero_loss_rate",
            "signal/advantage_abs_mean": "advantages_abs_mean",
            "signal/reward_std_mean": "reward_std_mean",
            "signal/group_all_wrong_rate": "group_all_wrong_rate",
            "signal/group_mixed_rate": "group_mixed_rate",
            "signal/reward_std_lt_0_01_rate": "reward_std_lt_0_01_rate",
            "signal/reward_std_lt_0_05_rate": "reward_std_lt_0_05_rate",
            "signal/reward_std_lt_0_10_rate": "reward_std_lt_0_10_rate",
            "loss/opsd_scheduled_base_weight": "opsd_scheduled_base_weight",
            "loss/opsd_effective_weight": "opsd_effective_weight",
            "loss/opsd_adaptive_multiplier": "opsd_adaptive_multiplier",
            "logits/p_greedy_first": "p_greedy_first",
            "logits/p_eos_first": "p_eos_first",
            "logits/p_answer_first": "p_answer_first",
            "completions/degenerate_rate_format": "degenerate_rate_format",
            "completions/degenerate_rate_repeat": "degenerate_rate_repeat",
            "metrics/format_without_thinking_rate": "format_without_thinking_rate",
            "phase/sft_cold_start": "phase_sft_cold_start",
            "phase/training_progress": "training_progress",
            "phase/max_training_steps": "max_training_steps",
            "phase/teacher_traj_decay_active": "teacher_traj_decay_active",
            "phase/effective_sampling_active": "effective_sampling_active",
            "phase/opd_decay_active": "opd_decay_active",
            "phase/opd_route_cap_active": "opd_route_cap_active",
            "phase/dynamic_mixed_rate_ema": "dynamic_mixed_rate_ema",
            "phase/dynamic_zero_loss_rate_ema": "dynamic_zero_loss_rate_ema",
            "phase/dynamic_mixed_ready": "dynamic_mixed_ready",
            "phase/dynamic_zero_loss_ready": "dynamic_zero_loss_ready",
            "phase/dynamic_joint_ready": "dynamic_joint_ready",
            "phase/dynamic_ready_streak": "dynamic_ready_streak",
            "phase/dynamic_would_trigger": "dynamic_would_trigger",
            "health/alert_count": "alert_count",
            "visual/ic_ok_rate": "visual/ic_ok_rate",
            "visual/checker_mean": "visual/checker_mean",
            "visual/refiner_changed_rate": "visual/refiner_changed_rate",
            "visual/pool_updates": "visual/pool_updates",
            "visual/ic_latency_ms": "visual/ic_latency_ms",
            "visual/checker_latency_ms": "visual/checker_latency_ms",
            "visual/refiner_latency_ms": "visual/refiner_latency_ms",
            "visual/ic_calls": "visual/ic_calls",
            "visual/checker_calls": "visual/checker_calls",
            "visual/refiner_calls": "visual/refiner_calls",
            "visual/teacher_batch_calls": "visual/teacher_batch_calls",
        }
        for metric_key, field_key in mapping.items():
            val = snapshot.get(field_key)
            if val is not None:
                metrics[metric_key] = _safe_float(val)
        return metrics
