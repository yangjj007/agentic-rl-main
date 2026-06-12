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
ALERT_OPSD_LEAKAGE_PATTERN = "OPSD_LEAKAGE_PATTERN"
ALERT_OPSD_ON_CORRECT = "OPSD_ON_CORRECT"


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
        self._eos_history: deque[float] = deque(maxlen=5)
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
                "entropy_first": logits_stats.get("entropy_first"),
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
            "routing/sft_replaced_ratio": "sft_replaced_ratio",
            "routing/opsd_skipped_degenerate": "opsd_skipped_degenerate",
            "routing/opsd_skipped_leakage": "opsd_skipped_leakage",
            "routing/opsd_on_correct_rate": "opsd_on_correct_rate",
            "routing/grpo_on_correct_rate": "grpo_on_correct_rate",
            "routing/opd_teacher_call_rate": "opd_teacher_call_rate",
            "teacher/privileged_suffix_has_gold_rate": "privileged_suffix_has_gold_rate",
            "teacher/visual_fact_empty_rate": "visual_fact_empty_rate",
            "teacher/suffix_len_mean": "teacher_suffix_len_mean",
            "signal/grpo_zero_loss_rate": "grpo_zero_loss_rate",
            "signal/advantage_abs_mean": "advantages_abs_mean",
            "logits/p_greedy_first": "p_greedy_first",
            "logits/p_eos_first": "p_eos_first",
            "health/alert_count": "alert_count",
        }
        for metric_key, field_key in mapping.items():
            val = snapshot.get(field_key)
            if val is not None:
                metrics[metric_key] = _safe_float(val)
        return metrics
