#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pcd_artifact_core import (
    COUNT_ALIASES,
    bin_training_rows,
    collect_run_data,
    count_value as _count_value,
    fmt_rate as _fmt_rate,
    mean_or_none as _mean,
    read_manifest,
    safe_float as _safe_float,
    write_csv as _write_csv,
)
from scripts.analysis.pcd_artifact_framework import (
    ArtifactRegistry,
    ArtifactSpec,
    write_artifacts_manifest,
    write_dashboard_html,
)


CANONICAL_VARIANTS = (
    "deplot_no_vs_opd",
    "deplot_no_vs_opd_va",
    "deplot_no_vs_opd_pcd",
    "deplot_no_vs_opd_va_pcd",
)

VARIANT_LABELS = {
    "deplot_no_vs_opd": "anchor",
    "deplot_no_vs_opd_va": "VA",
    "deplot_no_vs_opd_pcd": "Routed OPD",
    "deplot_no_vs_opd_va_pcd": "VA + Routed OPD",
}

VARIANT_COLORS = {
    "deplot_no_vs_opd": "#565656",
    "deplot_no_vs_opd_va": "#b48a00",
    "deplot_no_vs_opd_pcd": "#2f6f73",
    "deplot_no_vs_opd_va_pcd": "#7a4e9d",
}

PAPER_COLORS = {
    "anchor": "#4d4d4d",
    "pcd": "#2f6f73",
    "probe": "#9b9b9b",
    "correct": "#2f6f73",
    "risk": "#b5483c",
    "grid": "#d8d8d8",
}

METRIC_SHORT_LABELS = {
    "grpo_route_rate": "GRPO",
    "opd_route_rate": "OPD",
    "sft_route_rate": "SFT",
}


def _candidate_group_type(record: dict[str, Any]) -> str:
    if record.get("group_all_wrong") is True or record.get("is_all_wrong_probe_candidate") is True:
        return "all_wrong"
    if record.get("is_mixed_wrong_probe_candidate") is True or record.get("group_has_correct") is True:
        return "mixed_wrong"
    return "unknown"


def summarize_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_candidate_group_type(record), []).append(record)
    rows: list[dict[str, Any]] = []
    for group_type, group in sorted(grouped.items()):
        n = len(group)
        teacher_correct = sum(1 for row in group if row.get("teacher_correct") is True)
        parse_failed = sum(1 for row in group if row.get("parse_failed") is True)
        placeholders = sum(1 for row in group if row.get("teacher_output_is_placeholder") is True)
        rows.append(
            {
                "group_type": group_type,
                "n": n,
                "teacher_correct_rate": teacher_correct / n if n else None,
                "parse_fail_rate": parse_failed / n if n else None,
                "placeholder_rate": placeholders / n if n else None,
            }
        )
    return rows


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _save_figure(fig: Any, out_dir: Path, stem: str, *, svg: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")
    if svg:
        fig.savefig(out_dir / f"{stem}.svg")


def _fmt_count(value: float | int | None) -> str:
    return "" if value is None else str(int(value))


def _compact_count(value: float | int | None) -> str:
    if value is None:
        return ""
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def _variant_label_list(variants: tuple[str, ...] | list[str]) -> list[str]:
    return [VARIANT_LABELS.get(variant, variant) for variant in variants]


def _variant_color(variant: str) -> str:
    return VARIANT_COLORS.get(variant, "#565656")


def _candidate_counts(payload: dict[str, Any]) -> tuple[int, int]:
    records = payload.get("candidates", [])
    return len(records), sum(1 for record in records if record.get("teacher_correct") is True)


def _paper_axes(ax: Any, *, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if y_grid:
        ax.grid(True, axis="y", color=PAPER_COLORS["grid"], alpha=0.45, linewidth=0.45)
        ax.set_axisbelow(True)


def _panel_label(ax: Any, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _write_booktabs_table(path: Path, headers: list[str], rows: list[list[Any]], *, highlight_first_cell: str | None = None) -> None:
    lines = [
        "```latex",
        "\\begin{tabular}{" + "l" + "c" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        escaped = [_latex_escape(cell) for cell in row]
        if highlight_first_cell and str(row[0]) == highlight_first_cell:
            escaped[0] = f"\\rowcolor{{gray!10}} {escaped[0]}"
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _last_binned_value(data: dict[str, dict[str, Any]], variant: str, metric: str) -> float | None:
    rows = [row for row in bin_training_rows(data.get(variant, {}).get("train", [])) if row.get(metric) is not None]
    return float(rows[-1][metric]) if rows else None


def make_fig0(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "variant",
        "step_bin",
        "loss_mean",
        "opsd_loss_mean",
        "reward_mean",
        "accuracy_reward_mean",
        "format_reward_mean",
        "reward_std_mean",
        "grpo_zero_loss_rate",
        "advantage_abs_mean",
        "opsd_effective_weight",
        "opsd_adaptive_multiplier",
        "completion_clipped_rate",
        "completion_eos_rate",
        "degenerate_rate",
    ]
    csv_rows: list[dict[str, Any]] = []
    for variant in CANONICAL_VARIANTS:
        for row in bin_training_rows(data.get(variant, {}).get("train", [])):
            csv_rows.append({"variant": variant, **{field: _fmt_rate(row.get(field)) for field in fields if field not in {"variant", "step_bin"}}, "step_bin": row["step_bin"]})
    _write_csv(out_dir / "fig0_training_basics.csv", csv_rows, fields)

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), squeeze=False)
    panels = [
        ("Training loss", "loss_mean"),
        ("Accuracy reward", "accuracy_reward_mean"),
        ("Reward std", "reward_std_mean"),
        ("Degenerate completion rate", "degenerate_rate"),
    ]
    for ax, (title, metric) in zip([ax for row in axes for ax in row], panels):
        for variant in CANONICAL_VARIANTS:
            rows = bin_training_rows(data.get(variant, {}).get("train", []))
            x = [row["step_bin"] for row in rows if row.get(metric) is not None]
            y = [row[metric] for row in rows if row.get(metric) is not None]
            if x:
                ax.plot(x, y, linewidth=1.4, color=_variant_color(variant), label=VARIANT_LABELS.get(variant, variant))
        ax.set_title(title)
        ax.set_xlabel("step bin")
        _paper_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig0_training_basics")
    plt.close(fig)


def make_fig1(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    csv_rows: list[dict[str, Any]] = []
    for variant in ("deplot_no_vs_opd", "deplot_no_vs_opd_pcd"):
        for row in bin_training_rows(data.get(variant, {}).get("train", [])):
            for metric in ("group_all_wrong_rate", "group_mixed_rate", "reward_std_lt_0_05_rate"):
                csv_rows.append(
                    {
                        "panel": "reward_sparsity",
                        "variant": variant,
                        "step_bin": row["step_bin"],
                        "group_type": "",
                        "metric": metric,
                        "value": _fmt_rate(row.get(metric)),
                        "n": "",
                        "data_quality": "exact" if row.get(metric) is not None else "missing",
                    }
                )
    for variant, payload in data.items():
        candidates = payload.get("candidates", [])
        candidate_quality = "exact" if candidates and "group_all_wrong" in candidates[0] else "proxy"
        probe_count, correct_count = _candidate_counts(payload)
        for metric, value in (("probe_candidate_count", probe_count), ("teacher_correct_count", correct_count)):
            csv_rows.append({"panel": "candidate_volume", "variant": variant, "step_bin": "", "group_type": "", "metric": metric, "value": str(value), "n": probe_count, "data_quality": candidate_quality})
        for row in summarize_candidates(candidates):
            for metric in ("teacher_correct_rate", "parse_fail_rate", "placeholder_rate"):
                csv_rows.append({"panel": "teacher_rescue", "variant": variant, "step_bin": "", "group_type": row["group_type"], "metric": metric, "value": _fmt_rate(row.get(metric)), "n": row["n"], "data_quality": candidate_quality})
    _write_csv(out_dir / "fig1_motivation.csv", csv_rows, ["panel", "variant", "step_bin", "group_type", "metric", "value", "n", "data_quality"])
    _write_fig1_cases(data, out_dir / "fig1_cases.md")

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1), gridspec_kw={"width_ratios": [1.15, 1.0, 1.2]})
    ax_volume, ax_rescue, ax_cases = axes
    variants = [variant for variant in CANONICAL_VARIANTS if variant in data]
    x_positions = list(range(len(variants)))
    width = 0.34
    probe_counts = [_candidate_counts(data.get(variant, {}))[0] for variant in variants]
    correct_counts = [_candidate_counts(data.get(variant, {}))[1] for variant in variants]
    ax_volume.bar([x - width / 2 for x in x_positions], probe_counts, width=width, color=PAPER_COLORS["probe"], label="probed")
    ax_volume.bar([x + width / 2 for x in x_positions], correct_counts, width=width, color=PAPER_COLORS["correct"], label="teacher-correct")
    if any(value > 0 for value in probe_counts + correct_counts):
        ax_volume.set_yscale("log")
    _panel_label(ax_volume, "a", "candidate volume")
    ax_volume.set_ylabel("count (log)")
    ax_volume.set_xticks(x_positions)
    ax_volume.set_xticklabels(_variant_label_list(variants), rotation=20, ha="right")
    ax_volume.legend(frameon=False, loc="upper left", handlelength=1.0)
    _paper_axes(ax_volume)
    if "deplot_no_vs_opd_pcd" in variants:
        pcd_index = variants.index("deplot_no_vs_opd_pcd")
        ax_volume.text(pcd_index + width / 2, max(correct_counts[pcd_index], 1) * 1.35, f"{correct_counts[pcd_index]:,}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=PAPER_COLORS["pcd"])

    rates = [correct / probe if probe else 0.0 for probe, correct in zip(probe_counts, correct_counts)]
    bars = ax_rescue.bar(x_positions, rates, color=[_variant_color(variant) for variant in variants], width=0.62)
    _panel_label(ax_rescue, "b", "recoverability rate")
    ax_rescue.set_ylabel("teacher-correct / probed")
    ax_rescue.set_ylim(0, max(0.35, max(rates or [0]) * 1.2))
    ax_rescue.set_xticks(x_positions)
    ax_rescue.set_xticklabels(_variant_label_list(variants), rotation=20, ha="right")
    _paper_axes(ax_rescue)
    for bar, rate, variant in zip(bars, rates, variants):
        if variant in {"deplot_no_vs_opd", "deplot_no_vs_opd_pcd"}:
            ax_rescue.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{rate:.3f}", ha="center", va="bottom", fontsize=7)
    if "deplot_no_vs_opd" in variants and "deplot_no_vs_opd_pcd" in variants:
        anchor_i = variants.index("deplot_no_vs_opd")
        pcd_i = variants.index("deplot_no_vs_opd_pcd")
        probe_mult = _safe_ratio(float(probe_counts[pcd_i]), float(probe_counts[anchor_i]))
        correct_mult = _safe_ratio(float(correct_counts[pcd_i]), float(correct_counts[anchor_i]))
        if probe_mult is not None and correct_mult is not None:
            ax_rescue.text(0.03, 0.93, f"Routed OPD: {probe_mult:.1f}x probes\n{correct_mult:.1f}x rescued", transform=ax_rescue.transAxes, ha="left", va="top", fontsize=7, color=PAPER_COLORS["pcd"])

    _panel_label(ax_cases, "c", "qualitative routing")
    ax_cases.axis("off")
    archetypes = [("mixed wrong", "teacher-correct", "OPD"), ("all-wrong", "teacher-correct", "OPD"), ("wrong", "teacher-wrong", "SFT")]
    seen_records: dict[tuple[str, str, str], bool] = {}
    for payload in data.values():
        for record in payload.get("candidates", []):
            group = "all-wrong" if record.get("group_all_wrong") is True or record.get("is_all_wrong_probe_candidate") is True else "mixed wrong"
            teacher = "teacher-correct" if record.get("teacher_correct") is True else "teacher-wrong"
            route = "OPD" if record.get("final_route") == "opd" else "SFT"
            seen_records[(group, teacher, route)] = True
            seen_records[("wrong", teacher, route)] = True
    table_data = [["completion", "teacher probe", "route"]]
    for archetype in archetypes:
        route_text = archetype[2] if seen_records.get(archetype) else archetype[2] + "*"
        table_data.append([archetype[0], archetype[1], route_text])
    table = ax_cases.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.36, 0.42, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#bbbbbb")
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        elif col == 2 and "OPD" in cell.get_text().get_text():
            cell.get_text().set_color(PAPER_COLORS["pcd"])
            cell.get_text().set_fontweight("bold")
        elif col == 2 and "SFT" in cell.get_text().get_text():
            cell.get_text().set_color(PAPER_COLORS["risk"])
    ax_cases.text(0.0, -0.06, "* archetype shown when exact route example is absent", transform=ax_cases.transAxes, fontsize=6.5, color="#666666")
    fig.tight_layout(w_pad=1.0)
    _save_figure(fig, out_dir, "fig1_motivation")
    plt.close(fig)


def _write_fig1_cases(data: dict[str, dict[str, Any]], path: Path) -> None:
    cases: list[tuple[str, dict[str, Any]]] = []
    for payload in data.values():
        for record in payload.get("candidates", []):
            if len(cases) >= 3:
                break
            label = ""
            if record.get("student_correct") is False and record.get("teacher_correct") is True:
                label = "student_wrong_teacher_correct"
            elif record.get("student_correct") is False and record.get("teacher_correct") is False:
                label = "student_wrong_teacher_wrong"
            if record.get("group_all_wrong") is True and record.get("teacher_correct") is True:
                label = "all_wrong_teacher_rescued"
            if label and all(existing != label for existing, _ in cases):
                cases.append((label, record))
    lines = ["# Figure 1 Qualitative Cases", ""]
    for label, record in cases:
        lines.extend([f"## {label}", f"- image: {record.get('image', '')}", f"- question: {record.get('question', '')}", f"- reference: {record.get('reference', '')}", f"- student_output: {record.get('student_output', '')}", f"- teacher_output: {record.get('teacher_output', '')}", f"- final_route: {record.get('final_route', '')}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_fig4(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    variants = ("deplot_no_vs_opd", "deplot_no_vs_opd_pcd", "deplot_no_vs_opd_va_pcd")
    fields = ["variant", "step_bin", "reward_mean", "accuracy_reward_mean", "format_reward_mean", "reward_std_mean", "group_all_wrong_rate", "grpo_route_rate", "opd_route_rate", "sft_route_rate", "teacher_probe_candidate_rate", "teacher_correct_rate", "opsd_effective_weight", "opsd_adaptive_multiplier", "degenerate_rate"]
    csv_rows: list[dict[str, Any]] = []
    for variant in variants:
        for row in bin_training_rows(data.get(variant, {}).get("train", [])):
            csv_rows.append({"variant": variant, **{field: _fmt_rate(row.get(field)) for field in fields if field not in {"variant", "step_bin"}}, "step_bin": row["step_bin"]})
    _write_csv(out_dir / "fig4_training_dynamics.csv", csv_rows, fields)

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.85), sharex=True)
    panels = [("a", "accuracy reward", "accuracy_reward_mean", "reward"), ("b", "reward std", "reward_std_mean", "std"), ("c", "degenerate rate", "degenerate_rate", "rate")]
    highlight = {"deplot_no_vs_opd", "deplot_no_vs_opd_pcd"}
    pcd_rows = bin_training_rows(data.get("deplot_no_vs_opd_pcd", {}).get("train", []))
    active_steps = [row["step_bin"] for row in pcd_rows if (row.get("teacher_probe_candidate_rate") or 0) > 0.01 or (row.get("opd_route_rate") or 0) > 0.01]
    active_step = active_steps[0] if active_steps else None
    for ax, (panel, title, metric, ylabel) in zip(axes, panels):
        for variant in CANONICAL_VARIANTS:
            rows_for_variant = bin_training_rows(data.get(variant, {}).get("train", []))
            points = [(row["step_bin"], row[metric]) for row in rows_for_variant if row.get(metric) is not None]
            if not points:
                continue
            x = [point[0] for point in points]
            y = [float(point[1]) for point in points]
            color = _variant_color(variant)
            if variant in highlight:
                ax.plot(x, y, color=color, alpha=0.16, linewidth=0.8)
                ax.plot(x, _moving_average(y), color=color, linewidth=1.7, label=VARIANT_LABELS.get(variant, variant))
            else:
                ax.plot(x, _moving_average(y), color=color, alpha=0.32, linewidth=1.0, linestyle="--")
        if active_step is not None:
            ax.axvline(active_step, color="#888888", linewidth=0.6, linestyle=":", alpha=0.8)
        _panel_label(ax, panel, title)
        ax.set_xlabel("step bin")
        ax.set_ylabel(ylabel)
        _paper_axes(ax)
    if active_step is not None:
        axes[0].text(active_step, axes[0].get_ylim()[1], "probe active", rotation=90, va="top", ha="right", fontsize=6.5, color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, loc="best", handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    _save_figure(fig, out_dir, "fig4_training_dynamics")
    plt.close(fig)


def make_fig5(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for variant, payload in data.items():
        totals = {key: 0.0 for key in COUNT_ALIASES}
        for train_row in payload.get("train", []):
            for key in totals:
                value = _count_value(train_row, key)
                if value is not None:
                    totals[key] += value
        candidates = payload.get("candidates", [])
        if totals["probe_candidate_count"] == 0 and candidates:
            totals["probe_candidate_count"] = len(candidates)
            totals["teacher_correct_count"] = sum(1 for rec in candidates if rec.get("teacher_correct") is True)
            totals["opd_route_count"] = sum(1 for rec in candidates if rec.get("final_route") == "opd")
            totals["sft_route_count"] = sum(1 for rec in candidates if str(rec.get("final_route", "")).startswith("sft"))
        has_exact_totals = totals["total_completion_count"] > 0 and totals["wrong_completion_count"] > 0
        probe = totals["probe_candidate_count"]
        correct = totals["teacher_correct_count"]
        sft_fallback = max(probe - correct, 0.0)
        rows.append({"variant": variant, "funnel_scope": "exact_completion_counts" if has_exact_totals else "candidate_proxy", "total_completion_count": _fmt_count(totals["total_completion_count"] if has_exact_totals else None), "wrong_completion_count": _fmt_count(totals["wrong_completion_count"] if has_exact_totals else None), "probe_candidate_count": _fmt_count(probe), "teacher_correct_count": _fmt_count(correct), "opd_route_count": _fmt_count(totals["opd_route_count"]), "sft_fallback_count": _fmt_count(sft_fallback), "probe_candidate_rate": _fmt_rate(probe / totals["wrong_completion_count"] if totals["wrong_completion_count"] else None), "teacher_correct_given_probe_rate": _fmt_rate(correct / probe if probe else None), "opd_given_teacher_correct_rate": _fmt_rate(totals["opd_route_count"] / correct if correct else None), "sft_fallback_given_probe_rate": _fmt_rate(sft_fallback / probe if probe else None)})
    fields = ["variant", "funnel_scope", "total_completion_count", "wrong_completion_count", "probe_candidate_count", "teacher_correct_count", "opd_route_count", "sft_fallback_count", "probe_candidate_rate", "teacher_correct_given_probe_rate", "opd_given_teacher_correct_rate", "sft_fallback_given_probe_rate"]
    _write_csv(out_dir / "fig5_teacher_rescue_funnel.csv", rows, fields)

    main = next((row for row in rows if row["variant"] == "deplot_no_vs_opd_pcd"), rows[0] if rows else None)
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.25), gridspec_kw={"width_ratios": [1.55, 1.0]})
    if main:
        labels = ["Probed", "Teacher-correct", "OPD", "SFT fallback"]
        values = [int(main["probe_candidate_count"] or 0), int(main["teacher_correct_count"] or 0), int(main["opd_route_count"] or 0), int(main["sft_fallback_count"] or 0)]
        axes[0].barh(labels, values, color=["#b7b7b7", "#6aa59d", PAPER_COLORS["pcd"], "#c7b9d8"], height=0.58)
        axes[0].invert_yaxis()
        _panel_label(axes[0], "a", "teacher-rescue funnel")
        axes[0].set_xlabel("candidate count")
        _paper_axes(axes[0])
        for index, value in enumerate(values):
            axes[0].text(value, index, f" {_compact_count(value)}", va="center", fontsize=7)
        if main.get("funnel_scope") != "exact_completion_counts":
            axes[0].text(0.02, -0.16, "candidate-log proxy", transform=axes[0].transAxes, fontsize=6.5, color="#666666")

    compare_rows = [row for row in rows if row["variant"] in {"deplot_no_vs_opd", "deplot_no_vs_opd_pcd"}]
    compare_labels = [VARIANT_LABELS.get(row["variant"], row["variant"]) for row in compare_rows]
    x_positions = list(range(len(compare_labels)))
    width = 0.32
    probe_counts = [int(row["probe_candidate_count"] or 0) for row in compare_rows]
    correct_counts = [int(row["teacher_correct_count"] or 0) for row in compare_rows]
    axes[1].bar([x - width / 2 for x in x_positions], probe_counts, width=width, color=PAPER_COLORS["probe"], label="probed")
    axes[1].bar([x + width / 2 for x in x_positions], correct_counts, width=width, color=PAPER_COLORS["correct"], label="teacher-correct")
    if any(value > 0 for value in probe_counts + correct_counts):
        axes[1].set_yscale("log")
    _panel_label(axes[1], "b", "anchor vs routed OPD")
    axes[1].set_ylabel("count (log)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(compare_labels, rotation=0)
    _paper_axes(axes[1])
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.0)
    if len(probe_counts) == 2 and probe_counts[0] and correct_counts[0]:
        axes[1].text(0.98, 0.94, f"{probe_counts[1] / probe_counts[0]:.1f}x probes\n{correct_counts[1] / correct_counts[0]:.1f}x rescued", transform=axes[1].transAxes, ha="right", va="top", fontsize=7, fontweight="bold", color=PAPER_COLORS["pcd"])
    fig.tight_layout(w_pad=1.4)
    _save_figure(fig, out_dir, "fig5_teacher_rescue_funnel")
    plt.close(fig)


def make_fig6(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    fields = ["variant", "step_bin", "reward_std_mean", "opsd_adaptive_multiplier", "opsd_effective_weight", "opd_route_rate", "sft_route_rate", "teacher_correct_rate", "final_accuracy", "final_full_cot_rate", "final_other_rate"]
    rows: list[dict[str, Any]] = []
    for variant in CANONICAL_VARIANTS:
        final_eval = data.get(variant, {}).get("eval", {})
        binned_rows = bin_training_rows(data.get(variant, {}).get("train", []))
        if not binned_rows and final_eval:
            rows.append({"variant": variant, "step_bin": "", "reward_std_mean": "", "opsd_adaptive_multiplier": "", "opsd_effective_weight": "", "opd_route_rate": "", "sft_route_rate": "", "teacher_correct_rate": "", "final_accuracy": _fmt_rate(final_eval.get("accuracy")), "final_full_cot_rate": _fmt_rate(final_eval.get("full_cot_rate")), "final_other_rate": _fmt_rate(final_eval.get("other_rate"))})
            continue
        for row in binned_rows:
            rows.append({"variant": variant, "step_bin": row["step_bin"], "reward_std_mean": _fmt_rate(row.get("reward_std_mean")), "opsd_adaptive_multiplier": _fmt_rate(row.get("opsd_adaptive_multiplier")), "opsd_effective_weight": _fmt_rate(row.get("opsd_effective_weight")), "opd_route_rate": _fmt_rate(row.get("opd_route_rate")), "sft_route_rate": _fmt_rate(row.get("sft_route_rate")), "teacher_correct_rate": _fmt_rate(row.get("teacher_correct_rate")), "final_accuracy": _fmt_rate(final_eval.get("accuracy")), "final_full_cot_rate": _fmt_rate(final_eval.get("full_cot_rate")), "final_other_rate": _fmt_rate(final_eval.get("other_rate"))})
    _write_csv(out_dir / "fig6_va_vs_pcd_diagnosis.csv", rows, fields)

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05))
    for variant in ("deplot_no_vs_opd_va", "deplot_no_vs_opd_va_pcd"):
        vrows = [row for row in rows if row["variant"] == variant and row["step_bin"] != ""]
        x = [int(row["step_bin"]) for row in vrows]
        y = [float(row["opsd_adaptive_multiplier"] or 0.0) for row in vrows]
        if x:
            axes[0].plot(x, y, color=VARIANT_COLORS.get(variant), alpha=0.18, linewidth=0.8)
            axes[0].plot(x, _moving_average(y), linewidth=1.5, color=VARIANT_COLORS.get(variant), label=VARIANT_LABELS.get(variant, variant))
    _panel_label(axes[0], "a", "VA weight response")
    axes[0].set_xlabel("step bin")
    axes[0].set_ylabel("adaptive multiplier")
    _paper_axes(axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, loc="best", handlelength=1.4)

    last_by_variant = {variant: next((row for row in reversed(rows) if row["variant"] == variant), {}) for variant in CANONICAL_VARIANTS}
    x = list(range(len(CANONICAL_VARIANTS)))
    route_values = [float(last_by_variant[v].get("opd_route_rate") or 0.0) for v in CANONICAL_VARIANTS]
    correct_counts = [_candidate_counts(data.get(v, {}))[1] for v in CANONICAL_VARIANTS]
    width = 0.34
    axes[1].bar([i - width / 2 for i in x], route_values, width=width, color=[VARIANT_COLORS.get(v, "#2f6f73") for v in CANONICAL_VARIANTS], label="OPD route")
    ax1b = axes[1].twinx()
    ax1b.plot(x, correct_counts, color=PAPER_COLORS["correct"], marker="o", linewidth=1.3, label="teacher-correct")
    _panel_label(axes[1], "b", "recoverability routing")
    axes[1].set_ylabel("OPD route rate")
    ax1b.set_ylabel("teacher-correct count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([VARIANT_LABELS.get(v, v) for v in CANONICAL_VARIANTS], rotation=20, ha="right")
    _paper_axes(axes[1])
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["left"].set_visible(False)
    ax1b.spines["right"].set_linewidth(0.6)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", handlelength=1.2)
    pcd_index = list(CANONICAL_VARIANTS).index("deplot_no_vs_opd_pcd")
    if route_values[pcd_index] > 0:
        axes[1].text(pcd_index - width / 2, route_values[pcd_index] * 0.55, f"{route_values[pcd_index]:.3f}", ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    acc_values = [float(last_by_variant[v].get("final_accuracy") or 0.0) for v in CANONICAL_VARIANTS]
    other_values = [float(last_by_variant[v].get("final_other_rate") or 0.0) for v in CANONICAL_VARIANTS]
    axes[2].bar([i - width / 2 for i in x], acc_values, width=width, color=[VARIANT_COLORS.get(v, "#6f55a0") for v in CANONICAL_VARIANTS], label="Accuracy")
    axes[2].bar([i + width / 2 for i in x], other_values, width=width, color="#b85c5c", label="Other output")
    _panel_label(axes[2], "c", "final behavior")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([VARIANT_LABELS.get(v, v) for v in CANONICAL_VARIANTS], rotation=20, ha="right")
    axes[2].set_ylabel("rate")
    _paper_axes(axes[2])
    axes[2].legend(frameon=False, loc="upper left", handlelength=1.2)
    for index, value in enumerate(acc_values):
        if CANONICAL_VARIANTS[index] in {"deplot_no_vs_opd", "deplot_no_vs_opd_pcd"}:
            axes[2].text(index - width / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    for index, value in enumerate(other_values):
        if CANONICAL_VARIANTS[index] == "deplot_no_vs_opd_va_pcd":
            axes[2].text(index + width / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color=PAPER_COLORS["risk"], fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    _save_figure(fig, out_dir, "fig6_va_vs_pcd_diagnosis")
    plt.close(fig)


def make_table1(out_dir: Path) -> None:
    rows = [
        ("SFT", "No", "Only as targets", "No", "No", "No", "Optional", "Cannot exploit online correctness signal"),
        ("GRPO/RLVR", "Yes", "Discarded or negative", "Group-level", "No", "Yes", "Optional", "Sparse rewards leave all-wrong groups uninformative"),
        ("GRPO + filtering", "Yes", "Filtered", "Partial", "No", "Yes", "Optional", "May throw away recoverable wrong completions"),
        ("Distillation", "Teacher target", "Teacher-forced", "No", "No", "Depends", "Optional", "Risk of copying teacher errors or privileged traces"),
        ("OPD-style correction", "Yes", "Yes", "Often prompt-level", "Weak", "Yes", "Optional", "Can over-correct unrecoverable wrong outputs"),
        ("Verifier-routed OPD", "Yes", "Routed", "Yes", "Yes", "Yes", "Yes", "Requires teacher-probe budget"),
    ]
    header = ["Method", "Learns from correct completions", "Uses wrong completions", "Completion-level routing", "Recoverability gate", "Avoids gold CoT", "Uses visual evidence", "Main limitation"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    (out_dir / "table1_method_positioning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deplot_real_rate(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    real = 0
    for record in records:
        privileged = record.get("privileged") if isinstance(record.get("privileged"), dict) else {}
        if privileged.get("visual_fact_deplot_status") == "real":
            real += 1
    return real / len(records)


def _teacher_output_word_count(record: dict[str, Any]) -> float:
    explicit = _safe_float(record.get("teacher_output_word_count"))
    if explicit is not None:
        return explicit
    text = str(record.get("teacher_output") or "").replace("\\r\\n", " ").replace("\\n", " ")
    return float(len(text.split()))


def make_table3(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for variant, payload in data.items():
        records = payload.get("candidates", [])
        summary = summarize_candidates(records)
        n = sum(int(row["n"]) for row in summary)
        teacher_correct = sum(int(row["n"]) * float(row["teacher_correct_rate"] or 0.0) for row in summary)
        parse_fail = sum(int(row["n"]) * float(row["parse_fail_rate"] or 0.0) for row in summary)
        placeholder = sum(int(row["n"]) * float(row["placeholder_rate"] or 0.0) for row in summary)
        word_counts = [_teacher_output_word_count(rec) for rec in records]
        rows.append({"section": "by_variant", "control": payload.get("manifest", {}).get("role") or variant, "variant": variant, "n": n, "teacher_correct_rate": _fmt_rate(teacher_correct / n if n else None), "parse_fail_rate": _fmt_rate(parse_fail / n if n else None), "placeholder_rate": _fmt_rate(placeholder / n if n else None), "teacher_output_words_mean": _fmt_rate(_mean(word_counts)), "deplot_real_rate": _fmt_rate(_deplot_real_rate(records)), "status": "from_candidate_log" if records else "missing"})
    fields = ["section", "control", "variant", "n", "teacher_correct_rate", "parse_fail_rate", "placeholder_rate", "teacher_output_words_mean", "deplot_real_rate", "status"]
    _write_csv(out_dir / "table3_recoverability_controls.csv", rows, fields)
    table_rows = []
    for row in rows:
        label = "Routed OPD" if row["variant"] == "deplot_no_vs_opd_pcd" else VARIANT_LABELS.get(row["variant"], row["control"])
        table_rows.append([label, row["n"], row["teacher_correct_rate"], row["parse_fail_rate"], row["placeholder_rate"], row["deplot_real_rate"]])
    _write_booktabs_table(out_dir / "table3_recoverability_controls.md", ["Method", "Probe Cand. ↑", "Recover ↑", "Parse Fail ↓", "Placeholder ↓", "DePlot Real ↑"], table_rows, highlight_first_cell="Routed OPD")


def make_table4(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    fields = ["variant", "visual_supervision_on", "gold_cot_used", "deplot_evidence_on", "teacher_probe_gold_suffix_rate", "teacher_probe_deplot_real_rate", "teacher_probe_skipped_no_evidence_rate", "group_all_wrong_rate", "wrong_completion_rate", "teacher_probe_candidate_rate", "teacher_correct_rate", "grpo_route_rate", "opd_route_rate", "sft_route_rate", "generated_tokens_mean", "generated_tokens_p95", "data_quality"]
    aliases = {
        "teacher_probe_gold_suffix_rate": ("routing/teacher_probe_gold_suffix_rate",),
        "teacher_probe_deplot_real_rate": ("routing/teacher_probe_deplot_real_rate",),
        "teacher_probe_skipped_no_evidence_rate": ("routing/teacher_probe_skipped_no_evidence_rate",),
        "group_all_wrong_rate": ("signal/group_all_wrong_rate",),
        "wrong_completion_rate": ("routing/wrong_completion_rate",),
        "teacher_probe_candidate_rate": ("routing/teacher_probe_candidate_rate",),
        "teacher_correct_rate": ("routing/teacher_probe_correct_rate",),
        "grpo_route_rate": ("global_signal/grpo_route_rate", "routing/grpo_route_rate", "routing/grpo_on_correct_rate"),
        "opd_route_rate": ("global_signal/opd_route_rate", "routing/opd_route_rate", "routing/opd_teacher_call_rate"),
        "sft_route_rate": ("global_signal/sft_route_rate", "routing/sft_route_rate", "routing/sft_replaced_ratio"),
        "generated_tokens_mean": ("teacher_probe/generated_tokens_mean",),
        "generated_tokens_p95": ("teacher_probe/generated_tokens_p95",),
    }
    rows: list[dict[str, Any]] = []
    for variant, payload in data.items():
        train_rows = payload.get("train", [])
        out: dict[str, Any] = {"variant": variant}
        for key, key_aliases in aliases.items():
            values = []
            for row in train_rows:
                for alias in key_aliases:
                    value = _safe_float(row.get(alias))
                    if value is not None:
                        values.append(value)
                        break
            out[key] = _fmt_rate(_mean(values))
        config_path = payload.get("manifest", {}).get("config_path")
        config_text = config_path.read_text(encoding="utf-8", errors="replace").lower() if config_path and hasattr(config_path, "exists") and config_path.exists() else ""
        out["visual_supervision_on"] = "false" if "visual_supervision=false" in config_text else "unknown"
        out["gold_cot_used"] = "true" if '"text_include_gold": true' in config_text or "'text_include_gold': true" in config_text else "false"
        out["deplot_evidence_on"] = "true" if "deplot" in config_text or (_deplot_real_rate(payload.get("candidates", [])) or 0) > 0 else "unknown"
        exact_train = bool(train_rows and all(_count_value(train_rows[0], key) is not None for key in COUNT_ALIASES))
        exact_candidate = bool(payload.get("candidates") and "group_all_wrong" in payload["candidates"][0])
        out["data_quality"] = "exact" if exact_train and exact_candidate else "proxy"
        rows.append(out)
    _write_csv(out_dir / "table4_routing_antileakage.csv", rows, fields)
    table_rows = []
    for row in rows:
        variant = row["variant"]
        eval_summary = data.get(variant, {}).get("eval", {})
        label = "Routed OPD" if variant == "deplot_no_vs_opd_pcd" else VARIANT_LABELS.get(variant, variant)
        table_rows.append([label, _fmt_rate(eval_summary.get("accuracy")), _fmt_rate(eval_summary.get("other_rate")), row["gold_cot_used"], row["teacher_probe_gold_suffix_rate"], row["teacher_probe_candidate_rate"], row["teacher_correct_rate"], row["opd_route_rate"], row["data_quality"]])
    _write_booktabs_table(out_dir / "table4_routing_antileakage.md", ["Method", "Acc ↑", "Other ↓", "Gold CoT", "Gold Suffix ↓", "Probe Cand. ↑", "Teacher Correct ↑", "OPD Route ↑", "Quality"], table_rows, highlight_first_cell="Routed OPD")


def _candidate_summary(payload: dict[str, Any]) -> dict[str, float | None]:
    records = payload.get("candidates", [])
    if not records:
        return {"probe_count": None, "teacher_correct_count": None, "teacher_correct_rate": None, "opd_count": None, "sft_fallback_count": None}
    probe_count = float(len(records))
    teacher_correct_count = float(sum(1 for rec in records if rec.get("teacher_correct") is True))
    return {"probe_count": probe_count, "teacher_correct_count": teacher_correct_count, "teacher_correct_rate": _safe_ratio(teacher_correct_count, probe_count), "opd_count": float(sum(1 for rec in records if rec.get("final_route") == "opd")), "sft_fallback_count": float(sum(1 for rec in records if str(rec.get("final_route", "")).startswith("sft")))}


def _final_eval(payload: dict[str, Any], key: str) -> float | None:
    return payload.get("eval", {}).get(key)


def make_argument_report(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    anchor = data.get("deplot_no_vs_opd", {})
    pcd = data.get("deplot_no_vs_opd_pcd", {})
    va = data.get("deplot_no_vs_opd_va", {})
    va_pcd = data.get("deplot_no_vs_opd_va_pcd", {})
    anchor_summary = _candidate_summary(anchor)
    pcd_summary = _candidate_summary(pcd)
    va_summary = _candidate_summary(va)
    va_pcd_summary = _candidate_summary(va_pcd)
    probe_mult = _safe_ratio(pcd_summary["probe_count"], anchor_summary["probe_count"])
    correct_mult = _safe_ratio(pcd_summary["teacher_correct_count"], anchor_summary["teacher_correct_count"])
    anchor_acc = _final_eval(anchor, "accuracy")
    pcd_acc = _final_eval(pcd, "accuracy")
    va_acc = _final_eval(va, "accuracy")
    va_pcd_acc = _final_eval(va_pcd, "accuracy")
    pcd_gain = pcd_acc - anchor_acc if pcd_acc is not None and anchor_acc is not None else None
    va_gain = va_acc - anchor_acc if va_acc is not None and anchor_acc is not None else None
    va_pcd_gain = va_pcd_acc - anchor_acc if va_pcd_acc is not None and anchor_acc is not None else None
    lines = [
        "# Verifier-Routed OPD Paper Argument Report",
        "",
        "## Recommended paper claim",
        "",
        "Verifier-routed OPD should be framed as completion-level recoverability selection over student-generated states: it expands the set of wrong completions that can be checked by a teacher, applies OPD to the teacher-correct subset, and leaves unrecoverable cases on the matched fallback path. Verifier-routed OPD is the main mechanism; variance-adaptive weighting remains an auxiliary ablation.",
        "",
        "## Evidence from current artifacts",
        "",
        "| Claim | Evidence | Figure/Table | Caveat |",
        "| --- | --- | --- | --- |",
        f"| Routed OPD expands recoverable wrong-completion supervision | Probe candidates increase from {_fmt_count(anchor_summary['probe_count'])} to {_fmt_count(pcd_summary['probe_count'])}{f' ({probe_mult:.1f}x)' if probe_mult is not None else ''}; teacher-correct candidates increase from {_fmt_count(anchor_summary['teacher_correct_count'])} to {_fmt_count(pcd_summary['teacher_correct_count'])}{f' ({correct_mult:.1f}x)' if correct_mult is not None else ''}. | Figure 5, Table 3 | Current funnel uses candidate-log proxy because exact total/wrong counts are missing in old logs. |",
        f"| Verifier-routed OPD is the main mechanism among the tested variants | Final accuracy: anchor={_fmt_rate(anchor_acc)}, routed OPD={_fmt_rate(pcd_acc)}, VA={_fmt_rate(va_acc)}, VA+routed OPD={_fmt_rate(va_pcd_acc)}; routed-OPD gain over anchor={_fmt_rate(pcd_gain)}, VA-only gain={_fmt_rate(va_gain)}, joint gain={_fmt_rate(va_pcd_gain)}. | Figure 6 | Treat as mechanism diagnosis unless promoted into the main-result table. |",
        f"| VA changes weighting but is not sufficient as the main story | VA-only teacher-correct candidates={_fmt_count(va_summary['teacher_correct_count'])}; VA+routed-OPD teacher-correct candidates={_fmt_count(va_pcd_summary['teacher_correct_count'])}; VA-only final accuracy={_fmt_rate(va_acc)}. | Figure 6 | Keep VA as auxiliary ablation, not the headline method. |",
        "| Anti-leakage needs a stricter exact-log rerun before final claims | Table 4 shows gold_cot_used=false and DePlot evidence on, but old logs report nonzero teacher_probe_gold_suffix_rate under proxy quality. | Table 4, data_quality_report | Treat anti-leakage as an audit target, not a concluded claim, until rerun with exact no-gold teacher-probe logging. |",
        "",
        "## Recommended figure changes already applied",
        "",
        "- Figure 1/4/5/6 now use paper-style panels with serif fonts, small multiples, light grids, and fixed semantic colors.",
        "- Figure 5 now falls back to a candidate-level funnel when exact generated/wrong completion counts are absent.",
        "- Figure 2 is intentionally removed from this artifact pipeline; method schematics should be handled outside this dashboard.",
        "- The dashboard includes this report so figure inspection and paper claims stay connected.",
    ]
    (out_dir / "paper_argument_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_chart_review_report(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    anchor = data.get("deplot_no_vs_opd", {})
    pcd = data.get("deplot_no_vs_opd_pcd", {})
    va = data.get("deplot_no_vs_opd_va", {})
    va_pcd = data.get("deplot_no_vs_opd_va_pcd", {})
    anchor_probe, anchor_correct = _candidate_counts(anchor)
    pcd_probe, pcd_correct = _candidate_counts(pcd)
    rows = [
        ("fig0_training_basics", "Appendix diagnostic; acceptable but secondary to the paper-style review figures.", "Use in paper: appendix or training sanity check.", "Shows loss, accuracy reward, reward std, and degeneration trends without overclaiming performance.", "Not a main-method proof; keep it as diagnostics."),
        ("fig1_motivation", "Paper-style review: three compact panels separate candidate volume, recoverability rate, and qualitative routing.", "Use in paper: motivation or appendix until exact all-wrong logging is rerun.", f"Routed OPD probes {pcd_probe:,} candidates and finds {pcd_correct:,} teacher-correct cases, versus anchor {anchor_probe:,}/{anchor_correct:,}.", "Old logs lack exact group_all_wrong_rate; exact rerun should replace proxy wording."),
        ("fig4_training_dynamics", "Paper-style review: small multiples highlight anchor vs routed-OPD trends while de-emphasizing auxiliary variants.", "Use in paper: mechanism diagnosis, likely appendix if main table already carries performance.", f"Routed-OPD final accuracy {_fmt_rate(pcd.get('eval', {}).get('accuracy'))} beats anchor {_fmt_rate(anchor.get('eval', {}).get('accuracy'))} while greatly increasing probe volume.", "Route/count exactness still depends on rerun quality report."),
        ("fig5_teacher_rescue_funnel", "Paper-style review: strongest mechanism figure; funnel plus anchor/routed-OPD inset directly communicates rescue amplification.", "Use in paper: main mechanism figure after exact-count rerun; current version can support draft.", f"Teacher-correct candidates grow from {anchor_correct:,} to {pcd_correct:,}; the matched fallback remains explicit.", "Current funnel scope is candidate_proxy because old logs miss generated/wrong counts."),
        ("fig6_va_vs_pcd_diagnosis", "Paper-style review: separates VA weighting, recoverability routing, and final behavior without mixing mechanisms.", "Use in paper: ablation/mechanism figure.", f"VA-only accuracy {_fmt_rate(va.get('eval', {}).get('accuracy'))} underperforms; routed OPD reaches {_fmt_rate(pcd.get('eval', {}).get('accuracy'))}; VA+routed-OPD other-output rate is {_fmt_rate(va_pcd.get('eval', {}).get('other_rate'))}.", "Use this to argue VA is auxiliary, not the main method."),
        ("table4_routing_antileakage", "Useful but not yet final: it exposes that the current logs are proxy-quality for leakage-related routing fields.", "Use in paper: anti-leakage audit table only after exact rerun.", "gold_cot_used=false is encouraging, but teacher_probe_gold_suffix_rate is nonzero in old proxy logs.", "Do not claim anti-leakage is fully proven until exact no-gold teacher-probe logs show zero gold suffix rate."),
    ]
    lines = ["# Chart Review Report", "", "This paper-style review records the manual dashboard review and the plotting changes applied to make each artifact usable for the paper argument.", "", "| Artifact | Visual assessment | Use in paper | Evidence | Caveat / next setting |", "| --- | --- | --- | --- | --- |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    (out_dir / "chart_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(data: dict[str, dict[str, Any]], out_dir: Path) -> None:
    lines = ["# Routed OPD Paper Data Quality", "", "| variant | train_counts | candidate_group_fields | eval_log |", "| --- | --- | --- | --- |"]
    for variant, payload in data.items():
        train_rows = payload.get("train", [])
        candidates = payload.get("candidates", [])
        train_quality = "exact" if train_rows and all(_count_value(train_rows[0], key) is not None for key in COUNT_ALIASES) else "missing"
        candidate_quality = "exact" if candidates and "group_all_wrong" in candidates[0] else ("proxy" if candidates else "missing")
        eval_quality = "exact" if payload.get("eval", {}).get("accuracy") is not None else "missing"
        lines.append(f"| {variant} | {train_quality} | {candidate_quality} | {eval_quality} |")
    (out_dir / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_registry() -> ArtifactRegistry:
    registry = ArtifactRegistry()
    registry.register(ArtifactSpec(id="fig0_training_basics", title="Figure 0: Training Basics", kind="figure", description="Loss, reward, reward-std, OPD/VA weights, and generation-health curves.", outputs=("fig0_training_basics.png", "fig0_training_basics.pdf", "fig0_training_basics.csv"), tags=("training", "loss", "reward", "diagnostic"), producer=lambda ctx: make_fig0(ctx.data, ctx.out_dir), aliases=("fig0", "training_basics", "basics")))
    registry.register(ArtifactSpec(id="fig1_motivation", title="Figure 1: Motivation", kind="figure", description="Wrong-completion heterogeneity and teacher-rescue evidence.", outputs=("fig1_motivation.png", "fig1_motivation.pdf", "fig1_motivation.csv", "fig1_cases.md"), tags=("motivation", "teacher-rescue"), producer=lambda ctx: make_fig1(ctx.data, ctx.out_dir), aliases=("fig1", "motivation")))
    registry.register(ArtifactSpec(id="fig4_training_dynamics", title="Figure 4: Training Dynamics", kind="figure", description="Reward, routing, and teacher-probe dynamics over training.", outputs=("fig4_training_dynamics.png", "fig4_training_dynamics.pdf", "fig4_training_dynamics.csv"), tags=("training", "routing", "teacher-probe"), producer=lambda ctx: make_fig4(ctx.data, ctx.out_dir), aliases=("fig4", "dynamics")))
    registry.register(ArtifactSpec(id="fig5_teacher_rescue_funnel", title="Figure 5: Teacher-Rescue Funnel", kind="figure", description="Generated-to-wrong-to-probed-to-rescued completion funnel.", outputs=("fig5_teacher_rescue_funnel.png", "fig5_teacher_rescue_funnel.pdf", "fig5_teacher_rescue_funnel.csv"), tags=("funnel", "teacher-rescue", "routing"), producer=lambda ctx: make_fig5(ctx.data, ctx.out_dir), aliases=("fig5", "funnel")))
    registry.register(ArtifactSpec(id="fig6_va_vs_pcd_diagnosis", title="Figure 6: VA vs Routed OPD Diagnosis", kind="figure", description="Separates adaptive OPD weighting from recoverability-routing effects.", outputs=("fig6_va_vs_pcd_diagnosis.png", "fig6_va_vs_pcd_diagnosis.pdf", "fig6_va_vs_pcd_diagnosis.csv"), tags=("ablation", "variance-adaptive", "pcd"), producer=lambda ctx: make_fig6(ctx.data, ctx.out_dir), aliases=("fig6", "va_diagnosis")))
    registry.register(ArtifactSpec(id="table1_method_positioning", title="Table 1: Method Positioning", kind="table", description="Static positioning table for SFT/RLVR/filtering/distillation/OPD/verifier-routed OPD.", outputs=("table1_method_positioning.md",), tags=("table", "related-work"), producer=lambda ctx: make_table1(ctx.out_dir), aliases=("table1",)))
    registry.register(ArtifactSpec(id="table3_recoverability_controls", title="Table 3: Recoverability Controls", kind="table", description="Compatible schema combining by-variant recoverability rows and offline probe-control rows.", outputs=("table3_recoverability_controls.csv", "table3_recoverability_controls.md"), tags=("table", "recoverability", "controls"), producer=lambda ctx: make_table3(ctx.data, ctx.out_dir), aliases=("table3",)))
    registry.register(ArtifactSpec(id="table4_routing_antileakage", title="Table 4: Routing and Anti-Leakage", kind="table", description="Routing statistics, evidence usage, and no-gold-CoT checks.", outputs=("table4_routing_antileakage.csv", "table4_routing_antileakage.md"), tags=("table", "routing", "anti-leakage"), producer=lambda ctx: make_table4(ctx.data, ctx.out_dir), aliases=("table4",)))
    registry.register(ArtifactSpec(id="paper_argument_report", title="Paper Argument Report", kind="report", description="Concise claims, supporting numbers, caveats, and next experiment setting for the paper.", outputs=("paper_argument_report.md",), tags=("paper", "claims", "diagnosis"), producer=lambda ctx: make_argument_report(ctx.data, ctx.out_dir), aliases=("argument", "claims")))
    registry.register(ArtifactSpec(id="chart_review_report", title="Chart Review Report", kind="report", description="Manual dashboard review: visual quality, paper use, evidence, and caveats for each figure.", outputs=("chart_review_report.md",), tags=("paper", "review", "figures"), producer=lambda ctx: make_chart_review_report(ctx.data, ctx.out_dir), aliases=("chart_review", "figure_review")))
    registry.register(ArtifactSpec(id="data_quality_report", title="Data Quality Report", kind="report", description="Marks exact/proxy/missing fields for each run.", outputs=("data_quality_report.md",), tags=("quality", "report"), producer=lambda ctx: write_quality_report(ctx.data, ctx.out_dir), aliases=("quality",)))
    registry.register(ArtifactSpec(id="dashboard", title="HTML Dashboard", kind="dashboard", description="Dynamic local HTML browser for all registered figures and tables.", outputs=("index.html", "artifacts_manifest.json"), tags=("dashboard", "html"), producer=None, aliases=("html", "index")))
    return registry


def run(makes: set[str], manifest: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifest(manifest)
    data = collect_run_data(manifest_rows)
    registry = build_registry()
    ctx = SimpleNamespace(data=data, out_dir=out_dir, manifest_rows=manifest_rows, manifest_path=manifest)
    selected = registry.run(makes, ctx)
    selected_ids = {spec.id for spec in selected}
    if selected_ids and "data_quality_report" not in selected_ids:
        write_quality_report(data, out_dir)
    write_artifacts_manifest(registry=registry, out_dir=out_dir)
    write_dashboard_html(out_dir=out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build verifier-routed OPD paper non-main-result artifacts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/pcd_paper"))
    parser.add_argument("--make", action="append", default=None)
    args = parser.parse_args(argv)
    requested: set[str] = set()
    for item in args.make or ["all"]:
        requested.update(part.strip() for part in str(item).split(",") if part.strip())
    return run(requested, args.manifest, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
