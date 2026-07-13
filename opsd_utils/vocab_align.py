"""Cross-model OPD vocab alignment checks (student vs teacher)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug

# Log slice/renorm diagnostics once per process (first mismatched-vocab JSD).
_renorm_debug_logged = False


def align_cross_model_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    log_renorm_check: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align vocab dimension for cross-model OPD (e.g. 0.5B vs 7B LLaVA-OneVision).

    Renormalization: generalized_jsd_loss applies log_softmax on the *aligned*
    slice, so each distribution is re-normalized over shared vocab only.
  """
    global _renorm_debug_logged

    vs = student_logits.size(-1)
    vt = teacher_logits.size(-1)
    if vs == vt:
        return student_logits, teacher_logits

    shared = min(vs, vt)
    student_aligned = student_logits[..., :shared]
    teacher_aligned = teacher_logits[..., :shared]

    if log_renorm_check and not _renorm_debug_logged:
        _log_slice_renorm_diagnostics(
            student_logits,
            teacher_logits,
            student_aligned,
            teacher_aligned,
            shared_vocab=shared,
        )
        _renorm_debug_logged = True

    opsd_debug.log(
        "vocab_align",
        "align_cross_model_logits vocab slice",
        student_vocab=vs,
        teacher_vocab=vt,
        shared_vocab=shared,
        renorm_note="generalized_jsd_loss applies log_softmax on aligned slice",
    )
    return student_aligned, teacher_aligned


def _log_slice_renorm_diagnostics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_aligned: torch.Tensor,
    teacher_aligned: torch.Tensor,
    *,
    shared_vocab: int,
) -> None:
    """One-shot debug: truncated mass + post-log_softmax sums (detail1)."""
    with torch.no_grad():
        # Use first valid completion position for a cheap probe.
        t_row = teacher_logits[0, 0].float()
        t_slice = teacher_aligned[0, 0].float()
        full_probs = F.softmax(t_row, dim=-1)
        tail_mass = float(full_probs[shared_vocab:].sum().item()) if t_row.numel() > shared_vocab else 0.0
        slice_probs_from_full = full_probs[:shared_vocab]
        slice_sum_before_renorm = float(slice_probs_from_full.sum().item())
        slice_probs_renorm = F.softmax(t_slice, dim=-1)
        slice_sum_after_renorm = float(slice_probs_renorm.sum().item())
        log_probs_after = F.log_softmax(t_slice, dim=-1)
        exp_log_sum = float(torch.exp(log_probs_after).sum().item())

    msg = (
        f"[OPSD-VOCAB][detail1-renorm] shared_vocab={shared_vocab} "
        f"teacher_tail_prob_mass_dropped={tail_mass:.6f} "
        f"teacher_slice_prob_sum_before_renorm={slice_sum_before_renorm:.6f} "
        f"teacher_slice_prob_sum_after_softmax={slice_sum_after_renorm:.6f} "
        f"teacher_exp_log_softmax_sum={exp_log_sum:.6f} "
        f"| JSD uses log_softmax on aligned logits → distributions re-normalized"
    )
    print(msg, flush=True)
    opsd_debug.log(
        "vocab_align",
        "slice renorm diagnostics",
        shared_vocab=shared_vocab,
        teacher_tail_prob_mass_dropped=tail_mass,
        teacher_slice_prob_sum_before_renorm=slice_sum_before_renorm,
        teacher_slice_prob_sum_after_softmax=slice_sum_after_renorm,
        teacher_exp_log_softmax_sum=exp_log_sum,
        jsd_renormalizes=True,
    )


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return repr(tokenizer.convert_ids_to_tokens(token_id))


def verify_shared_tokenizer_alignment(
    student_tokenizer,
    teacher_tokenizer,
    *,
    shared_vocab: Optional[int] = None,
    full_scan: bool = False,
    sample_stride: int = 500,
    max_mismatches_to_report: int = 5,
) -> dict[str, Any]:
    """
    Verify token id -> string mapping matches for shared vocab (detail2).

    Returns a report dict; prints summary on main process when called from main.py.
    """
    st_size = len(student_tokenizer)
    tt_size = len(teacher_tokenizer)
    shared = shared_vocab if shared_vocab is not None else min(st_size, tt_size)
    shared = min(shared, st_size, tt_size)

    indices_to_check: list[int] = list(range(shared)) if full_scan else list(range(0, shared, sample_stride))
    if (shared - 1) not in indices_to_check:
        indices_to_check.append(shared - 1)

    mismatches: list[dict[str, Any]] = []
    for i in indices_to_check:
        s_dec = _decode_token(student_tokenizer, i)
        t_dec = _decode_token(teacher_tokenizer, i)
        if s_dec != t_dec:
            mismatches.append({"id": i, "student": s_dec, "teacher": t_dec})
            if full_scan and len(mismatches) >= max_mismatches_to_report:
                break

    report = {
        "student_vocab_size": st_size,
        "teacher_vocab_size": tt_size,
        "shared_vocab_checked": shared,
        "full_scan": full_scan,
        "sample_stride": sample_stride,
        "indices_checked": len(indices_to_check),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:max_mismatches_to_report],
        "aligned": len(mismatches) == 0,
    }
    return report


def print_vocab_align_report(report: dict[str, Any]) -> None:
    """Human-readable startup report for detail2."""
    status = "PASS" if report["aligned"] else "FAIL"
    print(
        f"[OPSD-VOCAB][detail2-tokenizer] {status} "
        f"student_vocab={report['student_vocab_size']} "
        f"teacher_vocab={report['teacher_vocab_size']} "
        f"shared={report['shared_vocab_checked']} "
        f"checked={report['indices_checked']} indices "
        f"full_scan={report['full_scan']} "
        f"mismatches={report['mismatch_count']}",
        flush=True,
    )
    for m in report.get("mismatches", []):
        print(
            f"  mismatch id={m['id']!r}: student={m['student']!r} teacher={m['teacher']!r}",
            flush=True,
        )
    if report["aligned"]:
        print(
            "[OPSD-VOCAB][detail2-tokenizer] sampled ids decode identically; "
            "set DYME_VOCAB_ALIGN_FULL=1 for exhaustive scan",
            flush=True,
        )
    else:
        print(
            "[OPSD-VOCAB][detail2-tokenizer] WARNING: token id mapping differs — "
            "vocab slice may be misaligned; consider explicit id mapping",
            flush=True,
        )
    opsd_debug.log("vocab_align", "tokenizer alignment report", **report)


def reset_vocab_align_debug() -> None:
    """Reset one-shot renorm logging (for tests)."""
    global _renorm_debug_logged
    _renorm_debug_logged = False
