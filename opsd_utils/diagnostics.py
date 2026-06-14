"""Periodic full-detail diagnostics for weak reward / gradient signals."""
from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug
from opsd_utils.vocab_align import align_cross_model_logits

PAREN_TOKEN_ID = 340

# Reuse logits from the OPSD loss path instead of running extra forwards.
_OPSD_JSD_DETAIL_CAPTURE: dict[str, Any] = {
    "active": False,
    "global_step": None,
    "target_indices": set(),
    "entries": [],
    "skipped_memory": False,
    "skip_reason": "",
    "max_samples": 2,
}


def _detail_min_free_gib() -> float:
    raw = os.environ.get("DYME_OPSD_DETAIL_MIN_FREE_GB", "").strip()
    if not raw:
        return 4.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 4.0


def cuda_free_gib(device: Optional[torch.device | int] = None) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        if device is None:
            free_bytes, _ = torch.cuda.mem_get_info()
        else:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            with torch.cuda.device(dev):
                free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / (1024**3)
    except Exception:
        return None


def check_detail_cuda_memory(
    min_free_gib: Optional[float] = None,
    device: Optional[torch.device | int] = None,
) -> tuple[bool, str, Optional[float]]:
    """Return (ok, reason, free_gib). Skips heavy detail work when GPU headroom is low."""
    threshold = _detail_min_free_gib() if min_free_gib is None else max(0.0, float(min_free_gib))
    if not torch.cuda.is_available():
        return True, "", None
    free_gib = cuda_free_gib(device)
    if free_gib is None:
        return True, "", None
    if free_gib < threshold:
        return (
            False,
            f"cuda_free_gib={free_gib:.2f} < min_free_gib={threshold:.2f}",
            free_gib,
        )
    return True, "", free_gib


def begin_opsd_jsd_detail_capture(
    global_step: int,
    opsd_indices: list[int],
    max_samples: int = 2,
) -> None:
    """Prepare to record JSD stats during OPSD loss (no extra model forwards)."""
    _OPSD_JSD_DETAIL_CAPTURE.update(
        active=False,
        global_step=global_step,
        target_indices=set(),
        entries=[],
        skipped_memory=False,
        skip_reason="",
        max_samples=max(1, int(max_samples)),
    )
    if not opsd_debug.should_log_detail(global_step) or not opsd_indices:
        return

    ok, reason, free_gib = check_detail_cuda_memory()
    if not ok:
        _OPSD_JSD_DETAIL_CAPTURE["skipped_memory"] = True
        _OPSD_JSD_DETAIL_CAPTURE["skip_reason"] = reason
        opsd_debug.log_detail(
            "opsd_jsd",
            "skip JSD detail capture (CUDA memory guard)",
            global_step=global_step,
            reason=reason,
            cuda_free_gib=free_gib,
            min_free_gib=_detail_min_free_gib(),
        )
        return

    _OPSD_JSD_DETAIL_CAPTURE["active"] = True
    _OPSD_JSD_DETAIL_CAPTURE["target_indices"] = set(opsd_indices[: _OPSD_JSD_DETAIL_CAPTURE["max_samples"]])


def maybe_capture_opsd_jsd_detail(
    *,
    global_idx: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    beta: float,
    tokenizer: Any = None,
    student_prompt_len: Optional[int] = None,
    teacher_prompt_len: Optional[int] = None,
) -> None:
    """Record token-level JSD stats from logits already computed in the loss path."""
    capture = _OPSD_JSD_DETAIL_CAPTURE
    if not capture["active"] or global_idx not in capture["target_indices"]:
        return

    try:
        with torch.no_grad():
            s_logits, t_logits = align_cross_model_logits(
                student_logits.detach(),
                teacher_logits.detach(),
            )
            stats = jsd_token_stats(s_logits, t_logits, completion_mask.float(), beta=beta)
        stats["sample_index"] = global_idx
        if student_prompt_len is not None:
            stats["student_prompt_len"] = int(student_prompt_len)
        if teacher_prompt_len is not None:
            stats["teacher_prompt_len"] = int(teacher_prompt_len)
        if tokenizer is not None:
            decoded = tokenizer.decode(
                completion_ids[0][completion_mask[0].bool()],
                skip_special_tokens=True,
            )
            stats["completion_text"] = _preview_text(decoded)
        capture["entries"].append(stats)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            opsd_debug.log_detail(
                "opsd_jsd",
                f"skip JSD detail for sample[{global_idx}] (OOM during stats)",
                global_step=capture.get("global_step"),
                error=repr(exc),
            )
            return
        raise


def _tensor_stats(t: torch.Tensor, name: str) -> dict[str, Any]:
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return {name: "empty"}
    with torch.no_grad():
        flat = t.detach().float().reshape(-1)
        return {
            f"{name}/shape": tuple(t.shape),
            f"{name}/mean": float(flat.mean().item()),
            f"{name}/std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            f"{name}/min": float(flat.min().item()),
            f"{name}/max": float(flat.max().item()),
            f"{name}/abs_mean": float(flat.abs().mean().item()),
        }


def _preview_text(text: str, max_len: int = 320) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text or "<EMPTY>"


def _generation_config_summary(generation_config: Any) -> dict[str, Any]:
    if generation_config is None:
        return {}
    keys = (
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "eos_token_id",
        "pad_token_id",
        "bos_token_id",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(generation_config, k):
            v = getattr(generation_config, k)
            out[k] = v.tolist() if isinstance(v, torch.Tensor) else v
    return out


def _slice_generate_inputs(batch: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    """Take one row from batched generate/forward tensors (VLM-safe)."""
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) == batch_size:
            sliced[key] = value[index : index + 1]
        else:
            sliced[key] = value
    return sliced


def _last_valid_prompt_positions(prompt_mask: torch.Tensor) -> torch.Tensor:
    """Return index of last valid (non-pad) prompt token per row; supports left padding."""
    with torch.no_grad():
        seq_len = prompt_mask.size(1)
        positions = torch.arange(seq_len, device=prompt_mask.device).expand_as(prompt_mask)
        masked = torch.where(prompt_mask.bool(), positions, torch.full_like(positions, -1))
        return masked.max(dim=1).values


def _max_same_token_run(token_ids: list[int]) -> tuple[int, Optional[int]]:
    """Longest run of identical consecutive token ids."""
    if not token_ids:
        return 0, None
    best_run = 1
    best_tok = token_ids[0]
    run = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            run += 1
            if run > best_run:
                best_run = run
                best_tok = token_ids[i]
        else:
            run = 1
    return best_run, best_tok


def _detect_single_token_repeat(token_ids: list[int], min_run: int = 8) -> bool:
    return _max_same_token_run(token_ids)[0] >= min_run


def _detect_char_repeat(text: str, min_run: int = 6) -> bool:
    """Detect consecutive repeated characters (CJK or ASCII), e.g. 其其其."""
    if not text or len(text) < min_run:
        return False
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1] and not text[i].isspace():
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def _count_char_repeat_samples(completions: Optional[list[str]], min_run: int = 6) -> int:
    if not completions:
        return 0
    return sum(1 for c in completions if _detect_char_repeat(c or "", min_run=min_run))


def _detect_repeat_loop(token_ids: list[int], min_repeats: int = 4, ngram: int = 3) -> bool:
    if len(token_ids) < ngram * min_repeats:
        return False
    last_start = len(token_ids) - ngram * min_repeats + 1
    for start in range(max(0, last_start)):
        gram = token_ids[start : start + ngram]
        repeats = 1
        pos = start + ngram
        while pos + ngram <= len(token_ids) and token_ids[pos : pos + ngram] == gram:
            repeats += 1
            pos += ngram
        if repeats >= min_repeats:
            return True
    return False


def _detect_degeneration(
    token_ids: list[int],
    text: str,
    *,
    answer_flag: str = "Answer:",
    min_single_token_run: int = 8,
    require_answer_flag: bool = True,
) -> tuple[bool, list[str]]:
    """Heuristics for repetitive / format-broken completions."""
    reasons: list[str] = []
    if _detect_single_token_repeat(token_ids, min_run=min_single_token_run):
        run_len, tok = _max_same_token_run(token_ids)
        reasons.append(f"SINGLE_TOKEN_REPEAT(run={run_len},tok={tok})")
    if _detect_repeat_loop(token_ids):
        reasons.append("NGRAM_REPEAT")
    if len(token_ids) >= 16:
        unique_ratio = len(set(token_ids)) / len(token_ids)
        if unique_ratio < 0.12:
            reasons.append(f"LOW_UNIQUE_RATIO({unique_ratio:.3f})")
    if require_answer_flag:
        answer_count = len(re.findall(f"(?i){re.escape(answer_flag)}", text or ""))
        if answer_count != 1:
            reasons.append(f"ANSWER_FLAG_COUNT({answer_count})")
    if _detect_char_repeat(text or ""):
        reasons.append("CHAR_REPEAT")
    return bool(reasons), reasons


def _count_answer_flag(text: str, answer_flag: str = "Answer:") -> int:
    return len(re.findall(f"(?i){re.escape(answer_flag)}", text or ""))


def is_degenerate_completion(
    token_ids: list[int],
    text: str,
    *,
    answer_flag: str = "Answer:",
    min_single_token_run: int = 8,
    require_answer_flag: bool = True,
) -> bool:
    """Return True when completion looks like a repetition / format-broken sample."""
    is_deg, _ = _detect_degeneration(
        token_ids,
        text,
        answer_flag=answer_flag,
        min_single_token_run=min_single_token_run,
        require_answer_flag=require_answer_flag,
    )
    return is_deg


def _count_paren_then_eos(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    eos_id: Optional[int],
) -> int:
    if eos_id is None or completion_ids.size(0) == 0:
        return 0
    count = 0
    with torch.no_grad():
        lengths = completion_mask.sum(dim=1)
        for i in range(completion_ids.size(0)):
            eff = int(lengths[i].item())
            if eff <= 0:
                continue
            first = int(completion_ids[i, 0].item())
            if first != PAREN_TOKEN_ID:
                continue
            if eff <= 2:
                count += 1
            elif eff >= 2 and int(completion_ids[i, 1].item()) == eos_id:
                count += 1
    return count


def summarize_generate_probe_stats(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    is_eos: torch.Tensor,
    eos_id: Optional[int],
    completions: Optional[list[str]] = None,
    answer_flag: str = "Answer:",
    max_completion_length: Optional[int] = None,
) -> dict[str, Any]:
    with torch.no_grad():
        lengths = completion_mask.sum(dim=1).float()
        has_eos = is_eos.any(dim=1)
    paren_then_eos = _count_paren_then_eos(completion_ids, completion_mask, eos_id)
    repeat_loop = 0
    degenerate_count = 0
    max_run_lengths: list[float] = []
    unique_ratios: list[float] = []
    answer_flag_ok = 0
    clipped_count = 0
    degenerate_format_count = 0
    degenerate_repeat_count = 0
    format_without_thinking = 0
    for i in range(completion_ids.size(0)):
        eff = int(lengths[i].item())
        if eff <= 0:
            continue
        ids = completion_ids[i, :eff].tolist()
        text = completions[i] if completions and i < len(completions) else ""
        if _detect_repeat_loop(ids) or _detect_single_token_repeat(ids):
            repeat_loop += 1
        run_len, _ = _max_same_token_run(ids)
        max_run_lengths.append(float(run_len))
        unique_ratios.append(len(set(ids)) / max(len(ids), 1))
        if _count_answer_flag(text, answer_flag) == 1:
            answer_flag_ok += 1
            thinking = (text or "").lower().split(answer_flag.lower())[0]
            if len(thinking.strip()) < 8:
                format_without_thinking += 1
        else:
            degenerate_format_count += 1
        if max_completion_length is not None and eff >= max_completion_length - 1:
            clipped_count += 1
        is_deg, reasons = _detect_degeneration(ids, text, answer_flag=answer_flag)
        if is_deg:
            degenerate_count += 1
        non_flag = [r for r in reasons if not r.startswith("ANSWER_FLAG")]
        if non_flag:
            degenerate_repeat_count += 1
    char_repeat_count = _count_char_repeat_samples(completions)
    n = max(completion_ids.size(0), 1)
    return {
        "effective_tokens_mean": float(lengths.mean().item()),
        "char_repeat_count": char_repeat_count,
        "one_token_count": int((lengths == 1).sum().item()),
        "paren_then_eos_count": paren_then_eos,
        "repeat_loop_count": repeat_loop,
        "eos_terminated_rate": float(has_eos.float().mean().item()),
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_count / n,
        "degenerate_rate_format": degenerate_format_count / n,
        "degenerate_rate_repeat": degenerate_repeat_count / n,
        "format_without_thinking_rate": format_without_thinking / n,
        "max_token_run_mean": float(sum(max_run_lengths) / len(max_run_lengths)) if max_run_lengths else 0.0,
        "unique_token_ratio_mean": float(sum(unique_ratios) / len(unique_ratios)) if unique_ratios else 0.0,
        "answer_flag_exactly_once_rate": answer_flag_ok / n,
        "clipped_count": clipped_count,
        "clipped_rate": clipped_count / n,
    }


def log_generate_context(
    *,
    global_step: int,
    trainer_step: Optional[int],
    generate_call_index: int,
    model: Any,
    model_wrapped: Any,
    gradient_checkpointing: bool,
    generation_config: Any,
    is_fsdp_enabled: bool,
    generate_runs_under_no_grad: bool,
) -> None:
    if not opsd_debug.should_log_gendbg() or not opsd_debug.probe_log_model_context():
        return

    opsd_debug.set_detail_step(global_step)
    dropout_in_train = sum(
        1 for m in model.modules() if isinstance(m, nn.Dropout) and m.training
    )
    opsd_debug.log_gendbg(
        "context",
        "generate model context",
        generate_call_index=generate_call_index,
        trainer_step=trainer_step,
        global_step=global_step,
        model_training=bool(getattr(model, "training", None)),
        model_wrapped_training=bool(getattr(model_wrapped, "training", None)),
        gradient_checkpointing=gradient_checkpointing,
        generation_use_cache=getattr(generation_config, "use_cache", None),
        is_fsdp_enabled=is_fsdp_enabled,
        generate_runs_under_no_grad=generate_runs_under_no_grad,
        dropout_modules_in_train=dropout_in_train,
        generation_config=_generation_config_summary(generation_config),
    )


def log_prompt_tail_probe(
    *,
    global_step: int,
    trainer_step: Optional[int],
    generate_call_index: int,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    tokenizer: Any,
    sample_count: int = 4,
    tail_tokens: Optional[int] = None,
) -> None:
    if not opsd_debug.should_log_gendbg():
        return

    opsd_debug.set_detail_step(global_step)
    n_tail = tail_tokens if tail_tokens is not None else opsd_debug.probe_prompt_tail_tokens()
    last_pos = _last_valid_prompt_positions(prompt_mask)
    n = min(sample_count, prompt_ids.size(0))

    for i in range(n):
        end = int(last_pos[i].item()) + 1
        start = max(0, end - n_tail)
        tail_ids = prompt_ids[i, start:end].tolist()
        eff_len = int(prompt_mask[i].sum().item())
        tail_decode = tokenizer.decode(tail_ids, skip_special_tokens=False)
        opsd_debug.log_gendbg(
            "prompt_tail",
            f"sample[{i}]",
            generate_call_index=generate_call_index,
            trainer_step=trainer_step,
            prompt_effective_len=eff_len,
            last_valid_idx=end - 1,
            prompt_tail_token_ids=tail_ids,
            prompt_tail_decode=_preview_text(tail_decode, 400),
        )


def summarize_first_token_logits_stats(
    p_greedy_values: list[float],
    p_eos_values: list[float],
    entropy_values: list[float],
    p_answer_values: Optional[list[float]] = None,
) -> dict[str, float]:
    """Aggregate first-token logit probe scalars across samples."""
    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    out = {
        "p_greedy_first": _mean(p_greedy_values),
        "p_eos_first": _mean(p_eos_values),
        "entropy_first": _mean(entropy_values),
    }
    if p_answer_values:
        out["p_answer_first"] = _mean(p_answer_values)
    return out


def answer_first_token_id(tokenizer: Any) -> Optional[int]:
    for piece in ("Answer", "Answer:"):
        ids = tokenizer.encode(piece, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


def log_first_token_logits_probe(
    *,
    global_step: int,
    trainer_step: Optional[int],
    generate_call_index: int,
    unwrapped_model: Any,
    prompt_inputs_generate: dict[str, Any],
    prompt_mask: torch.Tensor,
    tokenizer: Any,
    sample_count: int = 4,
) -> dict[str, Any]:
    """Forward once before generate; return greedy ids and aggregated logit stats."""
    greedy_by_sample: dict[int, int] = {}
    p_greedy_vals: list[float] = []
    p_eos_vals: list[float] = []
    entropy_vals: list[float] = []
    p_answer_vals: list[float] = []
    if not opsd_debug.should_log_gendbg() or not opsd_debug.probe_first_token_logits():
        return {
            "greedy_by_sample": greedy_by_sample,
            **summarize_first_token_logits_stats([], [], [], []),
        }

    opsd_debug.set_detail_step(global_step)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    answer_tid = answer_first_token_id(tokenizer)
    forward_inputs = {k: v for k, v in prompt_inputs_generate.items() if k != "labels"}
    batch_size = prompt_mask.size(0)
    n = min(sample_count, batch_size)
    last_pos = _last_valid_prompt_positions(prompt_mask)

    for i in range(n):
        try:
            sample_inputs = _slice_generate_inputs(forward_inputs, i, batch_size)
            with torch.no_grad():
                outputs = unwrapped_model(**sample_inputs, use_cache=False)
                logits = outputs.logits
                pos = int(last_pos[i].item())
                next_logits = logits[0, pos, :].float()
                probs = F.softmax(next_logits, dim=-1)
                greedy_id = int(next_logits.argmax().item())
                greedy_by_sample[i] = greedy_id
                entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
                fields: dict[str, Any] = {
                    "generate_call_index": generate_call_index,
                    "trainer_step": trainer_step,
                    "last_prompt_idx": pos,
                    "greedy_token_id": greedy_id,
                    "p_greedy": float(probs[greedy_id].item()),
                    "entropy": entropy,
                    "probe_mode": "per_sample_forward",
                }
                if eos_id is not None:
                    fields["p_eos"] = float(probs[eos_id].item())
                if answer_tid is not None and answer_tid < probs.size(0):
                    fields["p_answer_first"] = float(probs[answer_tid].item())
                    p_answer_vals.append(fields["p_answer_first"])
                if PAREN_TOKEN_ID < probs.size(0):
                    fields["p_token_340"] = float(probs[PAREN_TOKEN_ID].item())
                topk = torch.topk(probs, k=min(5, probs.size(0)))
                fields["top5"] = [
                    (int(tid), float(p)) for tid, p in zip(topk.indices.tolist(), topk.values.tolist())
                ]
                p_greedy_vals.append(fields["p_greedy"])
                entropy_vals.append(entropy)
                if eos_id is not None:
                    p_eos_vals.append(fields["p_eos"])
                opsd_debug.log_gendbg("first_token_logits", f"sample[{i}]", **fields)
        except Exception as exc:
            opsd_debug.log_gendbg(
                "first_token_logits",
                f"FAILED forward for sample[{i}]",
                generate_call_index=generate_call_index,
                error=repr(exc),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logits_stats = summarize_first_token_logits_stats(
        p_greedy_vals, p_eos_vals, entropy_vals, p_answer_vals
    )
    return {"greedy_by_sample": greedy_by_sample, **logits_stats}


def log_first_token_logits_match(
    *,
    generate_call_index: int,
    completion_ids: torch.Tensor,
    greedy_by_sample: dict[int, int],
    sample_count: int = 4,
) -> None:
    """Compare pre-generate greedy next token vs actual first generated token."""
    if not opsd_debug.should_log_gendbg() or not opsd_debug.probe_first_token_logits():
        return

    n = min(sample_count, completion_ids.size(0))
    for i in range(n):
        if i not in greedy_by_sample or completion_ids.size(1) == 0:
            continue
        greedy_id = greedy_by_sample[i]
        actual_id = int(completion_ids[i, 0].item())
        opsd_debug.log_gendbg(
            "first_token_match",
            f"sample[{i}]",
            generate_call_index=generate_call_index,
            greedy_token_id=greedy_id,
            actual_first_token=actual_id,
            greedy_matches_actual=(greedy_id == actual_id),
        )


def log_generate_delta(
    *,
    generate_call_index: int,
    current_stats: dict[str, Any],
    previous_stats: Optional[dict[str, Any]],
) -> None:
    if not opsd_debug.should_log_gendbg():
        return

    fields: dict[str, Any] = {
        "generate_call_index": generate_call_index,
        "current": current_stats,
    }
    if previous_stats is not None:
        prev_idx = previous_stats.get("generate_call_index", generate_call_index - 1)
        fields["prev_generate_call_index"] = prev_idx
        for key in (
            "effective_tokens_mean",
            "one_token_count",
            "paren_then_eos_count",
            "repeat_loop_count",
            "eos_terminated_rate",
            "degenerate_count",
            "degenerate_rate",
            "clipped_rate",
            "answer_flag_exactly_once_rate",
        ):
            cur = current_stats.get(key)
            prev = previous_stats.get(key)
            if cur is not None and prev is not None:
                if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
                    fields[f"delta_{key}"] = cur - prev
    opsd_debug.log_gendbg("delta", "generate stats vs previous regenerate", **fields)


def log_generate_probe(
    *,
    global_step: int,
    trainer_step: Optional[int],
    prompt_length: int,
    prompt_completion_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    is_eos: torch.Tensor,
    eos_idx: torch.Tensor,
    completions: list[str],
    tokenizer: Any,
    generation_config: Any,
    max_completion_length: int,
    num_generations: int,
    sample_count: int = 4,
    generate_call_index: Optional[int] = None,
    answer_flag: str = "Answer:",
) -> dict[str, Any]:
    """Emit [OPSD-PROBE] on every (re)generate — catches 1-token / empty completions early."""
    stats = summarize_generate_probe_stats(
        completion_ids,
        completion_mask,
        is_eos,
        getattr(tokenizer, "eos_token_id", None),
        completions=completions,
        answer_flag=answer_flag,
        max_completion_length=max_completion_length,
    )
    if generate_call_index is not None:
        stats["generate_call_index"] = generate_call_index

    if not opsd_debug.should_log_probe():
        return stats

    opsd_debug.set_detail_step(global_step)
    tok = tokenizer
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    bos_id = getattr(tok, "bos_token_id", None)

    with torch.no_grad():
        lengths = completion_mask.sum(dim=1).float()
        has_eos = is_eos.any(dim=1)
        raw_gen_len = completion_ids.size(1)

    opsd_debug.log_probe(
        "generate",
        "raw generate summary",
        trainer_step=trainer_step,
        global_step=global_step,
        generate_call_index=generate_call_index,
        prompt_length=prompt_length,
        prompt_completion_shape=tuple(prompt_completion_ids.shape),
        completion_ids_shape=tuple(completion_ids.shape),
        raw_gen_tokens=raw_gen_len,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        batch_size=completion_ids.size(0),
        effective_tokens_mean=stats["effective_tokens_mean"],
        effective_tokens_min=float(lengths.min().item()),
        effective_tokens_max=float(lengths.max().item()),
        zero_length_count=int((lengths == 0).sum().item()),
        one_token_count=stats["one_token_count"],
        paren_then_eos_count=stats["paren_then_eos_count"],
        repeat_loop_count=stats["repeat_loop_count"],
        eos_terminated_rate=stats["eos_terminated_rate"],
        degenerate_count=stats["degenerate_count"],
        degenerate_rate=stats["degenerate_rate"],
        max_token_run_mean=stats["max_token_run_mean"],
        unique_token_ratio_mean=stats["unique_token_ratio_mean"],
        answer_flag_exactly_once_rate=stats["answer_flag_exactly_once_rate"],
        clipped_count=stats["clipped_count"],
        clipped_rate=stats["clipped_rate"],
        tokenizer_eos_id=eos_id,
        tokenizer_pad_id=pad_id,
        tokenizer_bos_id=bos_id,
        generation_config=_generation_config_summary(generation_config),
    )

    suspicious: list[str] = []
    n = min(sample_count, len(completions))
    for i in range(n):
        eff = int(lengths[i].item())
        eidx = int(eos_idx[i].item())
        ids_head = completion_ids[i, : min(16, completion_ids.size(1))].tolist()
        ids_all = completion_ids[i].tolist()
        decode_skip = completions[i]
        decode_keep = tok.decode(completion_ids[i], skip_special_tokens=False)
        first_tok = int(completion_ids[i, 0].item()) if completion_ids.size(1) > 0 else None
        first_is_eos = first_tok is not None and first_tok == eos_id
        flags: list[str] = []
        patterns: list[str] = []
        if eff <= 0:
            flags.append("ZERO_LEN")
        if eff == 1:
            flags.append("ONE_TOKEN")
        if not (decode_skip or "").strip():
            flags.append("EMPTY_DECODE")
        if first_is_eos:
            flags.append("FIRST_IS_EOS")
        if eff <= 2 and first_tok == PAREN_TOKEN_ID:
            patterns.append("PAREN_THEN_EOS")
        if eff > 0:
            ids_eff = completion_ids[i, :eff].tolist()
            if _detect_repeat_loop(ids_eff):
                patterns.append("REPEAT_LOOP")
            if _detect_single_token_repeat(ids_eff):
                run_len, repeat_tok_id = _max_same_token_run(ids_eff)
                patterns.append(f"SINGLE_TOKEN_REPEAT({run_len}x{repeat_tok_id})")
            is_deg, deg_reasons = _detect_degeneration(ids_eff, decode_skip, answer_flag=answer_flag)
            if is_deg:
                patterns.extend(deg_reasons)
        if flags:
            suspicious.append(f"sample[{i}]:{','.join(flags)}")

        opsd_debug.log_probe(
            "generate",
            f"sample[{i}]",
            group=i // num_generations if num_generations else 0,
            effective_tokens=eff,
            eos_idx=eidx,
            has_eos=bool(has_eos[i].item()),
            first_token_id=first_tok,
            first_is_eos=first_is_eos,
            token_ids_head=ids_head,
            token_ids_all=ids_all if len(ids_all) <= 32 else ids_head + ["..."],
            decode_skip_special=_preview_text(decode_skip, 400),
            decode_keep_special=_preview_text(decode_keep, 400),
            flags=flags or None,
            patterns=patterns or None,
        )

    if suspicious:
        opsd_debug.log_probe(
            "generate",
            "ALERT suspicious completions",
            count=len(suspicious),
            samples=suspicious,
            hint="check eos_token_id, max_new_tokens, model.generate output, or training-time unwrap/FSDP",
        )

    if stats.get("degenerate_rate", 0) >= 0.25:
        opsd_debug.log_probe(
            "generate",
            "ALERT repetition / format degeneration",
            degenerate_count=stats["degenerate_count"],
            degenerate_rate=stats["degenerate_rate"],
            clipped_rate=stats["clipped_rate"],
            answer_flag_exactly_once_rate=stats["answer_flag_exactly_once_rate"],
            max_token_run_mean=stats["max_token_run_mean"],
            hint="typical RL collapse: raise repetition_penalty, lower temperature, or shorten max_completion_length",
        )

    return stats


def log_cross_rank_generate_summary(
    *,
    accelerator: Any,
    one_token_count: int,
    effective_tokens_mean: float,
    generate_call_index: int,
) -> None:
    """Gather per-rank generate stats; all ranks must call, log on rank 0 only."""
    if not opsd_debug.probe_on_generate():
        return

    from accelerate.utils import gather as accel_gather

    device = accelerator.device
    world_size = accelerator.num_processes
    local = torch.tensor(
        [float(one_token_count), effective_tokens_mean],
        dtype=torch.float32,
        device=device,
    )
    # gather_for_metrics on 1D [2] concatenates ranks into [world_size*2]; use gather + view instead.
    gathered = accel_gather(local.unsqueeze(0))
    if not opsd_debug.should_log_gendbg():
        return

    if gathered.dim() == 1:
        gathered = gathered.view(world_size, -1)
    elif gathered.size(0) != world_size and gathered.numel() == world_size * 2:
        gathered = gathered.reshape(world_size, 2)
    one_tokens = gathered[:, 0].tolist()
    means = gathered[:, 1].tolist()
    opsd_debug.log_gendbg(
        "cross_rank",
        "generate summary across ranks",
        generate_call_index=generate_call_index,
        world_size=world_size,
        one_token_count_per_rank=one_tokens,
        effective_tokens_mean_per_rank=means,
        one_token_count_total=int(sum(one_tokens)),
    )


def log_routed_completion_probe(
    *,
    global_step: int,
    trainer_step: Optional[int],
    raw_completion_shape: tuple[int, ...],
    final_completion_ids: torch.Tensor,
    final_completion_mask: torch.Tensor,
    opsd_mask_list: list[bool],
    sample_count: int = 4,
    tokenizer: Any,
    sft_replaced_list: Optional[list[bool]] = None,
    raw_completion_ids: Optional[torch.Tensor] = None,
) -> None:
    """Compare raw vs post-routing padded completion shapes (detect routing truncation)."""
    if not opsd_debug.should_log_probe():
        return

    with torch.no_grad():
        final_lengths = final_completion_mask.sum(dim=1).float()

    opsd_debug.log_probe(
        "route",
        "post-routing completion shapes",
        trainer_step=trainer_step,
        global_step=global_step,
        raw_completion_shape=raw_completion_shape,
        final_completion_shape=tuple(final_completion_ids.shape),
        final_mask_shape=tuple(final_completion_mask.shape),
        final_effective_tokens_mean=float(final_lengths.mean().item()),
        final_effective_tokens_min=float(final_lengths.min().item()),
        final_effective_tokens_max=float(final_lengths.max().item()),
        opsd_mask_true=sum(opsd_mask_list),
        opsd_mask_false=len(opsd_mask_list) - sum(opsd_mask_list),
        sft_replaced_count=sum(sft_replaced_list) if sft_replaced_list else None,
    )

    n = min(sample_count, final_completion_ids.size(0))
    for i in range(n):
        eff = int(final_lengths[i].item())
        head = final_completion_ids[i, : min(12, final_completion_ids.size(1))].tolist()
        text = tokenizer.decode(
            final_completion_ids[i][final_completion_mask[i].bool()],
            skip_special_tokens=True,
        )
        raw_head = None
        raw_matches_final = None
        if raw_completion_ids is not None and i < raw_completion_ids.size(0):
            raw_eff = min(int(raw_completion_ids.size(1)), eff)
            raw_head = raw_completion_ids[i, : min(12, raw_eff)].tolist()
            raw_matches_final = raw_head == head if raw_head and head else None
        opsd_debug.log_probe(
            "route",
            f"routed_sample[{i}]",
            opsd_mask=opsd_mask_list[i] if i < len(opsd_mask_list) else None,
            sft_replaced=sft_replaced_list[i] if sft_replaced_list and i < len(sft_replaced_list) else None,
            effective_tokens=eff,
            token_ids_head=head,
            raw_token_ids_head=raw_head,
            raw_matches_routed_head=raw_matches_final,
            decode=_preview_text(text, 300),
        )


def jsd_token_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Token-level JSD breakdown without building the graph."""
    with torch.no_grad():
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        per_token_jsd = jsd.sum(dim=-1)
        m = mask.float()
        valid = m.sum().clamp(min=1.0)
        per_token_jsd_masked = per_token_jsd * m

        # Top-1 agreement on completion tokens
        s_top = student_logits.argmax(dim=-1)
        t_top = teacher_logits.argmax(dim=-1)
        agree = ((s_top == t_top).float() * m).sum() / valid

        # Mean L2 distance of log-probs on gold completion tokens (if provided elsewhere)
        logprob_l2 = ((student_log_probs - teacher_log_probs) ** 2).sum(dim=-1)
        logprob_l2_masked = (logprob_l2 * m).sum() / valid

        # Entropy gap
        s_ent = -(student_probs * student_log_probs).sum(dim=-1)
        t_ent = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        ent_gap = ((s_ent - t_ent).abs() * m).sum() / valid

        stats: dict[str, Any] = {
            "jsd_per_token_mean": float((per_token_jsd_masked.sum() / valid).item()),
            "jsd_per_token_max": float(per_token_jsd[m.bool()].max().item()) if m.any() else 0.0,
            "jsd_per_token_min": float(per_token_jsd[m.bool()].min().item()) if m.any() else 0.0,
            "top1_agreement": float(agree.item()),
            "logprob_l2_mean": float(logprob_l2_masked.item()),
            "entropy_gap_mean": float(ent_gap.item()),
            "mask_valid_tokens": int(valid.item()),
        }

        if m.any():
            idx = int((per_token_jsd * m).argmax().item())
            pos = idx % per_token_jsd.size(-1)
            stats["max_jsd_token_pos"] = pos
            stats["max_jsd_value"] = float(per_token_jsd.reshape(-1)[idx].item())
            s_topk = torch.topk(student_probs[0, pos], k=min(top_k, student_probs.size(-1)))
            t_topk = torch.topk(teacher_probs[0, pos], k=min(top_k, teacher_probs.size(-1)))
            stats["student_topk_at_max_jsd"] = [
                (int(i), float(p)) for i, p in zip(s_topk.indices.tolist(), s_topk.values.tolist())
            ]
            stats["teacher_topk_at_max_jsd"] = [
                (int(i), float(p)) for i, p in zip(t_topk.indices.tolist(), t_topk.values.tolist())
            ]
        return stats


def log_generation_diagnostics(
    *,
    global_step: int,
    completions: list[str],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    is_eos: torch.Tensor,
    max_completion_length: int,
    num_generations: int,
    sample_count: int = 4,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "GENERATION & COMPLETION")

    with torch.no_grad():
        lengths = completion_mask.sum(dim=1).float()
        has_eos = is_eos.any(dim=1)
        eos_rate = has_eos.float().mean().item()
        clipped = (~has_eos).float().mean().item()
        all_pad_or_zero = (lengths == 0).float().mean().item()
        at_max_len = (lengths >= max_completion_length - 1).float().mean().item()

    fields: dict[str, Any] = {
        "batch_size": completion_ids.size(0),
        "num_generations": num_generations,
        "completion_max_len": completion_ids.size(1),
        "effective_tokens_mean": float(lengths.mean().item()),
        "effective_tokens_min": float(lengths.min().item()),
        "effective_tokens_max": float(lengths.max().item()),
        "eos_terminated_rate": eos_rate,
        "clipped_no_eos_rate": clipped,
        "zero_length_rate": all_pad_or_zero,
        "at_max_length_rate": at_max_len,
    }
    opsd_debug.log_detail("generation", "completion mask summary", **fields)

    n = min(sample_count, len(completions))
    for i in range(n):
        gen_group = i // num_generations if num_generations else 0
        opsd_debug.log_detail(
            "generation",
            f"sample[{i}]",
            group=gen_group,
            effective_tokens=int(lengths[i].item()),
            has_eos=bool(has_eos[i].item()),
            text=_preview_text(completions[i]),
            raw_token_head=completion_ids[i, :12].tolist(),
        )


def log_reward_diagnostics(
    *,
    global_step: int,
    format_rewards: torch.Tensor,
    acc_rewards: torch.Tensor,
    context_rewards: torch.Tensor,
    all_rewards: torch.Tensor,
    advantages: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
    answers: Optional[list[str]] = None,
    completions: Optional[list[str]] = None,
    sample_count: int = 4,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "REWARD & ADVANTAGE")

    w = reward_weights.detach().float()
    weighted = (
        format_rewards * w[0] + context_rewards * w[1] + acc_rewards * w[2]
    )

    fields: dict[str, Any] = {
        "reward_weights": w.tolist(),
        "format_sum": float(format_rewards.sum().item()),
        "acc_sum": float(acc_rewards.sum().item()),
        "context_sum": float(context_rewards.sum().item()),
        "total_sum": float(all_rewards.sum().item()),
        "format_zero_rate": float((format_rewards == 0).float().mean().item()),
        "acc_zero_rate": float((acc_rewards == 0).float().mean().item()),
        "context_zero_rate": float((context_rewards == 0).float().mean().item()),
        "weighted_mean": float(weighted.mean().item()),
        "weighted_std": float(weighted.std(unbiased=False).item()) if weighted.numel() > 1 else 0.0,
    }
    adv_flat = advantages.reshape(-1)
    fields.update(
        {
            "advantage_mean": float(adv_flat.mean().item()),
            "advantage_std": float(adv_flat.std(unbiased=False).item()) if adv_flat.numel() > 1 else 0.0,
            "advantage_zero_rate": float((adv_flat.abs() < 1e-8).float().mean().item()),
            "advantage_min": float(adv_flat.min().item()),
            "advantage_max": float(adv_flat.max().item()),
        }
    )
    opsd_debug.log_detail("reward", "aggregate reward stats", **fields)

    n = min(sample_count, format_rewards.numel())
    for i in range(n):
        g = i // num_generations if num_generations else 0
        extra: dict[str, Any] = {
            "group": g,
            "format": float(format_rewards.view(-1)[i].item()),
            "acc": float(acc_rewards.view(-1)[i].item()),
            "context": float(context_rewards.view(-1)[i].item()),
            "weighted": float(weighted.view(-1)[i].item()),
            "advantage": float(adv_flat.view(-1)[i].item()) if i < adv_flat.numel() else None,
        }
        if answers and i < len(answers):
            extra["gold_answer"] = _preview_text(str(answers[i // num_generations]), 80)
        if completions and i < len(completions):
            extra["completion"] = _preview_text(completions[i], 160)
        opsd_debug.log_detail("reward", f"per_sample[{i}]", **extra)


def log_routing_diagnostics(
    *,
    global_step: int,
    opsd_active: bool,
    opsd_mask_list: list[bool],
    has_correct: torch.Tensor,
    completion_modes: Optional[list[int]] = None,
    recoverable_flags: Optional[list[bool]] = None,
    completion_advantages: Optional[torch.Tensor] = None,
    completion_mask: Optional[torch.Tensor] = None,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "OPSD ROUTING & MASK")

    opsd_true = sum(opsd_mask_list)
    fields: dict[str, Any] = {
        "opsd_active": opsd_active,
        "opsd_mask_true": opsd_true,
        "opsd_mask_false": len(opsd_mask_list) - opsd_true,
        "opsd_mask_ratio": opsd_true / max(len(opsd_mask_list), 1),
        "has_correct": has_correct.tolist() if hasattr(has_correct, "tolist") else has_correct,
    }
    if recoverable_flags is not None:
        fields["recoverable_flags"] = recoverable_flags
    if completion_modes is not None:
        from opsd_utils.debug_log import MODE_NAMES

        counts = {MODE_NAMES.get(c, str(c)): completion_modes.count(c) for c in set(completion_modes)}
        fields["completion_mode_counts"] = counts

    if completion_advantages is not None and completion_mask is not None:
        with torch.no_grad():
            pos = ((completion_advantages > 0) & (completion_mask > 0)).float().sum(dim=1)
            neg = ((completion_advantages < 0) & (completion_mask > 0)).float().sum(dim=1)
            zero_adv = ((completion_advantages.abs() < 1e-8) & (completion_mask > 0)).float().sum(dim=1)
            fields["adv_pos_tokens_mean"] = float(pos.mean().item())
            fields["adv_neg_tokens_mean"] = float(neg.mean().item())
            fields["adv_zero_tokens_mean"] = float(zero_adv.mean().item())

    opsd_debug.log_detail("routing", "mode routing summary", **fields)


def log_loss_diagnostics(
    *,
    global_step: int,
    grpo_loss: torch.Tensor,
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    coef_1: torch.Tensor,
    per_token_loss: torch.Tensor,
    opsd_loss: Optional[torch.Tensor] = None,
    combined_loss: Optional[torch.Tensor] = None,
    opsd_mask: Optional[torch.Tensor] = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "LOSS & GRADIENT SIGNAL")

    with torch.no_grad():
        m = completion_mask.float()
        valid_per_sample = m.sum(dim=1).clamp(min=1.0)
        sample_grpo = (per_token_loss * m).sum(dim=1) / valid_per_sample

        fields: dict[str, Any] = {
            "grpo_loss_scalar": float(grpo_loss.detach().item()),
            "grpo_per_sample_mean": float(sample_grpo.mean().item()),
            "grpo_per_sample_max": float(sample_grpo.max().item()),
            "completion_mask_tokens_mean": float(valid_per_sample.mean().item()),
        }
        fields.update(_tensor_stats(advantages, "advantages"))
        fields.update(_tensor_stats(per_token_logps, "per_token_logps"))
        fields.update(_tensor_stats(old_per_token_logps, "old_per_token_logps"))
        fields.update(_tensor_stats((per_token_logps - old_per_token_logps) * m, "logps_delta_masked"))
        fields.update(_tensor_stats(coef_1 * m, "coef_1_masked"))

        low_clip = ((coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0) & (m > 0)).float().sum()
        high_clip = ((coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0) & (m > 0)).float().sum()
        fields["clipped_low_tokens"] = int(low_clip.item())
        fields["clipped_high_tokens"] = int(high_clip.item())
        fields["grpo_zero_loss_rate"] = float((sample_grpo.abs() < 1e-12).float().mean().item())

        if opsd_loss is not None:
            fields["opsd_loss_scalar"] = float(opsd_loss.detach().item())
        if combined_loss is not None:
            fields["combined_loss_scalar"] = float(combined_loss.detach().item())
        if opsd_mask is not None:
            fields["opsd_batch_count"] = int(opsd_mask.sum().item())

        # Weak-signal hints
        hints: list[str] = []
        if fields.get("advantages/abs_mean", 1.0) < 1e-6:
            hints.append("advantages≈0 → GRPO per-token loss≈0")
        if fields.get("grpo_zero_loss_rate", 0) > 0.9:
            hints.append("most samples have ~0 GRPO loss")
        if fields.get("completion_mask_tokens_mean", 0) < 2:
            hints.append("very few effective completion tokens in mask")
        if opsd_loss is not None and abs(fields.get("opsd_loss_scalar", 0)) < 1e-8:
            hints.append("OPSD JSD≈0 → student/teacher nearly identical on completion")
        fields["weak_signal_hints"] = hints or ["none"]

    opsd_debug.log_detail("loss", "GRPO / OPSD loss breakdown", **fields)


def summarize_loss_health(
    *,
    grpo_loss: torch.Tensor,
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    per_token_loss: torch.Tensor,
    opsd_loss: Optional[torch.Tensor] = None,
    combined_loss: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Lightweight loss/signal summary for every-step health monitoring."""
    with torch.no_grad():
        m = completion_mask.float()
        valid_per_sample = m.sum(dim=1).clamp(min=1.0)
        sample_grpo = (per_token_loss * m).sum(dim=1) / valid_per_sample
        adv_flat = advantages.detach().float().reshape(-1)
        fields: dict[str, float] = {
            "grpo_loss_scalar": float(grpo_loss.detach().item()),
            "grpo_zero_loss_rate": float((sample_grpo.abs() < 1e-12).float().mean().item()),
            "advantages_abs_mean": float(adv_flat.abs().mean().item()) if adv_flat.numel() else 0.0,
            "completion_mask_tokens_mean": float(valid_per_sample.mean().item()),
        }
        if opsd_loss is not None:
            fields["opsd_loss_scalar"] = float(opsd_loss.detach().item())
        if combined_loss is not None:
            fields["combined_loss_scalar"] = float(combined_loss.detach().item())
        logps_delta = (per_token_logps * m).sum() / m.sum().clamp(min=1.0)
        fields["logps_delta_mean"] = float(logps_delta.item()) if m.sum() > 0 else 0.0
    return fields


def summarize_batch_data_health(
    samples: list[dict[str, Any]],
    *,
    prompt_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Batch-level data I/O sanity for health monitoring."""
    n = max(len(samples), 1)
    vf_empty = 0
    prompt_lens: list[int] = []
    for sample in samples:
        vf = (
            sample.get("visual_fact_hint")
            or sample.get("visual_fact")
            or sample.get("visual_facts")
            or ""
        )
        if not str(vf).strip():
            vf_empty += 1
        if sample.get("prompt"):
            prompt_lens.append(len(str(sample["prompt"])))

    out: dict[str, Any] = {
        "visual_fact_empty_rate": vf_empty / n,
        "batch_size": len(samples),
        "prompt_len_mean": float(sum(prompt_lens) / len(prompt_lens)) if prompt_lens else 0.0,
    }

    if prompt_mask is not None:
        with torch.no_grad():
            lengths = prompt_mask.sum(dim=1).float()
            out["prompt_tokens_mean"] = float(lengths.mean().item())
            out["prompt_tokens_max"] = float(lengths.max().item())

    if pixel_values is not None and isinstance(pixel_values, torch.Tensor) and pixel_values.numel() > 0:
        with torch.no_grad():
            flat = pixel_values.detach().float().reshape(-1)
            out["pixel_mean"] = float(flat.mean().item())
            out["pixel_has_nan"] = bool(torch.isnan(flat).any().item())
            out["pixel_has_inf"] = bool(torch.isinf(flat).any().item())

    return out


def log_opsd_jsd_diagnostics(*, global_step: int) -> None:
    """Emit cached JSD stats recorded during OPSD loss (zero extra model forwards)."""
    if not opsd_debug.should_log_detail(global_step):
        return

    capture = _OPSD_JSD_DETAIL_CAPTURE
    if capture.get("global_step") != global_step:
        return

    opsd_debug.log_detail_banner(global_step, "OPSD JSD DECOMPOSITION")

    if capture.get("skipped_memory"):
        opsd_debug.log_detail(
            "opsd_jsd",
            "JSD detail skipped (CUDA memory guard)",
            global_step=global_step,
            reason=capture.get("skip_reason", ""),
            min_free_gib=_detail_min_free_gib(),
            cuda_free_gib=cuda_free_gib(),
        )
        _OPSD_JSD_DETAIL_CAPTURE["active"] = False
        return

    entries = capture.get("entries") or []
    if not entries:
        opsd_debug.log_detail(
            "opsd_jsd",
            "no JSD detail captured (no OPSD samples on this rank or capture disabled)",
            global_step=global_step,
        )
        _OPSD_JSD_DETAIL_CAPTURE["active"] = False
        return

    for k, stats in enumerate(entries):
        opsd_debug.log_detail("opsd_jsd", f"sample[{k}] token-level JSD", global_step=global_step, **stats)

    _OPSD_JSD_DETAIL_CAPTURE["active"] = False
