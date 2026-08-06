"""Standalone checkpoint evaluator for ChartQA.

For training-time evaluation, import :mod:`eval.chartqa_core` instead.  This
thin CLI intentionally owns all checkpoint loading so importing the core never
creates an ``Accelerator`` or reloads a student model.
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ChartQA checkpoint")
    parser.add_argument("--model_path", default=None)
    parser.add_argument(
        "--split",
        default=os.environ.get("DYME_EVAL_SPLIT", "test"),
        help="ChartQA split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=max(1, int(os.environ.get("DYME_EVAL_BATCH_SIZE", "32"))),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.environ.get("DYME_EVAL_MAX_NEW_TOKENS", "1024")),
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=os.environ.get("DYME_EVAL_DO_SAMPLE", "0").lower()
        in ("1", "true", "yes", "on"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("DYME_EVAL_TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(os.environ.get("DYME_EVAL_TOP_P", "1.0")),
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=float(os.environ.get("DYME_EVAL_REPETITION_PENALTY", "1.0")),
    )
    return parser


def _model_id(args: argparse.Namespace) -> str:
    return (
        args.model_path
        or os.environ.get("CHECKPOINT_DIR")
        or "/path/to/dyme-k-8/final_checkpoint"
    )


def main(argv: Sequence[str] | None = None) -> int:
    # The legacy evaluator intentionally ignored launcher-specific extra flags;
    # retain that behavior for existing ``accelerate launch`` wrappers.
    args, _unknown = build_parser().parse_known_args(argv)

    # These imports stay inside the CLI so ``from eval.chartqa_core import ...``
    # can be used by the trainer without a second model load or Accelerator.
    import torch
    from accelerate import Accelerator
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
        LlavaOnevisionForConditionalGeneration,
    )

    from eval.chartqa_core import (
        ChartQAEvaluationConfig,
        evaluate_chartqa_in_memory,
        print_chartqa_evaluation,
    )

    accelerator = Accelerator()
    model_id = _model_id(args)
    if accelerator.is_main_process:
        print(f"Loading model: {model_id}", flush=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Preserve the old loading behavior.  Some custom checkpoints rely on this
    # side effect even though generation goes through the processor tokenizer.
    AutoTokenizer.from_pretrained(model_id, config=config, trust_remote_code=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(accelerator.device)
    processor = AutoProcessor.from_pretrained(model_id)

    if accelerator.is_main_process:
        print(f"Loading ChartQA dataset split: {args.split}", flush=True)
    full_dataset = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)[args.split]

    result = evaluate_chartqa_in_memory(
        model=model,
        processor=processor,
        accelerator=accelerator,
        dataset=full_dataset,
        config=ChartQAEvaluationConfig(
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            input_dtype=torch.bfloat16,
        ),
    )
    if accelerator.is_main_process:
        print_chartqa_evaluation(result)
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
