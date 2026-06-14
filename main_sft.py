"""
Offline supervised fine-tuning for ChartQA (two-stage cold start before RLSD/OPD).

Usage:
  accelerate launch main_sft.py --config config/config_rlsd_chartqa.py
  bash scripts/train_chartqa_sft.sh
"""
from __future__ import annotations

import argparse
import os
from functools import partial

from accelerate import Accelerator
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from config.loader import load_config
from data_utils.commom_util import collate_fn, define_task_data_func
from main import load_model_and_processor


def main() -> None:
    parser = argparse.ArgumentParser(description="ChartQA offline SFT (hint + Answer GT).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_rlsd_chartqa.py",
        help="Config module (uses training.sft_args and dataset.train_dataset).",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Override CONFIG model path (e.g. base 0.5B before RL).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = dict(config["model"])
    if args.pretrained_model_path:
        model_config["pretrained_model_path"] = args.pretrained_model_path

    training_config = config["training"]
    task = training_config["task"]
    sft_args = dict(training_config.get("sft_args") or config.get("training", {}).get("sft_args", {}))
    if not sft_args:
        raise ValueError("Config must define training.sft_args for offline SFT.")

    output_dir = os.environ.get("DYME_SFT_OUTPUT_DIR", sft_args.get("output_dir", "./outputs/chartqa-sft"))
    sft_args["output_dir"] = output_dir
    sft_args.setdefault("remove_unused_columns", False)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    model, processor = load_model_and_processor(model_config)
    data_func = define_task_data_func(task, mode="sft")
    train_list = data_func(json_path=config["dataset"]["train_dataset"])
    train_dataset = Dataset.from_list(train_list)

    label_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    data_collator = partial(collate_fn, processor=processor, label_id=label_id)

    train_args = TrainingArguments(**sft_args)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
    if accelerator.is_main_process:
        processor.save_pretrained(os.path.join(output_dir, "final_checkpoint"))


if __name__ == "__main__":
    main()
