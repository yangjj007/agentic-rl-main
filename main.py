# train_grpo.py
"""
Main script for training a Llava-based model using the custom MyGRPOTrainer.

This script handles:
1. Configuration loading.
2. Initialization of Weights & Biases (wandb) and Hugging Face Accelerate.
3. Loading the model and processor.
4. Preparing the training and evaluation datasets.
5. Setting up and running the GPRO trainer.
"""
import argparse
import os
from functools import partial
from typing import Dict, Any

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from trl import GRPOConfig

from config.loader import load_config
from data_utils.commom_util import collate_fn, define_task_data_func
from trainer.DyMETrainer import DyMETrainer
from reward_utils.checker import RewardCalculator, RewardCalculatorLocal
from reward_utils.refiner import ContextRefiner, ContextRefinerLocal
from opsd_utils import debug_log as opsd_debug
from opsd_utils.teacher_batching import (
    log_teacher_placement,
    resolve_teacher_device_map,
)
from opsd_utils.deepspeed_utils import (
    deepspeed_zero_stage,
    gradient_checkpointing_enable_kwargs,
    is_deepspeed_accelerate_config,
    should_disable_gradient_checkpointing,
    uses_deepspeed_json_file,
)


def _run_cross_model_vocab_checks(model, processor, teacher_model, model_config: Dict[str, Any]) -> None:
    """Startup checks for cross-model OPD vocab slice + tokenizer alignment."""
    from transformers import AutoProcessor

    from opsd_utils.vocab_align import print_vocab_align_report, verify_shared_tokenizer_alignment

    student_vocab = getattr(model.config, "vocab_size", len(processor.tokenizer))
    teacher_vocab = getattr(teacher_model.config, "vocab_size", student_vocab)
    shared = min(student_vocab, teacher_vocab)
    print(
        f"[OPSD-VOCAB] lm_head widths: student={student_vocab} teacher={teacher_vocab} "
        f"shared_slice={shared}",
        flush=True,
    )
    if student_vocab == teacher_vocab:
        print("[OPSD-VOCAB] vocab sizes match — no slice needed", flush=True)
        return

    teacher_path = model_config.get("teacher_model_path")
    teacher_processor = AutoProcessor.from_pretrained(teacher_path)
    full_scan = os.environ.get("DYME_VOCAB_ALIGN_FULL", "0").strip().lower() in ("1", "true", "yes")
    stride = int(os.environ.get("DYME_VOCAB_ALIGN_STRIDE", "500"))
    report = verify_shared_tokenizer_alignment(
        processor.tokenizer,
        teacher_processor.tokenizer,
        shared_vocab=shared,
        full_scan=full_scan,
        sample_stride=stride,
    )
    print_vocab_align_report(report)


def _wandb_disabled_by_env() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes", "on"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "off"):
        return True
    return False


def _try_wandb_login() -> bool:
    """Return True if wandb credentials are available (env, offline, or prior login)."""
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        return True
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        return True
    try:
        wandb.login(relogin=False)
        key = wandb.api.api_key
        return bool(key and len(key) >= 40)
    except Exception:
        return False


def setup_accelerator_and_wandb(bf16, want_wandb: bool) -> tuple[Accelerator, bool]:
    """
    Initialize Accelerator and optionally wandb.

    Returns:
        (accelerator, use_wandb)
    """
    use_wandb = want_wandb and not _wandb_disabled_by_env()
    if use_wandb:
        use_wandb = _try_wandb_login()

    accel_kwargs: dict = {}
    # bf16 for DDP/MULTI_GPU only; with deepspeed_config_file, precision lives in the JSON.
    if bf16 and not uses_deepspeed_json_file():
        accel_kwargs["mixed_precision"] = "bf16"
    if use_wandb:
        accel_kwargs["log_with"] = "wandb"
    return Accelerator(**accel_kwargs), use_wandb


def load_model_and_processor(model_config: Dict[str, Any]):
    """
    Loads the pre-trained vision-language model and its associated processor.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.

    Returns:
        Tuple[LlavaOnevisionForConditionalGeneration, PreTrainedProcessor]: The loaded model and processor.
    """
    model_id = model_config['pretrained_model_path']

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config['use_flash_attention_2'] else 'sdpa',
        low_cpu_mem_usage=True,
    )

    # Freeze the vision tower to save memory and computation
    model.base_model.vision_tower.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    return model, processor


def load_teacher_model(model_config: Dict[str, Any], *, local_rank: int = 0, num_gpus: int = 1):
    """Load optional frozen teacher for cross-model OPD (e.g. LLaVA-OneVision 7B)."""
    teacher_path = model_config.get("teacher_model_path")
    if not teacher_path:
        return None

    dtype_name = model_config.get("teacher_dtype", model_config.get("torch_dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)
    requested_map = model_config.get("teacher_device_map")
    if not requested_map:
        env_map = os.environ.get("DYME_TEACHER_DEVICE_MAP", "").strip()
        if env_map:
            requested_map = env_map

    device_map = resolve_teacher_device_map(
        requested_map,
        local_rank=local_rank,
        num_gpus=max(1, num_gpus),
    )
    log_teacher_placement(
        local_rank=local_rank,
        num_gpus=max(1, num_gpus),
        teacher_path=teacher_path,
        resolved_device=device_map,
        requested_map=requested_map,
    )

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }

    teacher = LlavaOnevisionForConditionalGeneration.from_pretrained(
        teacher_path,
        attn_implementation='flash_attention_2' if model_config.get('use_flash_attention_2') else 'sdpa',
        **load_kwargs,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    if hasattr(teacher, "base_model") and hasattr(teacher.base_model, "vision_tower"):
        teacher.base_model.vision_tower.requires_grad_(False)

    return teacher


def prepare_datasets(task: str, dataset_config: Dict[str, Any], mode='rl') -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.

    Args:
        task (str): The name of the task (e.g., 'chartqa').
        dataset_config (Dict[str, Any]): Configuration for datasets.

    Returns:
        Tuple[Dataset, Dataset]: The training and evaluation datasets.
    """
    data_func = define_task_data_func(task, mode=mode)

    # Create training dataset
    train_data_list = data_func(json_path=dataset_config['train_dataset'])
    train_dataset = Dataset.from_list(train_data_list)

    # Create evaluation dataset
    if 'chart' in task:
        eval_dataset = load_dataset(dataset_config['eval_dataset'])['test']
        # Note: You can uncomment the line below for quick testing/debugging.
        # eval_dataset = eval_dataset.select(range(1000, 1100))

    else:
        # Extend this section for other tasks if needed in the future.
        eval_dataset = None

    return train_dataset, eval_dataset


def main():
    """
    Main function to orchestrate the model training pipeline.
    """

    parser = argparse.ArgumentParser(description="Train a Llava model using either SFT or GRPO.")

    parser.add_argument(
        '--config', type=str, default='config/config.py',
        help="Python config path (e.g. config/config.py, config/config_trimode.py) "
             "or shorthand alias: norm | trimode | llavacot | low | aok",
    )
    parser.add_argument(
        '--mode', type=str, default='rl',
    )
    parser.add_argument(
        '--opsd_mode', type=str, default=None,
        help="OPSD routing mode: dyme | trimode | rlsd | copsd_opd | opsd_only | replace_sft | opsd_on_wrong | grpo_opsd_joint",
    )
    parser.add_argument(
        '--opsd_providers', type=str, default=None,
        help="Comma-separated privileged providers: text,visual_facts,crop,hybrid",
    )
    parser.add_argument(
        '--opsd_privilege_profile', type=str, default=None,
        help="Privileged profile preset: text | visual | hybrid (default hybrid in config_trimode)",
    )
    parser.add_argument(
        '--reward_weights', type=str, default=None,
        help="Comma-separated reward weights: format,context,acc (e.g. 0.5,1.5,1.0). "
             "Overrides config; env DYME_REWARD_WEIGHTS also supported in antidegen config.",
    )
    parser.add_argument(
        '--opsd_enabled', action='store_true',
        help="Enable OPSD / TriMode training extensions",
    )
    parser.add_argument(
        '--opsd_debug', action='store_true',
        help="Enable verbose OPSD debug logs (or set env DYME_OPSD_DEBUG=1)",
    )
    parser.add_argument(
        '--opsd_detail_every', type=int, default=None,
        help="Emit full weak-signal diagnostic bundle every N global steps on rank 0 "
             "(default 10; config opsd.debug.detail_every or env DYME_OPSD_DETAIL_EVERY)",
    )
    parser.add_argument(
        '--opsd_probe_on_generate', dest='opsd_probe_on_generate', action='store_true',
        help="Emit [OPSD-PROBE] on every (re)generate on rank 0 (config_trimode default on)",
    )
    parser.add_argument(
        '--no_opsd_probe_on_generate', dest='opsd_probe_on_generate', action='store_false',
        help="Disable per-generate [OPSD-PROBE] logs",
    )
    parser.set_defaults(opsd_probe_on_generate=None)
    parser.add_argument(
        '--no_opsd_probe_first_token_logits', dest='opsd_probe_first_token_logits', action='store_false',
        help="Disable pre-generate first-token logits probe ([OPSD-GENDBG])",
    )
    parser.set_defaults(opsd_probe_first_token_logits=None)
    parser.add_argument(
        '--wandb', dest='wandb', action='store_true',
        help="Force enable Weights & Biases logging",
    )
    parser.add_argument(
        '--no_wandb', dest='wandb', action='store_false',
        help="Disable Weights & Biases logging (or set WANDB_MODE=offline/disabled)",
    )
    parser.set_defaults(wandb=None)

    args = parser.parse_args()
    mode = args.mode

    # 1. Load Configurations
    CONFIG = load_config(args.config)
    model_config = CONFIG['model']
    training_config = CONFIG['training']
    rl_config = CONFIG['rl']
    client_config = CONFIG['client']
    dataset_config = CONFIG['dataset']
    task = training_config['task']
    opsd_config = dict(CONFIG.get('opsd', {"enabled": False, "mode": "dyme"}))
    if args.opsd_enabled:
        opsd_config["enabled"] = True
    if args.opsd_mode is not None:
        opsd_config["enabled"] = True
        opsd_config["mode"] = args.opsd_mode
    if args.opsd_providers is not None:
        opsd_config["privileged_providers"] = [p.strip() for p in args.opsd_providers.split(",") if p.strip()]
    if args.opsd_privilege_profile is not None:
        opsd_config["privileged_profile"] = args.opsd_privilege_profile.strip()
    reward_weights_raw = args.reward_weights or os.environ.get("DYME_REWARD_WEIGHTS")
    if reward_weights_raw:
        parts = [p.strip() for p in reward_weights_raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"reward_weights must have exactly 3 comma-separated values (format,context,acc), got: {reward_weights_raw!r}"
            )
        opsd_config["reward_weights"] = [float(p) for p in parts]
    debug_cfg = opsd_config.setdefault("debug", {})
    detail_every = debug_cfg.get("detail_every", 10)
    if args.opsd_detail_every is not None:
        detail_every = max(0, args.opsd_detail_every)
        debug_cfg["detail_every"] = detail_every
    probe_on_generate = debug_cfg.get("probe_on_generate", False)
    if args.opsd_probe_on_generate is not None:
        probe_on_generate = args.opsd_probe_on_generate
        debug_cfg["probe_on_generate"] = probe_on_generate
    probe_first_token_logits = debug_cfg.get("probe_first_token_logits", True)
    if args.opsd_probe_first_token_logits is not None:
        probe_first_token_logits = args.opsd_probe_first_token_logits
        debug_cfg["probe_first_token_logits"] = probe_first_token_logits

    debug_enabled = opsd_debug.configure(
        enabled=args.opsd_debug or None,
        detail_every=detail_every,
        probe_on_generate=probe_on_generate,
        probe_first_token_logits=probe_first_token_logits,
        probe_prompt_tail_tokens=debug_cfg.get("probe_prompt_tail_tokens", 16),
        probe_log_model_context=debug_cfg.get("probe_log_model_context", True),
    )
    if debug_enabled:
        opsd_debug.log_config("main", "resolved OPSD config", opsd_config)
        opsd_debug.log("main", "training entry", mode=mode, config_path=args.config)

    # 2. Setup Environment
    want_wandb = True if args.wandb is None else args.wandb
    accelerator, use_wandb = setup_accelerator_and_wandb(
        bf16=training_config['dyme_args']['bf16'],
        want_wandb=want_wandb,
    )
    if want_wandb and not use_wandb and args.wandb is True:
        raise RuntimeError(
            "wandb was requested (--wandb) but no API key is configured. "
            "Run `wandb login`, set WANDB_API_KEY, or use WANDB_MODE=offline."
        )
    if accelerator.is_main_process:
        if use_wandb:
            print("[DyME] wandb enabled for training logs")
        elif want_wandb:
            print(
                "[DyME] wandb disabled (no credentials). Training continues with report_to=none. "
                "Run `wandb login`, export WANDB_API_KEY, or pass --wandb after configuring."
            )
    device_id = accelerator.process_index
    opsd_debug.configure(
        enabled=debug_enabled,
        detail_every=detail_every,
        probe_on_generate=probe_on_generate,
        probe_first_token_logits=probe_first_token_logits,
        probe_prompt_tail_tokens=debug_cfg.get("probe_prompt_tail_tokens", 16),
        probe_log_model_context=debug_cfg.get("probe_log_model_context", True),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if debug_enabled:
        opsd_debug.log(
            "main",
            "accelerator initialized",
            process_index=accelerator.process_index,
            local_process_index=accelerator.local_process_index,
            num_processes=accelerator.num_processes,
            device=str(accelerator.device),
        )

    visible_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    if visible_gpus == 0:
        raise RuntimeError("No CUDA devices are visible to this process.")
    if accelerator.num_processes > visible_gpus:
        raise RuntimeError(
            f"GPU/process mismatch: launched {accelerator.num_processes} distributed processes "
            f"but only {visible_gpus} CUDA device(s) are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}).\n"
            f"Fix: accelerate launch --num_processes {visible_gpus} ...\n"
            f"Or: bash scripts/train_local_gpus.sh  (auto-detects {visible_gpus} GPU(s))"
        )
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_gpus} GPU(s) visible. "
            f"Reduce --num_processes to {visible_gpus}."
        )
    if accelerator.is_main_process:
        print(
            f"[DyME] Distributed launch OK: num_processes={accelerator.num_processes}, "
            f"visible_gpus={visible_gpus}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    # 3. Initialize Model and Processor
    ds_zero_stage = deepspeed_zero_stage()
    if accelerator.is_main_process and is_deepspeed_accelerate_config():
        print(
            f"[DyME] DeepSpeed enabled via ACCELERATE_CONFIG "
            f"({os.environ.get('ACCELERATE_CONFIG', '<unset>')}), ZeRO stage={ds_zero_stage}",
            flush=True,
        )

    model, processor = load_model_and_processor(model_config)
    if os.environ.get("DYME_GRADIENT_CHECKPOINTING", "").strip().lower() in ("1", "true", "yes", "on"):
        if should_disable_gradient_checkpointing():
            if accelerator.is_main_process:
                print(
                    "[DyME] gradient checkpointing skipped: incompatible with DeepSpeed ZeRO-1/2 "
                    "(multiple student forwards / checkpoint backward). "
                    "Use ZeRO-3, DDP, or DYME_GRADIENT_CHECKPOINTING=0.",
                    flush=True,
                )
        else:
            gc_kwargs = gradient_checkpointing_enable_kwargs()
            if gc_kwargs:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            else:
                model.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                mode = f"use_reentrant={gc_kwargs['use_reentrant']}" if gc_kwargs else "default"
                print(
                    f"[DyME] gradient checkpointing enabled on student "
                    f"(DYME_GRADIENT_CHECKPOINTING, {mode})",
                    flush=True,
                )

    teacher_model = load_teacher_model(
        model_config,
        local_rank=local_rank,
        num_gpus=visible_gpus,
    )
    if accelerator.is_main_process and teacher_model is not None:
        _run_cross_model_vocab_checks(
            model,
            processor,
            teacher_model,
            model_config,
        )

    # 4. Prepare Datasets
    train_dataset, eval_dataset = prepare_datasets(task, dataset_config, mode=mode)

    # 5. Initialize Reward Calculator
    # checker = RewardCalculator(rl_config, client_config.copy(), gpu_id=device_id)
    # refiner = ContextRefiner(rl_config, client_config.copy(), gpu_id=device_id)

    checker = RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=device_id)
    refiner = ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=device_id)
    # 6. Define Training Arguments
    dyme_args = dict(training_config['dyme_args'])
    if ds_zero_stage is not None and ds_zero_stage >= 3:
        dyme_args.setdefault("ds3_gather_for_generation", True)
    if not use_wandb:
        dyme_args["report_to"] = "none"
    training_args = GRPOConfig(**dyme_args)

    collate_fn_with_processor = partial(collate_fn, processor=processor)
    # 7. Initialize the Trainer
    dyme_trainer = DyMETrainer(
        model=model,
        checker=checker,
        refiner=refiner,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        processing_func=collate_fn_with_processor,
        task_name=task,
        end_flag=rl_config['end_flag'],
        opsd_config=opsd_config,
        teacher_model=teacher_model,
    )

    # 8. Start Training
    dyme_trainer.train()

    output_dir = training_args.output_dir
    output_dir = os.path.join(output_dir, "final_checkpoint")
    if accelerator.is_main_process and is_deepspeed_accelerate_config():
        print(
            "[DyME] Saving consolidated student checkpoint (DeepSpeed ZeRO gather if configured)...",
            flush=True,
        )
    dyme_trainer.save_model(output_dir)
    if accelerator.is_main_process:
        processor.save_pretrained(output_dir)
        print(f"Model and processor saved to {output_dir}")
if __name__ == "__main__":
    main()