import os
import torch

from data_utils.paths import OUTPUTS_DIR, project_path

# ====== Model Configuration ======
MODEL_CONFIG = {
    "pretrained_model_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",  # two-stage grpo
    "use_flash_attention_2": True,
    "torch_dtype": "bfloat16",
}

# ====== Training Configuration ======
TRAINING_CONFIG = {
    "task": 'chart',
    "num_gpus": 8,  # 使用的 GPU 数量
    "num_client": 8,  # 并发客户端数量，通常与 GPU 数量相同
    # RL阶段的参数 (根据原脚本的rl_args)
    "dyme_args": {
        "output_dir": os.path.join(OUTPUTS_DIR, "dyme-k-8"),
        "logging_steps": 1,
        "num_generations": 8,  # RL 阶段可以生成多个响应进行比较
        "max_completion_length": 300,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 10,
        "learning_rate": 8e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "beta": 0.0,  # GRPO specific
        "loss_type": 'grpo',  # GRPO specific
        "seed": 42,
    },
    "sft_args": {
        "output_dir": os.path.join(OUTPUTS_DIR, "sft-chart-llava_cot"),
        "logging_steps": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        # "save_steps": 100,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "seed": 42,
        "remove_unused_columns": False
    },
    "grpo_args":{
        "output_dir": os.path.join(OUTPUTS_DIR, "grpo-chart-llava-beta"),
        "logging_steps": 1,
        "num_generations": 4,  # RL 阶段可以生成多个响应进行比较
        "max_completion_length": 576,
        "max_prompt_length": None,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "beta": 0.04,  # GRPO specific
        "loss_type": 'grpo',  # GRPO specific
        "seed": 42,
    }

}

RL_CONFIG = {
    "answer_flag": "Answer:",
    "end_flag": "<|im_end|>"
}

# OPSD / TriMode training (disabled by default -> original DyME behavior)
DYME_OPSD_CONFIG = {
    "enabled": False,
    "mode": "dyme",
    "privileged_profile": "hybrid",
    "privileged_providers": ["text"],
    "privileged_image": {
        "mode": "dual",
        "crop_strategy": "bbox_then_center",
        "bbox_coord": "normalized",
        "margin_ratio": 0.25,
    },
    "privileged_debug": {
        "save_images": True,
        "image_subdir": "logs/images",
        "max_samples_per_detail": 2,
    },
    "gate": {
        "correct_threshold": 0.5,
        "teacher_recoverable": "privileged_available",
        "recoverable_tau": 0.5,
        "use_edge_mask": False,
    },
    "loss": {
        "beta": 0.5,
        "opsd_weight": 1.0,
        "grpo_weight": 1.0,
        "sft_weight": 1.0,
    },
    "debug": {
        # Full weak-signal diagnostic bundle every N global steps (rank 0). 0 = off.
        "detail_every": 10,
        # Lightweight [OPSD-PROBE] on every (re)generate (rank 0). Independent of OPSD-DEBUG.
        "probe_on_generate": False,
        "probe_sample_count": 4,
        # Deep generate diagnostics ([OPSD-GENDBG]) — prompt tail, first-token logits, model context.
        "probe_first_token_logits": True,
        "probe_prompt_tail_tokens": 16,
        "probe_log_model_context": True,
    },
}

# ====== Client Configuration for Reward Calculation ======
CLIENT_CONFIG = {
    "client_type": "openai",  # 客户端主机地址
    "api_key": "none",  # 客户端主机
    "api_base": "http://127.0.0.1:%s/v1",  # 客户端，如果是本地服务需要预留端口
    "timeout": 60,  # 请求超时时间
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  # 使用的模型ID
    "init_port": 23333, # 或者none代表在线服务
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    "train_dataset": project_path("data/chartqa/train_medium_vf_full.json"),
    # 训练数据路径
    "eval_dataset": "HuggingFaceM4/ChartQA",  # 验证数据路径
}

# ====== Full Configuration ======
CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "opsd": DYME_OPSD_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
}

# Save configuration to a file for reference
def save_config(config, config_path="./config.json"):
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Example usage to save config
if __name__ == "__main__":
    save_config(CONFIG)

