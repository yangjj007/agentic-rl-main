import itertools
import json
import os
import time
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional, Union

from torch.nn.utils.rnn import pad_sequence

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import TrainOutput, seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
# from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax

import concurrent.futures
from datasets import Dataset, IterableDataset

from reward_utils import checker
from reward_utils.checker import RewardCalculator
from reward_utils.compute_rewards import (
    calculate_rewards_in_parallel,
    calculate_rewards_sequential,
    refine_context_in_parallel,
    refine_context_sequential,
)
from reward_utils.eval_format_reward import score_eval_format_rewards
from reward_utils.perception_reward import (
    score_image_teacher_perception_rewards,
    score_perception_rewards,
)
from data_utils.chart.evaluator import eval_one_chart, eval_teacher_probe_chart
from eval.chartqa_core import ChartQAEvaluationConfig, evaluate_chartqa_in_memory
from opsd_utils.checkpoint_eval import (
    CheckpointEvaluationPolicy,
    CheckpointEvaluationTriggerCallback,
    apply_checkpoint_evaluation_decision,
)

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT, MODE_SKIP, DEFAULT_OPSD_CONFIG
from opsd_utils.chart_cot_quality_gate import (
    ChartCoTQualityGateConfig,
    append_quality_sample_records,
    evaluate_teacher_trajectory_quality,
)
from opsd_utils.adaptive_supervision import (
    AdaptiveSupervisionConfig,
    AdaptiveSupervisionController,
    AdaptiveSupervisionState,
)
from opsd_utils.global_training_signal import (
    GlobalTrainingSignalCounts,
    GlobalTrainingSignalSnapshot,
    counts_from_local_batch,
    snapshot_from_counts,
)
from opsd_utils.dynamic_trigger_monitor import DynamicTriggerConfig, DynamicTriggerMonitor
from opsd_utils.effective_group_filter import (
    EffectiveGroupFilterConfig,
    apply_effective_group_filter_to_routes,
    compute_effective_group_keep_mask,
)
from opsd_utils.indexing import source_row_index
from opsd_utils.leakage import completion_has_leakage_pattern
from opsd_utils.mode_router import (
    route_prompt_modes,
    route_completion_modes,
    teacher_probe_route_confirmed,
)
from opsd_utils.recoverability import estimate_recoverable_flags
from opsd_utils.prompt_builder import build_teacher_prompt_batch
from opsd_utils.privileged.providers import teacher_probe_evidence_status
from opsd_utils.opsd_loss import compute_vlm_opsd_loss_masked_batch, slice_student_completion_logits
from opsd_utils.adaptive_weight import effective_opsd_weight
from opsd_utils.positive_replay import PositiveReplayBuffer, PositiveReplayConfig
from opsd_utils.rollout_replay import RolloutReplayBuffer, RolloutReplayConfig, stack_optional_compatible_tensors
from opsd_utils.signal_aware_routing import (
    CompletionQuality,
    ModeStableRouteState,
    apply_opd_route_cap,
    apply_signal_aware_routing,
    apply_signal_utility_routing,
    is_table_spam_completion,
    local_teacher_traj_indices,
)
from opsd_utils.teacher_sft_repair import (
    apply_teacher_sft_repair_routing,
    build_teacher_sft_repair_target,
    sanitize_teacher_sft_text,
    teacher_sft_repair_advantages,
)
from opsd_utils.teacher_traj_schedule import effective_linear_weight, effective_teacher_traj_weight
from opsd_utils.phase_schedule import boundary_reached, training_progress
from opsd_utils.teacher_probe_log import append_teacher_probe_record, build_teacher_probe_record
from opsd_utils.deepspeed_utils import (
    deepspeed_requires_single_student_forward,
    gradient_checkpointing_enable_kwargs,
    is_deepspeed_accelerate_config,
    student_forward_chunk_size,
    sync_global_sum_count,
)
from opsd_utils.teacher_batching import (
    align_teacher_prompt_image_tokens,
    as_batch_num_images_tensor,
    expand_teacher_tensors_to_full_batch,
    get_teacher_vision_for_sample,
    model_inference_device,
    move_batch_num_images_to_model_device,
    move_pixel_values_to_model_device,
    stack_teacher_vision_for_generate,
)
from opsd_utils import debug_log as opsd_debug
from opsd_utils import diagnostics as opsd_diagnostics
from opsd_utils.health_monitor import TrainingHealthMonitor

if is_wandb_available():
    import wandb


def validate_grpo_batch_geometry(
    *,
    num_generations: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_processes: int,
) -> None:
    if num_generations < 2:
        raise ValueError(
            "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
            f"{num_generations}, which is less than the minimum required."
        )
    effective_batch_size = per_device_train_batch_size * num_processes * gradient_accumulation_steps
    possible_values = [
        n_gen for n_gen in range(2, effective_batch_size + 1) if effective_batch_size % n_gen == 0
    ]
    if num_generations not in possible_values:
        raise ValueError(
            f"The effective train batch size ({num_processes} x {per_device_train_batch_size} x "
            f"{gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
            f"prompt ({num_generations}). Given the current effective train batch size, the valid values for "
            f"the number of generations are: {possible_values}."
        )
    local_effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    if local_effective_batch_size % num_generations != 0:
        raise ValueError(
            f"The local effective batch size ({per_device_train_batch_size} x {gradient_accumulation_steps} = "
            f"{local_effective_batch_size}) must also be evenly divisible by the number of generations per prompt "
            f"({num_generations}) on each rank. Increase per-device batch size or gradient accumulation steps, or "
            f"use a smaller num_generations."
        )



# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class DynamicSignalRepeatSampler(Sampler):
    """Repeat sampler with mutable prompt weights from recent training signal."""

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        after_step: int = 0,
        mixed_weight: float = 4.0,
        all_wrong_weight: float = 1.0,
        all_correct_weight: float = 0.7,
        unknown_weight: float = 1.0,
        reward_std_bonus: float = 2.0,
        schedule_mode: str = "step",
        start_progress: float = 0.5,
        always_active: bool = False,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        self.after_step = max(0, int(after_step))
        self.mixed_weight = float(mixed_weight)
        self.all_wrong_weight = float(all_wrong_weight)
        self.all_correct_weight = float(all_correct_weight)
        self.unknown_weight = float(unknown_weight)
        self.reward_std_bonus = float(reward_std_bonus)
        self.schedule_mode = str(schedule_mode or "step").lower()
        self.start_progress = float(start_progress)
        self.always_active = bool(always_active)
        self.current_step = 0
        self.max_steps: int | None = None
        self.prompt_weights = [max(self.unknown_weight, 1e-6)] * self.num_samples
        self.prompt_states = ["unknown"] * self.num_samples
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    @property
    def enabled_for_step(self) -> bool:
        if self.always_active:
            return True
        from opsd_utils.phase_schedule import boundary_reached

        return boundary_reached(
            self.current_step,
            self.max_steps,
            mode=self.schedule_mode,
            step_boundary=self.after_step,
            progress_boundary=self.start_progress,
        )

    def set_step(self, step: int, max_steps: int | None = None) -> None:
        self.current_step = int(step)
        self.max_steps = int(max_steps) if max_steps is not None else None

    def update_prompt_signal(
        self,
        *,
        dataset_index: int,
        correct_count: int,
        num_generations: int,
        reward_std: float,
    ) -> None:
        idx = int(dataset_index)
        if idx < 0 or idx >= self.num_samples:
            return
        correct = int(correct_count)
        total = max(int(num_generations), 1)
        std_bonus = 1.0 + self.reward_std_bonus * max(0.0, min(float(reward_std), 0.5))
        if correct <= 0:
            state = "all_wrong"
            weight = self.all_wrong_weight
        elif correct >= total:
            state = "all_correct"
            weight = self.all_correct_weight
        else:
            state = "mixed"
            weight = self.mixed_weight * std_bonus
        self.prompt_states[idx] = state
        self.prompt_weights[idx] = max(float(weight), 1e-6)

    def _static_chunks(self) -> list[list[int]]:
        if self.shuffle:
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))
        chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        return [chunk for chunk in chunks if len(chunk) == self.batch_size]

    def _sample_dynamic_chunk(self) -> list[int]:
        weights = torch.tensor(self.prompt_weights, dtype=torch.float)
        if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0:
            weights = torch.ones(self.num_samples, dtype=torch.float)
        replacement = self.batch_size > self.num_samples
        sampled = torch.multinomial(
            weights,
            num_samples=self.batch_size,
            replacement=replacement,
            generator=self.generator,
        )
        return sampled.tolist()

    def __iter__(self):
        num_chunks = self.num_samples // max(self.batch_size, 1)
        static_chunks = self._static_chunks()
        for chunk_idx in range(num_chunks):
            if self.enabled_for_step:
                chunk = self._sample_dynamic_chunk()
            else:
                chunk = static_chunks[chunk_idx]
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Non-tensor metadata (e.g. ``sft_cold_start``) and scalar tensor metadata
    (e.g. batch-level health metrics) are copied into every chunk unchanged.

    When teacher vision tensors are present, uses teacher_num_images-aware slicing
    (LLaVA-OV stacks images on dim 0, not batch size).
    """
    if (
        tensor_dict.get("teacher_pixel_values_list") is not None
        or tensor_dict.get("teacher_pixel_values") is not None
        or tensor_dict.get("teacher_num_images") is not None
    ):
        from opsd_utils.teacher_batching import split_tensor_dict_for_opsd

        return split_tensor_dict_for_opsd(tensor_dict, num_chunks)

    first_tensor = next(
        tensor
        for tensor in tensor_dict.values()
        if isinstance(tensor, torch.Tensor) and tensor.dim() > 0
    )
    chunk_size = first_tensor.shape[0] // num_chunks
    l1 = []
    for i in range(num_chunks):
        dt = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                dt[key] = None
            elif isinstance(tensor, torch.Tensor) and tensor.dim() == 0:
                dt[key] = tensor
            elif isinstance(tensor, torch.Tensor):
                dt[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            else:
                dt[key] = tensor
        l1.append(dt)

    return l1


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class DyMETrainer(Trainer):

    def __init__(
        self,
        model: PreTrainedModel,
        checker = None,
        refiner=None,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        processing_func = None,
        task_name: str = None,
        end_flag: str = '<|im_end|>',
        opsd_config: Optional[dict] = None,
        teacher_model: Optional[PreTrainedModel] = None,
        teacher_model_config: Optional[dict] = None,
        visual_supervision_meta: Optional[dict] = None,
        checkpoint_eval_config: Optional[dict] = None,
        training_stage: str = "rl",
    ):
        self.training_stage = str(training_stage or "rl").strip().lower()
        if self.training_stage not in {"rl", "opd_only"}:
            raise ValueError(f"Unknown training_stage: {training_stage!r}")
        self.opsd_config = opsd_config if opsd_config is not None else dict(DEFAULT_OPSD_CONFIG)
        self.teacher_model = teacher_model
        self._teacher_model_config = teacher_model_config
        self.visual_supervision_meta = visual_supervision_meta or {}
        self.checkpoint_eval_config = dict(checkpoint_eval_config or {})
        self._last_visual_batch_stats: dict[str, Any] = {}
        self._teacher_vocab_checked = False
        self._teacher_probe_preview_logged = False
        self._teacher_sft_repaired_prompt_keys: set[str] = set()
        self._mode_stable_route_states: dict[str, ModeStableRouteState] = {}
        self._effective_signal_sampler: DynamicSignalRepeatSampler | None = None
        self._positive_replay_buffer: PositiveReplayBuffer | None = None
        self._rollout_replay_buffer: RolloutReplayBuffer | None = None
        self._init_adaptive_supervision_controller()
        dynamic_cfg = self.opsd_config.get("dynamic_trigger_monitor") or {}
        self._dynamic_trigger_monitor = (
            DynamicTriggerMonitor(
                DynamicTriggerConfig(
                    ema_alpha=float(dynamic_cfg.get("ema_alpha", 0.10)),
                    min_progress=float(dynamic_cfg.get("min_progress", 0.20)),
                    patience_steps=int(dynamic_cfg.get("patience_steps", 20)),
                    sampling_mixed_max=float(dynamic_cfg.get("sampling_mixed_max", 0.20)),
                    sampling_zero_loss_min=float(dynamic_cfg.get("sampling_zero_loss_min", 0.70)),
                    rl_mixed_min=float(dynamic_cfg.get("rl_mixed_min", 0.30)),
                    rl_zero_loss_max=float(dynamic_cfg.get("rl_zero_loss_max", 0.30)),
                )
            )
            if bool(dynamic_cfg.get("enabled", False))
            else None
        )
        self._dynamic_trigger_last_step: int | None = None
        self._last_training_phase: Optional[str] = None
        self._perf_timing_enabled = False
        self._perf_step_start_s: Optional[float] = None
        self.task_name = task_name
        reward_weights = self.opsd_config.get("reward_weights", [1.0, 1.0, 1.0])
        if len(reward_weights) != 3:
            raise ValueError(
                f"opsd_config reward_weights must have length 3 (format, context, acc), got {reward_weights}"
            )
        self.reward_weights = torch.nn.Parameter(
            torch.tensor(reward_weights, dtype=torch.float32),
            requires_grad=False,
        )
        self.reward_func_names = ['format', 'thinking', 'accuracy']
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.end_flag = end_flag
        self.checker = checker
        self.refiner = refiner
        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None
        model.warnings_issued["estimate_tokens"] = True
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # This is intentionally separate from Trainer's generic evaluation
        # loop.  ChartQA checkpoint selection must run greedy inference on the
        # resident student, rather than GRPO's reward/evaluation path or a
        # freshly loaded checkpoint.
        self.checkpoint_eval_policy = None
        if self.checkpoint_eval_config.get("enabled", False):
            # The callback is an ExportableState, so use its policy instance.
            # This preserves best-score/patience state when Trainer restores
            # callback state during resume.
            checkpoint_callback = next(
                (
                    callback
                    for callback in self.callback_handler.callbacks
                    if isinstance(callback, CheckpointEvaluationTriggerCallback)
                ),
                None,
            )
            if checkpoint_callback is None:
                self.checkpoint_eval_policy = CheckpointEvaluationPolicy(
                    patience=int(self.checkpoint_eval_config.get("patience", 3)),
                    tie_policy=str(self.checkpoint_eval_config.get("tie_policy", "reset")),
                )
            else:
                self.checkpoint_eval_policy = checkpoint_callback.policy
        self.best_checkpoint_score: Optional[float] = None
        self.best_checkpoint_step: Optional[int] = None
        self.best_checkpoint_path: Optional[str] = None
        self.checkpoint_eval_state = (
            self.checkpoint_eval_policy.state if self.checkpoint_eval_policy is not None else None
        )

        # Reference model
        self.beta = args.beta
        assert self.beta == 0

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        num_processes = self.accelerator.num_processes
        effective_batch_size = (
            args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        )
        validate_grpo_batch_geometry(
            num_generations=self.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_processes=num_processes,
        )
        set_seed(args.seed, device_specific=True)


        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
            use_cache=False if self.args.gradient_checkpointing else True
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        self.processing_func = processing_func
        replay_cfg = PositiveReplayConfig.from_mapping(self.opsd_config.get("positive_replay"))
        self._positive_replay_buffer = PositiveReplayBuffer(
            replay_cfg,
            process_index=getattr(self.accelerator, "process_index", 0),
        )
        rollout_replay_cfg = RolloutReplayConfig.from_mapping(self.opsd_config.get("rollout_replay"))
        self._rollout_replay_buffer = RolloutReplayBuffer(
            rollout_replay_cfg,
            process_index=getattr(self.accelerator, "process_index", 0),
        )

        debug_cfg = self.opsd_config.get("debug", {})
        health_cfg = debug_cfg.get("health_monitor", {})
        detail_every = debug_cfg.get("detail_every", 10)
        probe_on_generate = debug_cfg.get("probe_on_generate", False)
        self._opsd_probe_sample_count = int(debug_cfg.get("probe_sample_count", 4))
        self._generate_call_index = 0
        self._last_generate_probe_stats = None
        self._last_logits_stats: dict[str, float] = {}
        self._health_monitor = (
            TrainingHealthMonitor(health_cfg) if health_cfg.get("enabled", True) else None
        )
        opsd_debug.configure(
            # ``main.py`` may already have enabled detailed OPSD tracing via
            # --opsd_debug.  Do not clear that state while registering the
            # per-rank settings below; otherwise the loss/teacher-forward
            # intermediate diagnostics silently disappear after construction.
            enabled=opsd_debug.is_enabled() or bool(debug_cfg.get("verbose", False)),
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
            detail_every=detail_every,
            probe_on_generate=probe_on_generate,
            probe_first_token_logits=debug_cfg.get("probe_first_token_logits", True),
            probe_prompt_tail_tokens=debug_cfg.get("probe_prompt_tail_tokens", 16),
            probe_log_model_context=debug_cfg.get("probe_log_model_context", True),
            hang_debug=debug_cfg.get("hang_debug", False),
            hang_force=debug_cfg.get("hang_force", True),
            health_monitor_enabled=health_cfg.get("enabled", True),
            health_log_on_generate=health_cfg.get("log_on_generate", True),
            health_log_every_step=health_cfg.get("log_every_step", True),
            health_log_detail_bundle=health_cfg.get("log_detail_bundle", True),
            health_log_alerts_immediately=health_cfg.get("log_alerts_immediately", True),
        )
        opsd_debug.log_config("init", "DyMETrainer OPSD config loaded", self.opsd_config)
        try:
            from opsd_utils.deepspeed_utils import deepspeed_zero_stage, is_deepspeed_accelerate_config

            if is_deepspeed_accelerate_config():
                opsd_debug.log(
                    "init",
                    "DeepSpeed accelerate layout",
                    zero_stage=deepspeed_zero_stage(),
                    teacher_colocated=bool(self.teacher_model is not None),
                    ds3_gather_for_generation=getattr(self.args, "ds3_gather_for_generation", None),
                )
        except Exception:
            pass
        if self.teacher_model is not None:
            self.teacher_model.eval()
        if self.accelerator.is_main_process and detail_every > 0:
            print(
                f"[OPSD-DETAIL] periodic full diagnostics every {detail_every} global steps "
                "(set opsd.debug.detail_every=0 to disable)"
            )
        opsd_debug.log(
            "init",
            "trainer distributed layout",
            task_name=self.task_name,
            num_generations=self.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_processes=num_processes,
            effective_batch_size=effective_batch_size,
            device=str(self.accelerator.device),
            local_rank=self.accelerator.local_process_index,
            process_index=self.accelerator.process_index,
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        dl = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return dl

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        effective_cfg = self.opsd_config.get("effective_sampling") or {}
        if bool(effective_cfg.get("enabled", False)):
            sampler = DynamicSignalRepeatSampler(
                data_source=self.train_dataset,
                mini_repeat_count=self.num_generations,
                batch_size=effective_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                after_step=int(effective_cfg.get("after_step", 294) or 294),
                schedule_mode=str(effective_cfg.get("schedule_mode", "step") or "step"),
                start_progress=float(effective_cfg.get("start_progress", 0.5)),
                mixed_weight=float(effective_cfg.get("mixed_weight", 4.0) or 4.0),
                all_wrong_weight=float(effective_cfg.get("all_wrong_weight", 1.0) or 1.0),
                all_correct_weight=float(effective_cfg.get("all_correct_weight", 0.7) or 0.7),
                unknown_weight=float(effective_cfg.get("unknown_weight", 1.0) or 1.0),
                reward_std_bonus=float(effective_cfg.get("reward_std_bonus", 2.0) or 2.0),
                always_active=self._adaptive_supervision_controller is not None,
            )
            self._effective_signal_sampler = sampler
            return sampler
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return DistributedSampler(
            dataset=eval_dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        ds_gc_kwargs = gradient_checkpointing_enable_kwargs()
        if ds_gc_kwargs is not None:
            gradient_checkpointing_kwargs = {**gradient_checkpointing_kwargs, **ds_gc_kwargs}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        enable_kwargs = (
            {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
            if gradient_checkpointing_kwargs
            else {}
        )
        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable(**enable_kwargs)
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable(**enable_kwargs)

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_sizes,
        logits_to_keep,
        batch_size=None,
        return_completion_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from opsd_utils.opsd_loss import _slice_image_sizes_batch
        from opsd_utils.teacher_batching import (
            align_teacher_prompt_image_tokens,
            student_batch_num_images_tensor,
        )

        batch_size = batch_size or input_ids.size(0)
        has_vision = pixel_values is not None
        chunk_size = student_forward_chunk_size(batch_size, has_vision)
        if deepspeed_requires_single_student_forward() and chunk_size > 1 and self.accelerator.is_main_process:
            opsd_debug.log(
                "opsd_loss",
                "DeepSpeed ZeRO-1/2: single student forward for full local batch",
                chunk_size=chunk_size,
                batch_size=batch_size,
            )
        all_logps = []
        all_completion_logits = [] if return_completion_logits else None
        for i in range(0, input_ids.size(0), chunk_size):
            end = i + chunk_size
            input_ids_batch = input_ids[i:end]
            attention_mask_batch = attention_mask[i:end]
            pixel_values_batch = pixel_values[i:end] if pixel_values is not None else None
            image_sizes_batch = _slice_image_sizes_batch(image_sizes, i, end)
            batch_rows = input_ids_batch.size(0)
            prompt_len = input_ids_batch.size(1) - logits_to_keep
            prompt_ids_batch = input_ids_batch[:, :prompt_len]
            prompt_mask_batch = attention_mask_batch[:, :prompt_len]
            completion_ids_batch = input_ids_batch[:, prompt_len:]
            completion_mask_batch = attention_mask_batch[:, prompt_len:]

            batch_num_images = student_batch_num_images_tensor(pixel_values_batch, batch_rows)
            if (
                pixel_values_batch is not None
                and self.processing_class is not None
                and batch_num_images is not None
            ):
                prompt_ids_batch, prompt_mask_batch = align_teacher_prompt_image_tokens(
                    model,
                    self.processing_class,
                    prompt_ids_batch,
                    prompt_mask_batch,
                    pixel_values_batch,
                    image_sizes_batch,
                    batch_num_images=batch_num_images,
                )
                input_ids_batch = torch.cat([prompt_ids_batch, completion_ids_batch], dim=1)
                attention_mask_batch = torch.cat([prompt_mask_batch, completion_mask_batch], dim=1)

            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }
            if pixel_values_batch is not None:
                forward_kwargs["pixel_values"] = pixel_values_batch
                forward_kwargs["image_sizes"] = image_sizes_batch
                forward_kwargs["batch_num_images"] = batch_num_images
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            raw_logits = model(**forward_kwargs).logits
            completion_logits = slice_student_completion_logits(raw_logits, logits_to_keep)
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            logits = completion_logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
            if all_completion_logits is not None:
                all_completion_logits.append(completion_logits)
        logps_out = torch.cat(all_logps, dim=0)
        if return_completion_logits:
            return logps_out, torch.cat(all_completion_logits, dim=0)
        return logps_out

    def _current_global_step(self) -> int:
        from opsd_utils.gate_policy import current_global_step

        return current_global_step(self)

    def _max_training_steps(self) -> Optional[int]:
        from opsd_utils.gate_policy import resolve_max_training_steps, sft_cold_start_steps

        resolved = resolve_max_training_steps(self)
        if not getattr(self, "_max_steps_logged", False):
            self._max_steps_logged = True
            if self.accelerator.is_main_process:
                cold_steps = sft_cold_start_steps(self.opsd_config, resolved)
                opsd_debug.log_probe(
                    "phase",
                    "resolved training horizon",
                    global_step=self._current_global_step(),
                    max_steps=resolved,
                    cold_start_steps=cold_steps,
                    sft_cold_start_frac=self.opsd_config.get("gate", {}).get("sft_cold_start_frac"),
                )
        return resolved

    def _record_dynamic_trigger_metrics(
        self,
        *,
        mode: str,
        global_step: int,
        health_metrics: dict[str, float],
    ) -> None:
        if (
            mode != "train"
            or self._dynamic_trigger_monitor is None
            or self._dynamic_trigger_last_step == int(global_step)
        ):
            return

        max_training_steps = self._max_training_steps()
        current_progress = training_progress(global_step, max_training_steps)
        trigger_metrics = self._dynamic_trigger_monitor.update(
            mixed_rate=float(health_metrics.get("signal/group_mixed_rate", 0.0)),
            zero_loss_rate=float(health_metrics.get("signal/grpo_zero_loss_rate", 1.0)),
            progress=current_progress,
        )
        self._dynamic_trigger_last_step = int(global_step)
        self._metrics[mode].setdefault("phase/training_progress", []).append(current_progress)
        self._metrics[mode].setdefault("phase/max_training_steps", []).append(float(max_training_steps or 0))
        for key, value in trigger_metrics.items():
            self._metrics[mode].setdefault(f"phase/{key}", []).append(float(value))

        phase_mode = str((self.opsd_config.get("phase_schedule") or {}).get("mode", "step") or "step")
        effective_cfg = self.opsd_config.get("effective_sampling") or {}
        loss_cfg = self.opsd_config.get("loss") or {}
        cap_cfg = loss_cfg.get("route_cap") or {}
        opsd_decay_cfg = loss_cfg.get("weight_decay") or {}
        traj_decay_cfg = (self.opsd_config.get("teacher_trajectory") or {}).get("weight_decay") or {}
        phase_specs = {
            "effective_sampling_active": (
                bool(effective_cfg.get("enabled", False)),
                effective_cfg,
                294,
                0.50,
            ),
            "opd_route_cap_active": (bool(cap_cfg.get("enabled", False)), cap_cfg, 0, 0.50),
            "opd_decay_active": (bool(opsd_decay_cfg.get("enabled", False)), opsd_decay_cfg, 294, 0.50),
            "teacher_traj_decay_active": (
                bool(traj_decay_cfg.get("enabled", False)),
                traj_decay_cfg,
                294,
                0.25,
            ),
        }
        for metric_name, (enabled, cfg, default_step, default_progress) in phase_specs.items():
            active = enabled and boundary_reached(
                global_step,
                max_training_steps,
                mode=str(cfg.get("schedule_mode", phase_mode)),
                step_boundary=int(cfg.get("after_step", cfg.get("start_step", default_step))),
                progress_boundary=float(cfg.get("start_progress", default_progress)),
            )
            self._metrics[mode].setdefault(f"phase/{metric_name}", []).append(float(active))

    def _perf_sync(self) -> None:
        if not self._perf_timing_enabled or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(self.accelerator.device)
        except Exception:
            torch.cuda.synchronize()

    def _perf_start(self) -> Optional[float]:
        if not self._perf_timing_enabled:
            return None
        self._perf_sync()
        return time.perf_counter()

    def _perf_elapsed(self, start: Optional[float]) -> float:
        if start is None:
            return 0.0
        self._perf_sync()
        return float(time.perf_counter() - start)

    def _perf_metric(self, mode: str, name: str, value: Any) -> None:
        if not self._perf_timing_enabled:
            return
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            return
        self._metrics[mode].setdefault(f"perf/{name}", []).append(metric_value)

    def _in_sft_cold_start(self) -> bool:
        from opsd_utils.gate_policy import in_sft_cold_start

        return in_sft_cold_start(
            self.opsd_config,
            self._current_global_step(),
            self._max_training_steps(),
        )

    def _opsd_distributed_barrier(self, label: str) -> None:
        if self.accelerator.num_processes > 1:
            opsd_debug.hang_probe("distributed_barrier_enter", label=label)
            opsd_debug.log_sync_point("dist", label)
            self.accelerator.wait_for_everyone()
            opsd_debug.hang_probe("distributed_barrier_done", label=label)

    def _ensure_teacher_model(self) -> None:
        if self.teacher_model is not None or not self._teacher_model_config:
            self._run_teacher_vocab_check_once()
            return
        teacher_path = self._teacher_model_config.get("teacher_model_path")
        if not teacher_path:
            return
        from main import load_teacher_model

        local_rank = self.accelerator.local_process_index
        num_gpus = max(1, self.accelerator.num_processes)
        self.teacher_model = load_teacher_model(
            self._teacher_model_config,
            local_rank=local_rank,
            num_gpus=num_gpus,
        )
        if self.teacher_model is not None:
            self.teacher_model.eval()
        self.accelerator.wait_for_everyone()
        opsd_debug.log_probe(
            "phase",
            "lazy teacher loaded after SFT cold start",
            global_step=self._current_global_step(),
            teacher_path=teacher_path,
        )
        self._run_teacher_vocab_check_once()

    def _bind_visual_teacher(self) -> None:
        if self.teacher_model is None:
            return
        for component in (self.checker, self.refiner):
            bind = getattr(component, "bind_teacher", None)
            if callable(bind):
                bind(self.teacher_model, self.processing_class)

    def _expand_visual_batch_rows(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        expanded_count: int,
    ) -> tuple[list[dict], list[Any], list[str]]:
        raw_count = len(inputs)
        samples: list[dict] = []
        images: list[Any] = []
        questions: list[str] = []
        for row in range(expanded_count):
            src = self._source_row_index(row, raw_count, expanded_count)
            sample = inputs[src]
            samples.append(sample)
            images.append(sample.get("image"))
            questions.append(sample.get("question_wo_prompt", sample.get("prompt", "")))
        return samples, images, questions

    def _refiner_skip_cold_start(self) -> bool:
        visual_cfg = self.visual_supervision_meta.get("visual_config") or {}
        return bool(visual_cfg.get("refiner", {}).get("skip_cold_start", True))

    def _prepare_visual_supervision_batch(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        *,
        global_step: int,
        expanded_count: int,
    ) -> None:
        if not self.visual_supervision_meta.get("enabled"):
            return
        if self.visual_supervision_meta.get("needs_teacher"):
            self._ensure_teacher_model()
            self._bind_visual_teacher()
        samples, images, questions = self._expand_visual_batch_rows(inputs, expanded_count)
        if hasattr(self.checker, "begin_generate_batch"):
            self.checker.begin_generate_batch(
                samples=samples,
                images=images,
                questions=questions,
                global_step=global_step,
                output_dir=self.args.output_dir,
            )
        recorder = getattr(self.checker, "_recorder", None)
        ic_cache = getattr(self.checker, "_ic_cache", None)
        skip_cold_start = self._in_sft_cold_start() and self._refiner_skip_cold_start()
        if hasattr(self.refiner, "begin_generate_batch"):
            self.refiner.begin_generate_batch(
                samples=samples,
                images=images,
                questions=questions,
                global_step=global_step,
                output_dir=self.args.output_dir,
                recorder=recorder,
                ic_cache=ic_cache,
                skip_cold_start=skip_cold_start,
            )

    def _finish_visual_supervision_batch(self, global_step: int) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if hasattr(self.checker, "end_generate_batch"):
            stats = self.checker.end_generate_batch() or {}
        if hasattr(self.refiner, "end_generate_batch"):
            self.refiner.end_generate_batch()
        self._last_visual_batch_stats = stats
        if self._health_monitor is not None and stats:
            self._health_monitor.record_visual(global_step, stats)
        self._opsd_distributed_barrier("wait_for_everyone after visual supervision batch")
        return stats

    def _run_teacher_vocab_check_once(self) -> None:
        if self._teacher_vocab_checked or self.teacher_model is None:
            return
        self._teacher_vocab_checked = True
        if not self.accelerator.is_main_process:
            return
        try:
            from main import _run_cross_model_vocab_checks

            _run_cross_model_vocab_checks(
                self.model,
                self.processing_class,
                self.teacher_model,
                self._teacher_model_config or {},
            )
        except Exception as exc:
            print(f"[OPSD-VOCAB] WARNING: lazy teacher vocab check failed: {exc}", flush=True)

    def _teacher_probe_config(self) -> dict[str, Any]:
        cfg = self.opsd_config.get("teacher_probe") or {}
        return {
            "enabled": bool(cfg.get("enabled", self.opsd_config.get("mode") == "dyme_teacher_probe_opd")),
            "context_providers": cfg.get(
                "context_providers",
                self.opsd_config.get("privileged_providers", ["format_only", "visual_facts"]),
            ),
            "batch_size": max(1, int(cfg.get("batch_size", 1) or 1)),
            "max_per_batch": int(cfg.get("max_per_batch", 0) or 0),
            "max_new_tokens": int(cfg.get("max_new_tokens", 96)),
            "do_sample": bool(cfg.get("do_sample", False)),
            "temperature": float(cfg.get("temperature", 0.0) or 0.0),
            "top_p": float(cfg.get("top_p", 1.0)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 1.2)),
            "max_relative_change": float(cfg.get("max_relative_change", 0.05)),
            "prompt_profile": cfg.get("prompt_profile", "chartqa_short_answer"),
            "answer_parser": cfg.get("answer_parser", "chartqa_final_answer"),
            "skip_no_evidence": bool(cfg.get("skip_no_evidence", True)),
            "probe_all_wrong_after_step": cfg.get("probe_all_wrong_after_step"),
        }

    def _teacher_trajectory_config(self) -> dict[str, Any]:
        cfg = self.opsd_config.get("teacher_trajectory") or {}
        decay_cfg = cfg.get("weight_decay") or {}
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "max_new_tokens": int(cfg.get("max_new_tokens", 128)),
            "loss_type": (cfg.get("loss_type") or "fkl").lower(),
            "weight": float(cfg.get("weight", self.opsd_config.get("loss", {}).get("teacher_traj_fkl_weight", 0.5))),
            "weight_decay": {
                "enabled": bool(decay_cfg.get("enabled", False)),
                "start_step": int(decay_cfg.get("start_step", 294) or 294),
                "end_step": int(decay_cfg.get("end_step", 441) or 441),
                "start_progress": float(decay_cfg.get("start_progress", 0.25)),
                "end_progress": float(decay_cfg.get("end_progress", 0.50)),
                "final_weight": float(decay_cfg.get("final_weight", 0.0) or 0.0),
            },
        }

    def _teacher_correct_repair_config(self) -> dict[str, Any]:
        cfg = self.opsd_config.get("teacher_correct_repair") or {}
        return {
            "mode": (cfg.get("mode") or "opd").lower(),
            "scope": (cfg.get("scope") or "all_wrong").lower(),
            "slots_per_prompt": max(0, int(cfg.get("slots_per_prompt", 1) or 0)),
            "target_max_tokens": max(1, int(cfg.get("target_max_tokens", 256) or 256)),
            "sanitize_privileged": bool(cfg.get("sanitize_privileged", True)),
            "target_constraint": (cfg.get("target_constraint") or "chartqa_hint").lower(),
            "target_style": (cfg.get("target_style") or cfg.get("target_constraint") or "chartqa_hint").lower(),
        }

    def _online_sft_target_style(self) -> str:
        gate_cfg = self.opsd_config.get("gate") or {}
        return str(gate_cfg.get("online_sft_target_style") or "").strip().lower()

    def _format_online_sft_target(
        self,
        hint: Any,
        answer: Any,
        sample: dict[str, Any] | None = None,
    ) -> str:
        style = self._online_sft_target_style()
        if style:
            raw_target = f"{str(hint or '').strip()}\n{str(answer or '').strip()}".strip()
            constrained = build_teacher_sft_repair_target(
                raw_target,
                sample=sample,
                reference_answer=answer,
                target_style=style,
                sanitize_privileged=True,
            )
            target = constrained.text
        else:
            target = f"{hint}\n{answer}"
        if self.end_flag and not target.endswith(self.end_flag):
            target = f"{target}{self.end_flag}"
        return target

    def _build_online_sft_targets(
        self,
        hints: list[Any],
        answers: list[Any],
        samples: list[dict[str, Any]],
    ) -> list[str]:
        targets: list[str] = []
        for row, (hint, answer) in enumerate(zip(hints, answers)):
            sample = samples[row] if row < len(samples) else {}
            targets.append(self._format_online_sft_target(hint, answer, sample))
        return targets

    def _chart_cot_quality_gate_config(self) -> ChartCoTQualityGateConfig:
        return ChartCoTQualityGateConfig.from_mapping(
            self.opsd_config.get("chart_cot_quality_gate")
        )

    def _init_adaptive_supervision_controller(self) -> None:
        cfg = self.opsd_config.get("adaptive_supervision") or {}
        self._adaptive_supervision_controller: AdaptiveSupervisionController | None = None
        self._adaptive_supervision_state: AdaptiveSupervisionState | None = None
        self._adaptive_supervision_readiness_source = str(
            cfg.get("readiness_source", "mixed_zero") or "mixed_zero"
        ).lower()
        if not bool(cfg.get("enabled", False)):
            return
        controller_cfg = AdaptiveSupervisionConfig(
            ema_alpha=float(cfg.get("ema_alpha", 0.10)),
            target_readiness=float(cfg.get("target_readiness", 0.20)),
            opsd_initial_weight=float(cfg.get("opsd_initial_weight", 1.50)),
            opsd_final_weight=float(cfg.get("opsd_final_weight", 0.50)),
            teacher_initial_weight=float(cfg.get("teacher_initial_weight", 0.50)),
            teacher_final_weight=float(cfg.get("teacher_final_weight", 0.0)),
            opd_initial_cap=int(cfg.get("opd_initial_cap", self.num_generations if hasattr(self, "num_generations") else 8)),
            opd_final_cap=int(cfg.get("opd_final_cap", 2)),
        )
        self._adaptive_supervision_controller = AdaptiveSupervisionController(controller_cfg)
        self._adaptive_supervision_state = self._adaptive_supervision_controller.state

    def _update_adaptive_supervision(
        self,
        *,
        mode: str,
        global_step: int,
        prompt_count: int,
        mixed_count: int,
        zero_loss_count: int,
    ) -> AdaptiveSupervisionState | None:
        controller = self._adaptive_supervision_controller
        if controller is None:
            return None
        if self._adaptive_supervision_readiness_source == "global_grpo_route":
            return self._adaptive_supervision_state
        counts = torch.tensor(
            [prompt_count, mixed_count, zero_loss_count],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        counts = self.accelerator.reduce(counts, reduction="sum")
        total = max(float(counts[0].item()), 1.0)
        state = controller.update(
            step=int(global_step),
            mixed_rate=float(counts[1].item()) / total,
            zero_loss_rate=float(counts[2].item()) / total,
        )
        self._adaptive_supervision_state = state
        self._log_adaptive_supervision_state(mode=mode, state=state)
        return state

    def _update_adaptive_supervision_from_signal(
        self,
        *,
        mode: str,
        global_step: int,
        signal_rate: float,
    ) -> AdaptiveSupervisionState | None:
        controller = self._adaptive_supervision_controller
        if controller is None:
            return None
        if self._adaptive_supervision_readiness_source != "global_grpo_route":
            return self._adaptive_supervision_state
        state = controller.update_signal(step=int(global_step), signal_rate=signal_rate)
        self._adaptive_supervision_state = state
        self._log_adaptive_supervision_state(
            mode=mode,
            state=state,
            signal_rate=state.mixed_rate,
            signal_ema=state.mixed_ema,
        )
        return state

    def _log_adaptive_supervision_state(
        self,
        *,
        mode: str,
        state: AdaptiveSupervisionState,
        signal_rate: float | None = None,
        signal_ema: float | None = None,
    ) -> None:
        values = {
            "enabled": 1.0,
            "mixed_rate": state.mixed_rate,
            "zero_loss_rate": state.zero_loss_rate,
            "mixed_ema": state.mixed_ema,
            "zero_loss_ema": state.zero_loss_ema,
            "readiness": state.readiness,
            "mastery": state.mastery,
            "supervision": state.supervision,
            "opsd_weight": state.opsd_weight,
            "teacher_traj_weight": state.teacher_traj_weight,
            "opd_max_per_prompt": float(state.opd_max_per_prompt),
            "update_count": float(state.update_count),
        }
        if signal_rate is not None:
            values["signal_rate"] = float(signal_rate)
        if signal_ema is not None:
            values["signal_ema"] = float(signal_ema)
        for name, value in values.items():
            self._metrics[mode].setdefault(f"adaptive/{name}", []).append(float(value))

    def _adaptive_opd_route_cap_config(self) -> dict[str, Any] | None:
        state = self._adaptive_supervision_state
        if self._adaptive_supervision_controller is None or state is None:
            return None
        legacy = self._opd_route_cap_config()
        return {
            **legacy,
            "enabled": True,
            "after_step": 0,
            "schedule_mode": "step",
            "start_progress": 0.0,
            "max_per_prompt": int(state.opd_max_per_prompt),
        }

    def _reduce_global_training_signal(
        self,
        *,
        mode: str,
        counts: GlobalTrainingSignalCounts,
        global_step: int | None = None,
    ) -> GlobalTrainingSignalSnapshot:
        field_names = tuple(counts.__dataclass_fields__)
        values = torch.tensor(
            [float(getattr(counts, name)) for name in field_names],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        reduced = self.accelerator.reduce(values, reduction="sum").detach().cpu().tolist()
        reduced_values = dict(zip(field_names, reduced))
        reduced_counts = GlobalTrainingSignalCounts(
            **{
                name: (
                    float(value)
                    if name == "accuracy_reward_sum"
                    else int(round(float(value)))
                )
                for name, value in reduced_values.items()
            }
        )
        snapshot = snapshot_from_counts(reduced_counts)
        for name, value in snapshot.__dict__.items():
            self._metrics[mode].setdefault(f"global_signal/{name}", []).append(float(value))
        if global_step is not None:
            self._update_adaptive_supervision_from_signal(
                mode=mode,
                global_step=int(global_step),
                signal_rate=snapshot.grpo_route_rate,
            )
        return snapshot

    def _global_signal_logging_enabled(self) -> bool:
        cfg = self.opsd_config.get("global_signal_logging") or {}
        return bool(cfg.get("enabled", False)) or (
            self._adaptive_supervision_controller is not None
            and self._adaptive_supervision_readiness_source == "global_grpo_route"
        )

    def _adaptive_loss_weights(self) -> tuple[float, float] | None:
        state = self._adaptive_supervision_state
        if self._adaptive_supervision_controller is None or state is None:
            return None
        return float(state.opsd_weight), float(state.teacher_traj_weight)

    def _opd_route_cap_config(self) -> dict[str, Any]:
        loss_cfg = self.opsd_config.get("loss") or {}
        cap_cfg = loss_cfg.get("route_cap") or {}
        return {
            "enabled": bool(cap_cfg.get("enabled", False)),
            "max_per_prompt": max(0, int(cap_cfg.get("max_per_prompt", 0) or 0)),
            "after_step": max(0, int(cap_cfg.get("after_step", 0) or 0)),
            "schedule_mode": str(cap_cfg.get("schedule_mode", "step") or "step").lower(),
            "start_progress": float(cap_cfg.get("start_progress", 0.5)),
            "overflow_route": (cap_cfg.get("overflow_route") or "sft").lower(),
        }

    def _signal_utility_routing_config(self) -> dict[str, Any]:
        cfg = self.opsd_config.get("signal_utility_routing") or {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "reward_std_scale": float(cfg.get("reward_std_scale", 0.10) or 0.10),
            "allow_grpo_on_format_bad": bool(cfg.get("allow_grpo_on_format_bad", False)),
            "grpo_base": float(cfg.get("grpo_base", 0.05)),
            "grpo_correct_bonus": float(cfg.get("grpo_correct_bonus", 1.00)),
            "grpo_mixed_bonus": float(cfg.get("grpo_mixed_bonus", 0.25)),
            "grpo_signal_weight": float(cfg.get("grpo_signal_weight", 0.40)),
            "grpo_readiness_weight": float(cfg.get("grpo_readiness_weight", 0.70)),
            "opd_base": float(cfg.get("opd_base", 0.10)),
            "opd_teacher_bonus": float(cfg.get("opd_teacher_bonus", 0.45)),
            "opd_wrong_bonus": float(cfg.get("opd_wrong_bonus", 0.20)),
            "opd_gap_weight": float(cfg.get("opd_gap_weight", 0.55)),
            "opd_all_wrong_bonus": float(cfg.get("opd_all_wrong_bonus", 0.20)),
            "opd_teacher_need_weight": float(cfg.get("opd_teacher_need_weight", 0.60)),
            "opd_format_penalty": float(cfg.get("opd_format_penalty", 1.00)),
            "sft_base": float(cfg.get("sft_base", 0.02)),
            "sft_format_bad_bonus": float(cfg.get("sft_format_bad_bonus", 1.10)),
            "sft_clipped_bonus": float(cfg.get("sft_clipped_bonus", 0.80)),
            "sft_all_wrong_bonus": float(cfg.get("sft_all_wrong_bonus", 0.20)),
            "sft_low_signal_bonus": float(cfg.get("sft_low_signal_bonus", 0.25)),
            "sft_correct_penalty": float(cfg.get("sft_correct_penalty", 1.00)),
            "skip_clipped_without_teacher": bool(cfg.get("skip_clipped_without_teacher", False)),
            "mode_stable_enabled": bool(cfg.get("mode_stable_enabled", False)),
            "mode_stable_ema_beta": float(cfg.get("mode_stable_ema_beta", 0.80)),
            "mode_stable_switch_margin": float(cfg.get("mode_stable_switch_margin", 0.20)),
            "mode_stable_min_hold_steps": int(cfg.get("mode_stable_min_hold_steps", 2)),
        }

    def _signal_utility_state_keys(self, inputs: list[Any], completion_count: int) -> list[str]:
        keys: list[str] = []
        raw_count = len(inputs) if isinstance(inputs, list) else 0
        expanded_count = max(int(completion_count), 0)
        for row in range(expanded_count):
            source_idx = self._source_row_index(row, raw_count, expanded_count)
            sample = inputs[source_idx] if raw_count and source_idx < raw_count else {}
            sample_id = None
            if isinstance(sample, dict):
                for field in ("source_idx", "idx", "id", "question_id", "qid"):
                    value = sample.get(field)
                    if value is not None and str(value).strip():
                        sample_id = str(value).strip()
                        break
                if sample_id is None:
                    image = str(sample.get("image", source_idx)).strip()
                    question = str(sample.get("question", "")).strip().replace("\n", " ")
                    sample_id = f"{image}|{question[:120]}"
            if sample_id is None:
                sample_id = str(source_idx)
            keys.append(f"{sample_id}::slot={row % max(self.num_generations, 1)}")
        return keys

    def _effective_group_filter_config(self) -> EffectiveGroupFilterConfig:
        return EffectiveGroupFilterConfig.from_mapping(self.opsd_config.get("effective_group_filter"))

    def _tokenize_teacher_sft_repair_target(
        self,
        text: str,
        *,
        sample: dict[str, Any] | None = None,
        reference_answer: Any = "",
        device: torch.device,
        max_tokens: int,
        sanitize_privileged: bool,
        target_constraint: str = "chartqa_hint",
        target_style: str = "chartqa_hint",
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, bool]]:
        if target_style:
            constrained = build_teacher_sft_repair_target(
                text,
                sample=sample,
                reference_answer=reference_answer,
                target_style=target_style,
                sanitize_privileged=sanitize_privileged,
            )
            target = constrained.text
            audit = {
                "raw_full_hint_format": constrained.raw_full_hint_format,
                "full_hint_format": constrained.full_hint_format,
                "exact_reference_answer_line": constrained.exact_reference_answer_line,
                "used_fallback_hint": constrained.used_fallback_hint,
                "raw_clipped": constrained.raw_clipped,
                "student_short_format": constrained.student_short_format,
                "answer_only_format": constrained.answer_only_format,
            }
        elif target_constraint == "chartqa_hint":
            constrained = build_teacher_sft_repair_target(
                text,
                sample=sample,
                reference_answer=reference_answer,
                target_style="chartqa_hint",
                sanitize_privileged=sanitize_privileged,
            )
            target = constrained.text
            audit = {
                "raw_full_hint_format": constrained.raw_full_hint_format,
                "full_hint_format": constrained.full_hint_format,
                "exact_reference_answer_line": constrained.exact_reference_answer_line,
                "used_fallback_hint": constrained.used_fallback_hint,
                "raw_clipped": constrained.raw_clipped,
                "student_short_format": constrained.student_short_format,
                "answer_only_format": constrained.answer_only_format,
            }
        else:
            target = sanitize_teacher_sft_text(text) if sanitize_privileged else (text or "").strip()
            audit = {
                "raw_full_hint_format": False,
                "full_hint_format": False,
                "exact_reference_answer_line": False,
                "used_fallback_hint": False,
                "raw_clipped": False,
                "student_short_format": False,
                "answer_only_format": False,
            }
        privileged_tag_present = any(
            tag in (target or "")
            for tag in ("[Verified Hint]", "[Reference Answer]", "[DePlot]", "[Visual Facts")
        )
        if self.end_flag and not target.endswith(self.end_flag):
            target = f"{target}{self.end_flag}"
        encoded = self.processing_class.tokenizer(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
        ids = encoded["input_ids"][0].to(device)
        mask = encoded["attention_mask"][0].to(device)
        return ids, mask, privileged_tag_present, audit

    def _teacher_sft_repair_prompt_key(self, sample: dict[str, Any]) -> str:
        question = sample.get("question") or sample.get("query") or sample.get("prompt") or ""
        image = sample.get("image") or sample.get("image_path") or sample.get("image_id") or ""
        return f"{str(image)}\n{str(question)}"

    def _teacher_new_ids_with_prefix(
        self,
        generated: torch.Tensor,
        prompt_len: int,
        prefix_lens: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        max_len = 0
        for row in range(int(generated.shape[0])):
            prefix_len = int(prefix_lens[row]) if row < len(prefix_lens) else 0
            start = max(0, prompt_len - prefix_len)
            row_ids = generated[row, start:]
            rows.append(row_ids)
            row_len = int(row_ids.numel())
            lengths.append(row_len)
            max_len = max(max_len, row_len)
        if not rows:
            return generated.new_empty((0, 0)), []
        pad_id = int(self.processing_class.tokenizer.pad_token_id)
        padded: list[torch.Tensor] = []
        for row_ids in rows:
            if int(row_ids.numel()) < max_len:
                pad = torch.full(
                    (max_len - int(row_ids.numel()),),
                    pad_id,
                    device=row_ids.device,
                    dtype=row_ids.dtype,
                )
                row_ids = torch.cat([row_ids, pad], dim=0)
            padded.append(row_ids)
        return torch.stack(padded, dim=0), lengths

    def _teacher_generated_mask(
        self,
        new_ids: torch.Tensor,
        valid_lengths: list[int],
    ) -> torch.Tensor:
        if new_ids.numel() == 0:
            return torch.zeros_like(new_ids, dtype=torch.int)
        seq_idx = torch.arange(new_ids.size(1), device=new_ids.device).expand(new_ids.size(0), -1)
        valid = torch.zeros_like(new_ids, dtype=torch.bool)
        for row, length in enumerate(valid_lengths):
            valid[row] = seq_idx[row] < int(length)
        eos_id = self.processing_class.tokenizer.eos_token_id
        if eos_id is None:
            return valid.int()
        is_eos = (new_ids == eos_id) & valid
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=new_ids.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        return ((seq_idx <= eos_idx.unsqueeze(1)) & valid).int()

    def _teacher_generate_from_tensors(
        self,
        teacher_tensors: dict[str, Any],
        row: int,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        prompt_ids = teacher_tensors["teacher_prompt_ids"][row : row + 1]
        prompt_mask = teacher_tensors["teacher_prompt_mask"][row : row + 1]
        t_pixel, t_sizes = get_teacher_vision_for_sample(
            teacher_tensors,
            row,
            teacher_tensors.get("teacher_num_images", None),
        )
        n_img = 1
        if isinstance(teacher_tensors.get("teacher_num_images"), torch.Tensor):
            n_img = int(max(1, teacher_tensors["teacher_num_images"][row].item()))
        teacher_batch_num_images = as_batch_num_images_tensor(n_img, t_pixel)
        prompt_ids, prompt_mask = align_teacher_prompt_image_tokens(
            self.teacher_model,
            self.processing_class,
            prompt_ids,
            prompt_mask,
            t_pixel,
            t_sizes,
            batch_num_images=teacher_batch_num_images,
        )
        teacher_device = model_inference_device(self.teacher_model)
        prompt_ids = prompt_ids.to(teacher_device)
        prompt_mask = prompt_mask.to(teacher_device)
        t_pixel = move_pixel_values_to_model_device(self.teacher_model, t_pixel)
        teacher_batch_num_images = move_batch_num_images_to_model_device(
            self.teacher_model, teacher_batch_num_images
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processing_class.tokenizer.pad_token_id,
            "eos_token_id": self.processing_class.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            generated = self.teacher_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                pixel_values=t_pixel,
                image_sizes=t_sizes,
                batch_num_images=teacher_batch_num_images,
                **gen_kwargs,
            )
        prefix_len = 0
        prefix_lens_tensor = teacher_tensors.get("teacher_response_prefix_lens")
        if isinstance(prefix_lens_tensor, torch.Tensor) and row < int(prefix_lens_tensor.numel()):
            prefix_len = int(prefix_lens_tensor[row].item())
        new_ids, valid_lengths = self._teacher_new_ids_with_prefix(
            generated,
            int(prompt_ids.shape[1]),
            [prefix_len],
        )
        new_ids = new_ids.to(self.accelerator.device)
        valid_lengths = [int(v) for v in valid_lengths]
        if new_ids.numel() == 0:
            new_ids = torch.tensor(
                [[self.processing_class.tokenizer.eos_token_id]],
                device=self.accelerator.device,
                dtype=torch.long,
            )
            valid_lengths = [1]
        mask = self._teacher_generated_mask(new_ids, valid_lengths)
        text = self.processing_class.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return new_ids[0], mask[0], text

    def _teacher_generate_batch_from_tensors(
        self,
        teacher_tensors: dict[str, Any],
        rows: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor, str]], bool]:
        if len(rows) <= 1:
            return [
                self._teacher_generate_from_tensors(
                    teacher_tensors,
                    rows[0],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            ], False

        def _fallback(reason: str) -> tuple[list[tuple[torch.Tensor, torch.Tensor, str]], bool]:
            opsd_debug.log(
                "teacher_probe",
                "batched teacher generate fallback to per-row",
                rows=rows,
                reason=reason,
            )
            outputs = [
                self._teacher_generate_from_tensors(
                    teacher_tensors,
                    row,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                for row in rows
            ]
            return outputs, True

        try:
            prompt_ids = teacher_tensors["teacher_prompt_ids"][rows]
            prompt_mask = teacher_tensors["teacher_prompt_mask"][rows]
            t_pixel, t_sizes, teacher_batch_num_images = stack_teacher_vision_for_generate(
                teacher_tensors,
                rows,
            )
            prompt_ids, prompt_mask = align_teacher_prompt_image_tokens(
                self.teacher_model,
                self.processing_class,
                prompt_ids,
                prompt_mask,
                t_pixel,
                t_sizes,
                batch_num_images=teacher_batch_num_images,
            )
        except ValueError as exc:
            return _fallback(str(exc))

        teacher_device = model_inference_device(self.teacher_model)
        prompt_ids = prompt_ids.to(teacher_device)
        prompt_mask = prompt_mask.to(teacher_device)
        t_pixel = move_pixel_values_to_model_device(self.teacher_model, t_pixel)
        teacher_batch_num_images = move_batch_num_images_to_model_device(
            self.teacher_model, teacher_batch_num_images
        )
        gen_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processing_class.tokenizer.pad_token_id,
            "eos_token_id": self.processing_class.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if t_pixel is not None:
            gen_kwargs["pixel_values"] = t_pixel
            gen_kwargs["image_sizes"] = t_sizes
            gen_kwargs["batch_num_images"] = teacher_batch_num_images
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
        try:
            with torch.no_grad():
                generated = self.teacher_model.generate(**gen_kwargs)
        except RuntimeError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _fallback(f"generate runtime error: {exc}")

        prefix_lens_tensor = teacher_tensors.get("teacher_response_prefix_lens")
        if isinstance(prefix_lens_tensor, torch.Tensor):
            prefix_lens = [int(prefix_lens_tensor[row].item()) for row in rows]
        else:
            prefix_lens = [0 for _ in rows]
        new_ids, valid_lengths = self._teacher_new_ids_with_prefix(
            generated,
            int(prompt_ids.shape[1]),
            prefix_lens,
        )
        new_ids = new_ids.to(self.accelerator.device)
        if new_ids.numel() == 0:
            new_ids = torch.full(
                (len(rows), 1),
                int(self.processing_class.tokenizer.eos_token_id),
                device=self.accelerator.device,
                dtype=torch.long,
            )
            valid_lengths = [1 for _ in rows]
        mask = self._teacher_generated_mask(new_ids, valid_lengths)
        texts = [
            text.strip()
            for text in self.processing_class.batch_decode(new_ids, skip_special_tokens=True)
        ]
        return [
            (new_ids[i], mask[i], texts[i])
            for i in range(len(rows))
        ], False

    def _apply_teacher_probe_routing(
        self,
        *,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        completion_modes: list[int],
        acc_rewards: torch.Tensor,
        answers: list[str],
        completions: list[str],
        answer_flag: str,
        global_step: int,
        device,
        group_has_correct: list[bool] | None = None,
        group_reward_std: list[float] | None = None,
    ) -> tuple[
        list[int],
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        dict[int, str],
        dict[str, Any],
    ]:
        stats: dict[str, Any] = {
            "teacher_probe_candidates": 0,
            "teacher_probe_correct": 0,
            "teacher_probe_wrong": 0,
            "teacher_probe_skipped_budget": 0,
            "teacher_probe_skipped_no_evidence": 0,
            "teacher_probe_evidence_present": 0,
            "teacher_probe_deplot_placeholder": 0,
            "teacher_probe_deplot_real": 0,
            "teacher_probe_visual_fact_used": 0,
            "teacher_probe_answer_flag": 0,
            "teacher_probe_parse_failed": 0,
            "teacher_probe_gold_suffix": 0,
            "teacher_probe_generated_tokens_mean": 0.0,
            "teacher_probe_generated_tokens_p95": 0.0,
            "teacher_probe_clipped_rate": 0.0,
            "teacher_probe_batch_size": 1,
            "teacher_probe_generate_s": 0.0,
            "teacher_probe_generate_batches": 0,
            "teacher_probe_fallback_batches": 0,
            "teacher_probe_text": {},
        }
        probe_mode = self.opsd_config.get("mode") == "dyme_teacher_probe_opd"
        opd_only_probe = self.training_stage == "opd_only"
        if (not probe_mode and not opd_only_probe) or self.teacher_model is None:
            return completion_modes, {}, {}, stats

        probe_cfg = self._teacher_probe_config()
        if not probe_cfg["enabled"]:
            return completion_modes, {}, {}, stats

        candidate_indices = [i for i, mode_i in enumerate(completion_modes) if mode_i == MODE_OPSD]
        stats["teacher_probe_candidates"] = len(candidate_indices)
        provider_names = probe_cfg["context_providers"]
        expanded_count = len(completion_modes)

        def source_idx_for(row: int) -> int:
            return self._source_row_index(row, len(inputs), expanded_count)

        def sample_for(row: int) -> dict[str, Union[torch.Tensor, Any]]:
            return inputs[source_idx_for(row)] if inputs else {}

        def reference_for(row: int) -> str:
            idx = source_idx_for(row)
            return str(answers[idx]) if idx < len(answers) else ""

        def group_metadata_for(row: int) -> tuple[bool | None, float | None, bool, bool, str]:
            prompt_idx = row // self.num_generations
            has_correct_i = (
                bool(group_has_correct[prompt_idx])
                if group_has_correct is not None and prompt_idx < len(group_has_correct)
                else None
            )
            reward_std_i = (
                float(group_reward_std[prompt_idx])
                if group_reward_std is not None and prompt_idx < len(group_reward_std)
                else None
            )
            is_all_wrong = has_correct_i is False
            is_mixed_wrong = has_correct_i is True
            route_reason = "all_wrong_teacher_rescue" if is_all_wrong else "mixed_wrong_teacher_probe"
            return has_correct_i, reward_std_i, is_all_wrong, is_mixed_wrong, route_reason

        eligible_indices: list[int] = []
        for global_idx in candidate_indices:
            sample = sample_for(global_idx)
            evidence_status = teacher_probe_evidence_status(sample, provider_names)
            if evidence_status["evidence_present"]:
                stats["teacher_probe_evidence_present"] += 1
            if evidence_status["deplot_placeholder"]:
                stats["teacher_probe_deplot_placeholder"] += 1
            if evidence_status["deplot_real"]:
                stats["teacher_probe_deplot_real"] += 1
            if evidence_status["visual_fact_used"]:
                stats["teacher_probe_visual_fact_used"] += 1

            if probe_cfg["skip_no_evidence"] and not evidence_status["evidence_present"] and not opd_only_probe:
                completion_modes[global_idx] = MODE_SFT
                stats["teacher_probe_skipped_no_evidence"] += 1
                prompt_idx = global_idx // self.num_generations
                generation_idx = global_idx % self.num_generations
                source_idx = source_idx_for(global_idx)
                reference = reference_for(global_idx)
                student_output = completions[global_idx] if global_idx < len(completions) else ""
                (
                    group_has_correct_i,
                    group_reward_std_i,
                    is_all_wrong_probe_candidate,
                    is_mixed_wrong_probe_candidate,
                    route_reason,
                ) = group_metadata_for(global_idx)
                append_teacher_probe_record(
                    output_dir=getattr(self.args, "output_dir", None),
                    opsd_config=self.opsd_config,
                    rank=self.accelerator.process_index,
                    record=build_teacher_probe_record(
                        sample=sample,
                        global_step=global_step,
                        rank=self.accelerator.process_index,
                        global_idx=global_idx,
                        source_idx=source_idx,
                        prompt_idx=prompt_idx,
                        generation_idx=generation_idx,
                        provider_names=provider_names,
                        reference=str(reference),
                        student_output=student_output,
                        teacher_output="",
                        score=0.0,
                        final_route="sft_no_evidence",
                        answer_flag=answer_flag,
                        evidence_status=evidence_status,
                        group_has_correct=group_has_correct_i,
                        group_reward_std=group_reward_std_i,
                        is_all_wrong_probe_candidate=is_all_wrong_probe_candidate,
                        is_mixed_wrong_probe_candidate=is_mixed_wrong_probe_candidate,
                        route_reason=route_reason,
                    ),
                )
            else:
                eligible_indices.append(global_idx)

        candidate_indices = eligible_indices
        max_per_batch = probe_cfg["max_per_batch"]
        if max_per_batch > 0 and len(candidate_indices) > max_per_batch:
            skipped = candidate_indices[max_per_batch:]
            for idx in skipped:
                completion_modes[idx] = MODE_SFT
            stats["teacher_probe_skipped_budget"] = len(skipped)
            candidate_indices = candidate_indices[:max_per_batch]
        if not candidate_indices:
            return completion_modes, {}, {}, stats

        teacher_tensors = build_teacher_prompt_batch(
            self.processing_class,
            inputs,
            candidate_indices,
            provider_names,
            device,
            opsd_config=self.opsd_config,
            global_step=global_step,
            output_dir=self.args.output_dir,
            expanded_count=expanded_count,
            num_generations=self.num_generations,
        )
        teacher_stats = teacher_tensors.get("teacher_stats", {}) if teacher_tensors else {}
        gold_rate = float(teacher_stats.get("privileged_suffix_has_gold_rate", 0.0) or 0.0)
        stats["teacher_probe_gold_suffix"] = int(round(gold_rate * len(candidate_indices)))
        teacher_trajs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        teacher_traj_texts: dict[int, str] = {}
        generated_token_counts: list[int] = []
        generated_clipped_count = 0
        probe_batch_size = max(1, int(probe_cfg.get("batch_size", 1) or 1))
        stats["teacher_probe_batch_size"] = probe_batch_size
        generated_rows: list[tuple[int, int, torch.Tensor, torch.Tensor, str]] = []
        for start in range(0, len(candidate_indices), probe_batch_size):
            end = min(start + probe_batch_size, len(candidate_indices))
            rows = list(range(start, end))
            gen_start = self._perf_start()
            batch_outputs, fallback_used = self._teacher_generate_batch_from_tensors(
                teacher_tensors,
                rows,
                max_new_tokens=probe_cfg["max_new_tokens"],
                do_sample=probe_cfg["do_sample"],
                temperature=probe_cfg["temperature"],
                top_p=probe_cfg["top_p"],
                repetition_penalty=probe_cfg["repetition_penalty"],
            )
            stats["teacher_probe_generate_s"] += self._perf_elapsed(gen_start)
            stats["teacher_probe_generate_batches"] += 1
            if fallback_used:
                stats["teacher_probe_fallback_batches"] += 1
            for offset, (gen_ids, gen_mask, text) in enumerate(batch_outputs):
                global_idx = candidate_indices[start + offset]
                generated_rows.append((start + offset, global_idx, gen_ids, gen_mask, text))

        for row, global_idx, gen_ids, gen_mask, text in generated_rows:
            effective_tokens = int(gen_mask.sum().item()) if hasattr(gen_mask, "sum") else int(len(gen_ids))
            generated_token_counts.append(effective_tokens)
            eos_id = self.processing_class.tokenizer.eos_token_id
            has_eos = bool((gen_ids == eos_id).any().item()) if eos_id is not None and hasattr(gen_ids, "numel") else False
            if not has_eos and int(gen_ids.numel()) >= int(probe_cfg["max_new_tokens"]):
                generated_clipped_count += 1
            prompt_idx = global_idx // self.num_generations
            generation_idx = global_idx % self.num_generations
            source_idx = source_idx_for(global_idx)
            reference = reference_for(global_idx)
            if probe_cfg["answer_parser"] == "chartqa_final_answer":
                score, parsed_answer = eval_teacher_probe_chart(
                    text,
                    str(reference),
                    probe_cfg["max_relative_change"],
                    answer_flag=answer_flag.lower(),
                )
            else:
                score = float(
                    eval_one_chart(
                        text,
                        str(reference).lower().replace(answer_flag.lower(), "").strip(),
                        probe_cfg["max_relative_change"],
                        answer_flag=answer_flag.lower(),
                    )
                )
                parsed_answer = None
            if parsed_answer is not None:
                if parsed_answer.has_answer_flag:
                    stats["teacher_probe_answer_flag"] += 1
                if parsed_answer.parse_failed:
                    stats["teacher_probe_parse_failed"] += 1
            stats["teacher_probe_text"][global_idx] = text[:160].replace("\n", "\\n")
            # In opd_only the probe is diagnostic/trajectory supervision only;
            # its answer score must never route or discard the completion.
            final_route = "opd_only_diagnostic" if opd_only_probe else ("opd" if score > 0 else "sft")
            sample = sample_for(global_idx)
            student_output = completions[global_idx] if global_idx < len(completions) else ""
            evidence_status = teacher_probe_evidence_status(sample, provider_names)
            (
                group_has_correct_i,
                group_reward_std_i,
                is_all_wrong_probe_candidate,
                is_mixed_wrong_probe_candidate,
                route_reason,
            ) = group_metadata_for(global_idx)
            probe_log_path = append_teacher_probe_record(
                output_dir=getattr(self.args, "output_dir", None),
                opsd_config=self.opsd_config,
                rank=self.accelerator.process_index,
                record=build_teacher_probe_record(
                    sample=sample,
                    global_step=global_step,
                    rank=self.accelerator.process_index,
                    global_idx=global_idx,
                    source_idx=source_idx,
                    prompt_idx=prompt_idx,
                    generation_idx=generation_idx,
                    provider_names=provider_names,
                    reference=str(reference),
                    student_output=student_output,
                    teacher_output=text,
                    score=score,
                    final_route=final_route,
                    answer_flag=answer_flag,
                    parsed_answer=parsed_answer.answer if parsed_answer is not None else "",
                    parse_failed=parsed_answer.parse_failed if parsed_answer is not None else False,
                    has_answer_flag=parsed_answer.has_answer_flag if parsed_answer is not None else ("answer:" in text.lower()),
                    evidence_status=evidence_status,
                    group_has_correct=group_has_correct_i,
                    group_reward_std=group_reward_std_i,
                    is_all_wrong_probe_candidate=is_all_wrong_probe_candidate,
                    is_mixed_wrong_probe_candidate=is_mixed_wrong_probe_candidate,
                    route_reason=route_reason,
                ),
            )
            if opd_only_probe:
                opsd_debug.log(
                    "teacher_probe",
                    "opd_only probe record persisted",
                    path=probe_log_path,
                    global_idx=global_idx,
                    score=score,
                    final_route=final_route,
                )
            if self.accelerator.is_main_process and not self._teacher_probe_preview_logged:
                self._teacher_probe_preview_logged = True
                preview_payload = {
                    "global_step": int(global_step),
                    "provider_names": list(provider_names),
                    "max_new_tokens": int(probe_cfg["max_new_tokens"]),
                    "prompt_idx": int(prompt_idx),
                    "source_idx": int(source_idx),
                    "generation_idx": int(generation_idx),
                    "reference": str(reference)[:160],
                    "teacher_output_preview": text[:240].replace("\n", "\\n"),
                    "parsed_answer": (
                        parsed_answer.answer[:160]
                        if parsed_answer is not None and parsed_answer.answer is not None
                        else ""
                    ),
                    "score": float(score),
                    "final_route": final_route,
                    "evidence_status": dict(evidence_status),
                }
                print(
                    f"[DyME-TEACHER-PROBE] {json.dumps(preview_payload, ensure_ascii=False, sort_keys=True)}",
                    flush=True,
                )
            if score > 0:
                stats["teacher_probe_correct"] += 1
                teacher_traj_texts[global_idx] = text
                if self._teacher_trajectory_config()["enabled"]:
                    teacher_trajs[global_idx] = (gen_ids.to(device), gen_mask.to(device))
            else:
                stats["teacher_probe_wrong"] += 1
                if opd_only_probe:
                    teacher_traj_texts[global_idx] = text
                    if self._teacher_trajectory_config()["enabled"]:
                        teacher_trajs[global_idx] = (gen_ids.to(device), gen_mask.to(device))
                else:
                    completion_modes[global_idx] = MODE_SFT

        if generated_token_counts:
            ordered_counts = sorted(generated_token_counts)
            p95_idx = min(len(ordered_counts) - 1, max(0, (95 * len(ordered_counts) + 99) // 100 - 1))
            stats["teacher_probe_generated_tokens_mean"] = float(
                sum(generated_token_counts) / len(generated_token_counts)
            )
            stats["teacher_probe_generated_tokens_p95"] = float(ordered_counts[p95_idx])
            stats["teacher_probe_clipped_rate"] = float(generated_clipped_count / len(generated_token_counts))

        opsd_debug.log(
            "teacher_probe",
            "teacher answer probe finished",
            **{k: v for k, v in stats.items() if k != "teacher_probe_text"},
            sample_text=next(iter(stats["teacher_probe_text"].values()), ""),
        )
        return completion_modes, teacher_trajs, teacher_traj_texts, stats

    def _log_training_phase(self, phase: str, global_step: int) -> None:
        if self._last_training_phase == phase:
            return
        self._last_training_phase = phase
        from opsd_utils.gate_policy import sft_cold_start_steps

        opsd_debug.log_probe(
            "phase",
            f"training_phase={phase}",
            global_step=global_step,
            max_steps=self._max_training_steps(),
            cold_start_steps=sft_cold_start_steps(
                self.opsd_config,
                self._max_training_steps(),
            ),
        )

    def _resolve_skip_degenerate_opsd(self) -> bool:
        from opsd_utils.gate_policy import resolve_skip_degenerate_opsd

        return resolve_skip_degenerate_opsd(
            self.opsd_config,
            self._current_global_step(),
            self._max_training_steps(),
        )

    def _sft_slots_for_step(self) -> int:
        from opsd_utils.gate_policy import sft_slots_for_step

        return sft_slots_for_step(
            self.opsd_config,
            self._current_global_step(),
            self._max_training_steps(),
        )

    def _should_force_sft_replace(
        self,
        i: int,
        completions: list[str],
        answer_flag: str,
    ) -> bool:
        if self.opsd_config.get("gate", {}).get("disable_force_sft_replace"):
            return False
        if self._in_sft_cold_start():
            return True
        text_i = completions[i] if i < len(completions) else ""
        if opsd_diagnostics._count_answer_flag(text_i, answer_flag) != 1:
            return True
        from reward_utils.format_checks import is_digit_spam_after_answer

        if is_digit_spam_after_answer(text_i, answer_flag.lower()):
            return True
        return False

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        mode = "train" if self.model.training else "eval"
        opsd_debug.set_step_label(f"prepare_inputs/{mode}/step={self._step}")
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            will_generate = self._step % generate_every == 0 or self._buffered_inputs is None
            opsd_debug.log(
                "prepare_inputs",
                "deciding whether to regenerate completions",
                mode=mode,
                trainer_step=self._step,
                generate_every=generate_every,
                will_generate=will_generate,
                buffered_inputs_is_none=self._buffered_inputs is None,
            )
            if will_generate:
                opsd_debug.log_probe(
                    "prepare_inputs",
                    "will_generate=True, calling _generate_and_score_completions",
                    trainer_step=self._step,
                    global_step=getattr(self.state, "global_step", None),
                    generate_every=generate_every,
                )
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            else:
                opsd_debug.log("prepare_inputs", "reuse buffered inputs slice", slice_index=self._step % self.args.gradient_accumulation_steps)
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            opsd_debug.log("prepare_inputs", "eval mode always regenerates completions")
            # In evaluation, there is neither gradient accumulation, nor multiple iterations
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        if opsd_debug.is_enabled() and inputs.get("opsd_mask") is not None:
            opsd_debug.log(
                "prepare_inputs",
                "prepared input batch summary",
                batch_size=inputs["prompt_ids"].shape[0],
                opsd_mask_true=int(inputs["opsd_mask"].sum().item()),
                has_teacher_prompt_ids="teacher_prompt_ids" in inputs,
            )
        return inputs

    def _source_row_index(self, row: int, raw_count: int, expanded_count: int) -> int:
        return source_row_index(
            row,
            raw_count=raw_count,
            expanded_count=expanded_count,
            num_generations=self.num_generations,
        )

    def _generate_sft_cold_start_batch(
        self,
        *,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        global_step: int,
        generate_call_index: int,
        mode: str,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Embedded offline SFT: skip generate, inject GT completions, SFT-only loss."""
        device = self.accelerator.device
        raw_count = len(inputs)
        prompts_count = prompt_ids.size(0)
        question_wo_prompts = [x["question_wo_prompt"] for x in inputs]
        hints = [x.get("hint", "") for x in inputs]
        answers = [x["answer"] for x in inputs]
        gpu_id = self.accelerator.device.index
        prompts_count = prompt_ids.size(0)
        if not self._refiner_skip_cold_start():
            self._prepare_visual_supervision_batch(
                inputs,
                global_step=global_step,
                expanded_count=prompts_count,
            )
            refine_fn = (
                refine_context_sequential
                if getattr(self.refiner, "requires_sequential", False)
                else refine_context_in_parallel
            )
            hints = refine_fn(
                self.refiner,
                question_wo_prompts,
                hints,
                answers,
                task=self.task_name,
                gpu_id=gpu_id,
            )
            self._finish_visual_supervision_batch(global_step)
        sft_gt_rows = []
        for i in range(prompts_count):
            src = self._source_row_index(i, raw_count, prompts_count)
            sample_i = inputs[src] if src < len(inputs) else {}
            sft_gt_rows.append(self._format_online_sft_target(hints[src], answers[src], sample_i))

        sft_dt = self.processing_class.tokenizer(
            sft_gt_rows,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        completion_ids = sft_dt["input_ids"].to(device)
        completion_mask = sft_dt["attention_mask"].to(device)
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        probe_stats = opsd_diagnostics.log_generate_probe(
            global_step=global_step,
            trainer_step=self._step,
            prompt_length=prompt_ids.size(1),
            prompt_completion_ids=torch.cat([prompt_ids, completion_ids], dim=1),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            is_eos=is_eos,
            eos_idx=eos_idx,
            completions=completions,
            tokenizer=self.processing_class.tokenizer,
            generation_config=self.generation_config,
            max_completion_length=self.max_completion_length,
            num_generations=self.num_generations,
            sample_count=self._opsd_probe_sample_count,
            generate_call_index=generate_call_index,
            answer_flag=getattr(self.checker, "answer_flag", "Answer:"),
            source="sft_cold_start",
        )
        self._last_generate_probe_stats = probe_stats
        self._generate_call_index += 1
        if self._health_monitor is not None:
            self._health_monitor.record_generate(global_step, probe_stats, self._last_logits_stats)

        batch_size = completion_ids.size(0)
        sft_replaced_list = [True] * batch_size
        opsd_mask_list = [False] * batch_size
        completion_advantange = completion_mask.float().clone()
        completion_advantange[:] = 1.0

        if self._health_monitor is not None:
            self._health_monitor.record_routing(
                global_step,
                {
                    "sft_replaced_ratio": 1.0,
                    "opsd_skipped_degenerate": 0,
                    "opsd_skipped_leakage": 0,
                    "opsd_on_correct_rate": 0.0,
                    "grpo_on_correct_rate": 0.0,
                    "opd_teacher_call_rate": 0.0,
                    "format_mean": 1.0,
                    "accuracy_mean": 0.0,
                },
            )

        input_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1).long()
        attention_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        logps_micro_batch = (
            self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            old_per_token_logps = None

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(
                attention_completion_mask.sum()
            ).sum().item()

        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / max(len(agg_completion_mask), 1)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        self._metrics[mode].setdefault("phase/sft_cold_start", []).append(1.0)
        self._metrics[mode]["reward"].append(0.0)
        self._metrics[mode]["reward_std"].append(0.0)
        for reward_func_name in self.reward_func_names:
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(0.0)

        result: dict[str, Union[torch.Tensor, Any]] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": completion_advantange,
            "old_per_token_logps": old_per_token_logps,
            "img_sizes": image_sizes,
            "acc_rewards": torch.zeros(batch_size, device=device),
            "sft_cold_start": True,
        }
        opsd_debug.log(
            "generate",
            "exit _generate_sft_cold_start_batch",
            batch_size=batch_size,
            global_step=global_step,
        )
        return result

    def _build_opd_only_batch(
        self,
        *,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        pixel_values: Any,
        image_sizes: Any,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completions: list[str],
        global_step: int,
        mode: str,
        teacher_trajs: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        teacher_traj_texts: dict[int, str] | None = None,
        teacher_probe_stats: dict[str, Any] | None = None,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Build a rollout batch whose only optimization route is OPD."""
        device = prompt_ids.device
        batch_size = int(completion_ids.size(0))
        opsd_indices = list(range(batch_size))
        # Build the teacher prompt for every completion.  Auxiliary teacher
        # components (probe/checker/refiner) are intentionally run separately
        # for diagnostics, but never participate in routing or sample choice.
        teacher_tensors = build_teacher_prompt_batch(
            self.processing_class,
            inputs,
            opsd_indices,
            self.opsd_config.get("privileged_providers", ["text"]),
            device,
            opsd_config=self.opsd_config,
            global_step=global_step,
            output_dir=self.args.output_dir,
            expanded_count=batch_size,
            num_generations=self.num_generations,
        )
        teacher_tensors = expand_teacher_tensors_to_full_batch(
            teacher_tensors,
            opsd_indices,
            batch_size,
        )
        result: dict[str, Union[torch.Tensor, Any]] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": torch.zeros_like(completion_mask, dtype=torch.float32, device=device),
            "old_per_token_logps": None,
            "img_sizes": image_sizes,
            "acc_rewards": torch.zeros(batch_size, dtype=torch.float32, device=device),
            "opsd_mask": torch.ones(batch_size, dtype=torch.bool, device=device),
            "group_mixed_rate": 0.0,
        }
        result.update(teacher_tensors)
        # Keep the OPD-only invariant explicit even when auxiliary diagnostics
        # are enabled: all completions remain OPD and no reward-derived signal
        # can affect the objective.
        teacher_trajs = teacher_trajs or {}
        traj_ids: list[torch.Tensor] = []
        traj_masks: list[torch.Tensor] = []
        traj_mask_values: list[bool] = []
        for row in range(batch_size):
            pair = teacher_trajs.get(row)
            if pair is None:
                traj_ids.append(completion_ids[row, :0])
                traj_masks.append(completion_mask[row, :0])
                traj_mask_values.append(False)
            else:
                ids_i, mask_i = pair
                traj_ids.append(ids_i.to(device))
                traj_masks.append(mask_i.to(device))
                traj_mask_values.append(bool(mask_i.numel() and mask_i.sum().item() > 0))
        result["teacher_traj_mask"] = torch.tensor(traj_mask_values, dtype=torch.bool, device=device)
        result["teacher_traj_completion_ids"] = pad_sequence(
            traj_ids, batch_first=True, padding_value=self.processing_class.tokenizer.pad_token_id
        ).long().to(device)
        result["teacher_traj_completion_mask"] = pad_sequence(
            traj_masks, batch_first=True, padding_value=0
        ).to(device)
        result["teacher_probe_stats"] = dict(teacher_probe_stats or {})
        result["teacher_traj_texts"] = dict(teacher_traj_texts or {})
        self._metrics[mode].setdefault("routing/opd_only", []).append(1.0)
        self._metrics[mode].setdefault("routing/opd_route_count", []).append(float(batch_size))
        self._metrics[mode].setdefault("routing/grpo_route_count", []).append(0.0)
        self._metrics[mode].setdefault("routing/sft_route_count", []).append(0.0)
        opsd_debug.log(
            "generate",
            "built isolated OPD-only rollout batch",
            global_step=global_step,
            batch_size=batch_size,
            completion_tokens=int(completion_mask.sum().item()),
            teacher_prompt_rows=int(teacher_tensors["teacher_prompt_ids"].size(0)),
            teacher_traj_rows=int(sum(traj_mask_values)),
            teacher_probe_enabled=bool(teacher_probe_stats),
        )
        return result

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        # TODO
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        if self._perf_timing_enabled:
            self._perf_step_start_s = self._perf_start()
        opsd_debug.set_step_label(f"generate_and_score/{mode}")
        opsd_debug.log(
            "generate",
            "enter _generate_and_score_completions",
            mode=mode,
            local_batch_size=len(inputs),
            opsd_enabled=self.opsd_config.get("enabled", False),
            opsd_mode=self.opsd_config.get("mode", "dyme"),
            privileged_providers=self.opsd_config.get("privileged_providers", []),
            num_generations=self.num_generations,
            global_step=getattr(self.state, "global_step", None),
        )

        inputs_for_generate = inputs.copy()

        # 去除answer key
        inputs_for_generate = [{k: v for k, v in x.items() if k != 'answer'} for x in inputs_for_generate]

        dt_generate_dt = self.processing_func(inputs_for_generate)
        prompt_inputs_generate = super(DyMETrainer, self)._prepare_inputs(dt_generate_dt)
        if 'labels' in prompt_inputs_generate:
            del prompt_inputs_generate["labels"]
        prompt_ids = prompt_inputs_generate["input_ids"]
        prompt_mask = prompt_inputs_generate["attention_mask"]
        pixel_values = prompt_inputs_generate["pixel_values"]
        image_sizes = prompt_inputs_generate["image_sizes"]

        global_step_for_probe = getattr(self.state, "global_step", self._step)
        generate_call_index = self._generate_call_index
        in_cold_start = (
            False if self.training_stage == "opd_only" else self._in_sft_cold_start()
        )
        self._log_training_phase(
            "sft_cold_start" if in_cold_start else self.training_stage,
            global_step_for_probe,
        )
        if not in_cold_start:
            self._ensure_teacher_model()

        if self._health_monitor is not None:
            self._health_monitor.reset_step(global_step_for_probe)
            data_health = opsd_diagnostics.summarize_batch_data_health(
                inputs,
                prompt_mask=prompt_mask,
                pixel_values=pixel_values,
            )
            self._health_monitor.record_data(global_step_for_probe, data_health)

        if in_cold_start:
            return self._generate_sft_cold_start_batch(
                inputs=inputs,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                global_step=global_step_for_probe,
                generate_call_index=generate_call_index,
                mode=mode,
            )

        # Regular generation path
        student_generate_start = self._perf_start()
        with opsd_debug.timed("generate", "model.generate"):
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    opsd_diagnostics.log_generate_context(
                        global_step=global_step_for_probe,
                        trainer_step=self._step,
                        generate_call_index=generate_call_index,
                        model=unwrapped_model,
                        model_wrapped=self.model_wrapped,
                        gradient_checkpointing=bool(self.args.gradient_checkpointing),
                        generation_config=self.generation_config,
                        is_fsdp_enabled=self.is_fsdp_enabled,
                        generate_runs_under_no_grad=False,
                    )
                    opsd_diagnostics.log_prompt_tail_probe(
                        global_step=global_step_for_probe,
                        trainer_step=self._step,
                        generate_call_index=generate_call_index,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        tokenizer=self.processing_class.tokenizer,
                        sample_count=self._opsd_probe_sample_count,
                    )
                    logits_probe_result = opsd_diagnostics.log_first_token_logits_probe(
                        global_step=global_step_for_probe,
                        trainer_step=self._step,
                        generate_call_index=generate_call_index,
                        unwrapped_model=unwrapped_model,
                        prompt_inputs_generate=prompt_inputs_generate,
                        prompt_mask=prompt_mask,
                        tokenizer=self.processing_class.tokenizer,
                        sample_count=self._opsd_probe_sample_count,
                    )
                    greedy_by_sample = logits_probe_result.get("greedy_by_sample", {})
                    self._last_logits_stats = {
                        k: logits_probe_result[k]
                        for k in ("p_greedy_first", "p_eos_first", "entropy_first")
                        if k in logits_probe_result
                    }
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs_generate, generation_config=self.generation_config
                    )
                    opsd_diagnostics.log_first_token_logits_match(
                        generate_call_index=generate_call_index,
                        completion_ids=prompt_completion_ids[:, prompt_ids.size(1) :],
                        greedy_by_sample=greedy_by_sample,
                        sample_count=self._opsd_probe_sample_count,
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        self._perf_metric(mode, "student_generate_s", self._perf_elapsed(student_generate_start))

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if self.training_stage == "opd_only":
            # Auxiliary teacher components are executed for observability and
            # optional trajectory supervision only.  Their rewards/refined
            # text are deliberately discarded; every completion stays OPD.
            aux_probe_modes = [MODE_OPSD] * int(completion_ids.size(0))
            aux_answers = [str(x.get("answer", "")) for x in inputs]
            aux_prompts = [str(x.get("prompt", "")) for x in inputs]
            aux_hints = [str(x.get("hint", "")) for x in inputs]
            aux_images = [x.get("image") for x in inputs]
            aux_image_paths = [
                image if isinstance(image, str) else getattr(image, "filename", "")
                for image in aux_images
            ]
            aux_batch = {
                "prompt": aux_prompts,
                "hints": aux_hints,
                "image": aux_image_paths,
                "response": completions,
                "answer": aux_answers,
            }
            aux_step = int(global_step_for_probe)
            self._prepare_visual_supervision_batch(
                inputs, global_step=aux_step, expanded_count=int(completion_ids.size(0))
            )
            try:
                aux_rewards = ([], [], torch.zeros((len(inputs), self.num_generations), device=device), [])
                if self.checker is not None:
                    reward_fn = (
                        calculate_rewards_sequential
                        if getattr(self.checker, "requires_sequential", False)
                        else calculate_rewards_in_parallel
                    )
                    aux_rewards = reward_fn(
                        self.checker, aux_batch, gpu_id=self.accelerator.device.index or 0,
                        task=self.task_name,
                    )
                    opsd_debug.log(
                        "opd_only_aux",
                        "visual checker/refiner diagnostics complete; rewards discarded",
                        checker=type(self.checker).__name__,
                        reward_count=len(aux_rewards[0]),
                        reward_preview=aux_rewards[0][:3],
                    )
                if self.refiner is not None:
                    refined = refine_context_sequential(
                        self.refiner,
                        [str(x.get("question_wo_prompt", x.get("prompt", ""))) for x in inputs],
                        aux_hints,
                        aux_answers,
                        self.task_name,
                        self.accelerator.device.index or 0,
                    )
                    opsd_debug.log(
                        "opd_only_aux",
                        "visual refiner diagnostics complete; outputs discarded",
                        refiner=type(self.refiner).__name__,
                        refined_preview=[str(x)[:240] for x in refined[:2]],
                    )
            finally:
                self._finish_visual_supervision_batch(aux_step)

            probe_stats: dict[str, Any] = {}
            teacher_trajs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            teacher_traj_texts: dict[int, str] = {}
            if self.teacher_model is not None and bool((self.opsd_config.get("teacher_probe") or {}).get("enabled", False)):
                probe_modes, teacher_trajs, teacher_traj_texts, probe_stats = self._apply_teacher_probe_routing(
                    inputs=inputs,
                    completion_modes=aux_probe_modes,
                    acc_rewards=torch.zeros((len(inputs), self.num_generations), device=device),
                    answers=aux_answers,
                    completions=completions,
                    answer_flag=getattr(self.checker, "answer_flag", "Answer:"),
                    global_step=aux_step,
                    device=device,
                    group_has_correct=[False] * len(inputs),
                    group_reward_std=[0.0] * len(inputs),
                )
                opsd_debug.log(
                    "opd_only_aux",
                    "teacher probe diagnostics complete; route unchanged",
                    candidates=probe_stats.get("teacher_probe_candidates", 0),
                    correct=probe_stats.get("teacher_probe_correct", 0),
                    wrong=probe_stats.get("teacher_probe_wrong", 0),
                    trajectory_rows=len(teacher_trajs),
                )

            return self._build_opd_only_batch(
                inputs=inputs,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completions=completions,
                global_step=global_step_for_probe,
                mode=mode,
                teacher_trajs=teacher_trajs,
                teacher_traj_texts=teacher_traj_texts,
                teacher_probe_stats=probe_stats,
            )

        probe_stats = opsd_diagnostics.log_generate_probe(
            global_step=global_step_for_probe,
            trainer_step=self._step,
            prompt_length=prompt_length,
            prompt_completion_ids=prompt_completion_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            is_eos=is_eos,
            eos_idx=eos_idx,
            completions=completions,
            tokenizer=self.processing_class.tokenizer,
            generation_config=self.generation_config,
            max_completion_length=self.max_completion_length,
            num_generations=self.num_generations,
            sample_count=self._opsd_probe_sample_count,
            generate_call_index=generate_call_index,
            answer_flag=getattr(self.checker, "answer_flag", "Answer:"),
        )
        opsd_diagnostics.log_generate_delta(
            generate_call_index=generate_call_index,
            current_stats=probe_stats,
            previous_stats=self._last_generate_probe_stats,
        )
        opsd_diagnostics.log_cross_rank_generate_summary(
            accelerator=self.accelerator,
            one_token_count=probe_stats.get("one_token_count", 0),
            effective_tokens_mean=probe_stats.get("effective_tokens_mean", 0.0),
            generate_call_index=generate_call_index,
        )
        self._last_generate_probe_stats = probe_stats
        self._generate_call_index += 1

        if self._health_monitor is not None:
            self._health_monitor.record_generate(
                global_step_for_probe,
                probe_stats,
                self._last_logits_stats,
            )

        batch_size = len(completion_ids)
        images = [x['image'] for x in inputs]
        prompts = [x['prompt'] for x in inputs]
        question_wo_prompts = [x['question_wo_prompt'] for x in inputs]
        hints = [x.get('hint', '') for x in inputs]
        answers = [x['answer'] for x in inputs]
        images_path = [image if isinstance(image, str) else image.filename for image in images]
        batch_data = {'prompt': prompts, 'hints': hints,
                   'image': images_path, 'response': completions, 'answer': answers}
        if 'world' in self.task_name:
            batch_data['direct_answers'] = [x.get('direct_answers', '') for x in inputs]

        gpu_id = self.accelerator.device.index
        opsd_debug.log(
            "reward",
            "start reward calculation",
            gpu_id=gpu_id,
            batch_size=batch_size,
            sample_prompt=(prompts[0][:120] + "...") if prompts and len(prompts[0]) > 120 else (prompts[0] if prompts else None),
        )
        reward_global_step = getattr(self.state, "global_step", self._step)
        opsd_debug.set_detail_step(reward_global_step)
        self._prepare_visual_supervision_batch(
            inputs,
            global_step=reward_global_step,
            expanded_count=batch_size,
        )
        reward_fn = (
            calculate_rewards_sequential
            if getattr(self.checker, "requires_sequential", False)
            else calculate_rewards_in_parallel
        )
        reward_start = self._perf_start()
        with opsd_debug.timed("reward", "calculate_rewards_in_parallel"):
            all_rewards, format_rewards, acc_rewards, context_rewards = reward_fn(
                self.checker,
                batch_data,
                gpu_id=gpu_id,
                task=self.task_name,
            )
        self._perf_metric(mode, "reward_s", self._perf_elapsed(reward_start))
        self._opsd_distributed_barrier("wait_for_everyone after visual checker rewards")
        opsd_debug.log(
            "reward",
            "reward calculation finished",
            format_rewards_sum=sum(format_rewards),
            acc_rewards_sum=sum(acc_rewards),
            context_rewards_sum=sum(context_rewards),
            all_rewards_sum=sum(all_rewards),
        )
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(self.accelerator.device)
        format_rewards = torch.tensor(format_rewards, dtype=torch.float32).to(self.accelerator.device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float32).to(self.accelerator.device)
        acc_rewards = torch.tensor(acc_rewards, dtype=torch.float32).to(self.accelerator.device)
        perception_cfg = self.opsd_config.get("perception_reward", {}) or {}
        perception_enabled = bool(perception_cfg.get("enabled", False))
        perception_weight = float(perception_cfg.get("weight", 0.2) or 0.0)
        perception_stats: dict[str, float] = {
            "mean": 0.0,
            "skipped_rate": 1.0 if perception_enabled else 0.0,
            "judge_parse_fail_rate": 0.0,
            "diagnostic_deplot_overlap_mean": 0.0,
        }
        perception_rewards = torch.zeros_like(all_rewards)
        if perception_enabled:
            perception_source = str(perception_cfg.get("source", "image_teacher"))
            if perception_source == "image_teacher":
                perception_result = score_image_teacher_perception_rewards(
                    samples=inputs,
                    responses=completions,
                    teacher_model=self.teacher_model,
                    processor=self.processing_class,
                    batch_size=int(perception_cfg.get("batch_size", 4) or 4),
                    max_new_tokens=int(perception_cfg.get("max_new_tokens", 8) or 8),
                )
            else:
                perception_result = score_perception_rewards(
                    samples=inputs,
                    responses=completions,
                    source=perception_source,
                )
            perception_stats = perception_result.stats
            perception_values = list(perception_result.rewards)
            if len(perception_values) < len(all_rewards):
                perception_values.extend([0.0] * (len(all_rewards) - len(perception_values)))
            perception_rewards = torch.tensor(
                perception_values[: len(all_rewards)],
                dtype=torch.float32,
                device=self.accelerator.device,
            )
        self._metrics[mode].setdefault("reward/perception_mean", []).append(float(perception_stats.get("mean", 0.0)))
        self._metrics[mode].setdefault("reward/perception_skipped_rate", []).append(float(perception_stats.get("skipped_rate", 0.0)))
        self._metrics[mode].setdefault("reward/perception_judge_parse_fail_rate", []).append(float(perception_stats.get("judge_parse_fail_rate", 0.0)))
        self._metrics[mode].setdefault("reward/diagnostic_deplot_overlap_mean", []).append(float(perception_stats.get("diagnostic_deplot_overlap_mean", 0.0)))

        eval_format_cfg = self.opsd_config.get("eval_format_reward", {}) or {}
        eval_format_enabled = bool(eval_format_cfg.get("enabled", False))
        eval_format_weight = float(eval_format_cfg.get("weight", 0.1) or 0.0)
        eval_format_rewards = torch.zeros_like(all_rewards)
        if eval_format_enabled:
            eval_format_rewards = torch.tensor(
                score_eval_format_rewards(completions),
                dtype=torch.float32,
                device=self.accelerator.device,
            )
        self._metrics[mode].setdefault("reward/eval_format_mean", []).append(
            float(eval_format_rewards.detach().float().mean().item())
        )
        self._metrics[mode].setdefault("reward/eval_format_weight", []).append(
            eval_format_weight if eval_format_enabled else 0.0
        )

        rewards_per_func = torch.zeros([len(all_rewards), 3], device=device)

        rewards_per_func[:, 0] = format_rewards.clone()
        rewards_per_func[:, 1] = context_rewards.clone()
        rewards_per_func[:, -1] = acc_rewards.clone()
        local_weighted_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        local_weighted_rewards = local_weighted_rewards + perception_weight * perception_rewards
        local_weighted_rewards = local_weighted_rewards + eval_format_weight * eval_format_rewards
        local_group_reward_std = local_weighted_rewards.view(-1, self.num_generations).std(dim=1)

        opsd_debug.log_sync_point(
            "dist",
            "before accelerate.gather(rewards_per_func)",
            local_shape=tuple(rewards_per_func.shape),
            device=str(rewards_per_func.device),
        )
        with opsd_debug.timed("dist", "accelerate.gather(rewards_per_func)"):
            rewards_per_func = gather(rewards_per_func)
        opsd_debug.log(
            "dist",
            "after accelerate.gather(rewards_per_func)",
            gathered_shape=tuple(rewards_per_func.shape),
        )

        # Apply weights to each reward function's output and sum
        perception_rewards_gathered = gather(perception_rewards)
        eval_format_rewards_gathered = gather(eval_format_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        rewards = rewards + perception_weight * perception_rewards_gathered
        rewards = rewards + eval_format_weight * eval_format_rewards_gathered

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        reward_std_mean = std_grouped_rewards.mean().detach()

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        advantages = advantages.reshape(-1, 1)
        acc_rewards = acc_rewards.view(-1, self.num_generations)
        format_rewards = format_rewards.view(-1, self.num_generations)

        threshold = self.opsd_config.get("gate", {}).get("correct_threshold", 0.5)
        has_correct = (acc_rewards > threshold).sum(1)
        group_has_correct_list = (has_correct > 0).detach().cpu().tolist()
        group_reward_std_list = local_group_reward_std.detach().float().cpu().tolist()
        global_step = getattr(self.state, "global_step", self._step)
        max_training_steps = self._max_training_steps()
        mixed_count = int(
            ((has_correct > 0) & (has_correct < self.num_generations)).sum().item()
        )
        advantage_groups = advantages.detach().view(-1, self.num_generations)
        zero_loss_count = int(
            (advantage_groups.abs().amax(dim=1) <= 1e-12).sum().item()
        )
        self._update_adaptive_supervision(
            mode=mode,
            global_step=int(global_step),
            prompt_count=int(has_correct.numel()),
            mixed_count=mixed_count,
            zero_loss_count=zero_loss_count,
        )
        sampler_updates = {"mixed": 0, "all_wrong": 0, "all_correct": 0, "missing_index": 0}
        if self._effective_signal_sampler is not None:
            self._effective_signal_sampler.set_step(global_step, max_steps=max_training_steps)
            correct_counts = has_correct.detach().cpu().tolist()
            for prompt_idx, correct_count in enumerate(correct_counts):
                source_row = prompt_idx * self.num_generations
                sample_i = inputs[source_row] if source_row < len(inputs) else {}
                dataset_index = sample_i.get("_dyme_index") if isinstance(sample_i, dict) else None
                if dataset_index is None:
                    sampler_updates["missing_index"] += 1
                    continue
                correct_int = int(correct_count)
                if correct_int <= 0:
                    sampler_updates["all_wrong"] += 1
                elif correct_int >= self.num_generations:
                    sampler_updates["all_correct"] += 1
                else:
                    sampler_updates["mixed"] += 1
                reward_std_i = group_reward_std_list[prompt_idx] if prompt_idx < len(group_reward_std_list) else 0.0
                self._effective_signal_sampler.update_prompt_signal(
                    dataset_index=int(dataset_index),
                    correct_count=correct_int,
                    num_generations=self.num_generations,
                    reward_std=float(reward_std_i),
                )
            self._metrics[mode].setdefault("sampling/effective_enabled", []).append(1.0)
            self._metrics[mode].setdefault("sampling/effective_mixed_updates", []).append(
                float(sampler_updates["mixed"])
            )
            self._metrics[mode].setdefault("sampling/effective_all_wrong_updates", []).append(
                float(sampler_updates["all_wrong"])
            )
            self._metrics[mode].setdefault("sampling/effective_all_correct_updates", []).append(
                float(sampler_updates["all_correct"])
            )
            self._metrics[mode].setdefault("sampling/effective_missing_index", []).append(
                float(sampler_updates["missing_index"])
            )
        repaired_prompt_seen = 0
        repaired_prompt_to_mixed = 0
        repaired_prompt_still_all_wrong = 0
        for prompt_idx, has_correct_i in enumerate(group_has_correct_list):
            source_row = prompt_idx * self.num_generations
            sample_i = inputs[source_row] if source_row < len(inputs) else {}
            if self._teacher_sft_repair_prompt_key(sample_i) not in self._teacher_sft_repaired_prompt_keys:
                continue
            repaired_prompt_seen += 1
            if bool(has_correct_i):
                repaired_prompt_to_mixed += 1
            else:
                repaired_prompt_still_all_wrong += 1

        opsd_debug.set_detail_step(global_step)

        opsd_active = self.opsd_config.get("enabled", False) and self.opsd_config.get("mode", "dyme") != "dyme"
        completion_modes = None
        recoverable_flags = None
        answer_flag = getattr(self.checker, "answer_flag", "Answer:")
        teacher_trajs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        teacher_traj_texts: dict[int, str] = {}
        teacher_sft_repairs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        teacher_sft_repair_indices: set[int] = set()
        teacher_sft_repair_stats = None
        teacher_sft_privileged_tag_count = 0
        teacher_sft_target_raw_full_hint_count = 0
        teacher_sft_target_full_hint_count = 0
        teacher_sft_target_exact_answer_count = 0
        teacher_sft_target_fallback_hint_count = 0
        teacher_sft_target_raw_clipped_count = 0
        teacher_sft_target_student_short_count = 0
        teacher_sft_target_answer_only_count = 0
        teacher_probe_stats: dict[str, Any] = {
            "teacher_probe_candidates": 0,
            "teacher_probe_correct": 0,
            "teacher_probe_wrong": 0,
            "teacher_probe_skipped_budget": 0,
        }
        route_guard_stats = None
        utility_routing_stats = None
        opd_cap_stats = None
        opsd_debug.log(
            "opsd_router",
            "OPSD activation check",
            opsd_active=opsd_active,
            config_enabled=self.opsd_config.get("enabled", False),
            config_mode=self.opsd_config.get("mode", "dyme"),
            correct_threshold=threshold,
        )
        if opsd_active:
            with opsd_debug.timed("opsd_router", "estimate_recoverable_flags"):
                recoverable_flags = estimate_recoverable_flags(
                    inputs,
                    self.num_generations,
                    self.opsd_config,
                )
            opsd_debug.log("opsd_router", "recoverable flags computed", recoverable_flags=recoverable_flags)
            with opsd_debug.timed("opsd_router", "route_completion_modes"):
                route_opsd_config = {**self.opsd_config, "global_step": global_step}
                completion_modes = route_completion_modes(
                    acc_rewards,
                    self.num_generations,
                    batch_size,
                    route_opsd_config,
                    recoverable_flags,
                    format_rewards=format_rewards,
                )
            prompt_modes = [
                completion_modes[i * self.num_generations]
                for i in range(acc_rewards.shape[0])
            ]
            opsd_debug.log_mode_summary("opsd_router", prompt_modes, completion_modes)
            teacher_probe_start = self._perf_start()
            completion_modes, teacher_trajs, teacher_traj_texts, teacher_probe_stats = self._apply_teacher_probe_routing(
                inputs=inputs,
                completion_modes=completion_modes,
                acc_rewards=acc_rewards,
                answers=answers,
                completions=completions,
                answer_flag=answer_flag,
                global_step=global_step,
                device=device,
                group_has_correct=group_has_correct_list,
                group_reward_std=group_reward_std_list,
            )
            self._perf_metric(mode, "teacher_probe_s", self._perf_elapsed(teacher_probe_start))
            self._perf_metric(mode, "teacher_probe_generate_s", teacher_probe_stats.get("teacher_probe_generate_s", 0.0))
            self._perf_metric(mode, "teacher_probe_candidates", teacher_probe_stats.get("teacher_probe_candidates", 0))
            route_qualities: list[CompletionQuality] | None = None

            def build_route_qualities() -> list[CompletionQuality]:
                qualities: list[CompletionQuality] = []
                for row in range(len(completion_modes)):
                    eff_len = int(completion_mask[row].sum().item())
                    ids_eff = completion_ids[row, :eff_len].tolist() if eff_len > 0 else []
                    text_i = completions[row] if row < len(completions) else ""
                    no_eos = not bool(is_eos[row].any().item())
                    clipped = no_eos or (
                        self.max_completion_length is not None
                        and eff_len >= int(self.max_completion_length) - 1
                    )
                    degenerate = opsd_diagnostics.is_degenerate_completion(
                        ids_eff,
                        text_i,
                        answer_flag=answer_flag,
                        require_answer_flag=True,
                    )
                    qualities.append(
                        CompletionQuality(
                            degenerate=degenerate,
                            clipped=clipped,
                            force_sft=self._should_force_sft_replace(row, completions, answer_flag),
                            table_spam=is_table_spam_completion(text_i),
                        )
                    )
                return qualities

            cot_quality_cfg = self._chart_cot_quality_gate_config()
            if cot_quality_cfg.enabled:
                prompt_count = (
                    len(completion_modes) + max(self.num_generations, 1) - 1
                ) // max(self.num_generations, 1)
                quality_samples: list[dict[str, Any]] = []
                for prompt_idx in range(prompt_count):
                    completion_idx = prompt_idx * max(self.num_generations, 1)
                    source_idx = self._source_row_index(
                        completion_idx,
                        len(inputs),
                        len(completion_modes),
                    )
                    quality_samples.append(inputs[source_idx] if source_idx < len(inputs) else {})
                cot_quality = evaluate_teacher_trajectory_quality(
                    teacher_traj_texts=teacher_traj_texts,
                    samples=quality_samples,
                    num_generations=self.num_generations,
                    config=cot_quality_cfg,
                )
                for metric_name, metric_value in cot_quality.gate_result.metrics.items():
                    self._metrics[mode].setdefault(
                        f"cot_verify/{metric_name}", []
                    ).append(float(metric_value))
                append_quality_sample_records(
                    output_dir=str(getattr(self.args, "output_dir", "") or ""),
                    rank=int(self.accelerator.process_index),
                    global_step=int(global_step),
                    records=cot_quality.sample_records,
                )
                if cot_quality_cfg.gate_active:
                    eligible = cot_quality.gate_result.eligible_indices
                    teacher_trajs = {
                        idx: trajectory
                        for idx, trajectory in teacher_trajs.items()
                        if idx in eligible
                    }
            utility_cfg = self._signal_utility_routing_config()
            if utility_cfg["enabled"]:
                if route_qualities is None:
                    route_qualities = build_route_qualities()
                teacher_correct_indices = {
                    idx
                    for idx, mode_i in enumerate(completion_modes)
                    if mode_i == MODE_OPSD
                }
                student_correct_flags = (
                    (acc_rewards.reshape(-1) > threshold)
                    .detach()
                    .cpu()
                    .tolist()
                )
                readiness_value = 0.0
                if self._adaptive_supervision_state is not None:
                    readiness_value = float(
                        getattr(
                            self._adaptive_supervision_state,
                            "mastery",
                            getattr(self._adaptive_supervision_state, "readiness", 0.0),
                        )
                    )
                stable_state_keys = (
                    self._signal_utility_state_keys(inputs, len(completion_modes))
                    if bool(utility_cfg.get("mode_stable_enabled", False))
                    else None
                )
                (
                    completion_modes,
                    kept_traj_indices,
                    utility_routing_stats,
                ) = apply_signal_utility_routing(
                    completion_modes=completion_modes,
                    teacher_traj_indices=set(teacher_trajs.keys()),
                    teacher_correct_indices=teacher_correct_indices,
                    student_correct=student_correct_flags,
                    group_has_correct=group_has_correct_list,
                    group_reward_std=group_reward_std_list,
                    qualities=route_qualities,
                    num_generations=self.num_generations,
                    readiness=readiness_value,
                    config=utility_cfg,
                    state_keys=stable_state_keys,
                    mode_stable_states=self._mode_stable_route_states,
                    global_step=global_step,
                )
                if utility_routing_stats.updated_stable_states:
                    self._mode_stable_route_states.update(
                        utility_routing_stats.updated_stable_states
                    )
                teacher_trajs = {
                    idx: traj
                    for idx, traj in teacher_trajs.items()
                    if idx in kept_traj_indices
                }
                prompt_modes = [
                    completion_modes[i * self.num_generations]
                    for i in range(acc_rewards.shape[0])
                ]
                opsd_debug.log_mode_summary(
                    "opsd_router_post_utility", prompt_modes, completion_modes
                )
            repair_cfg = self._teacher_correct_repair_config()
            teacher_correct_indices = {
                idx
                for idx, mode_i in enumerate(completion_modes)
                if mode_i == MODE_OPSD
            }
            (
                completion_modes,
                kept_traj_indices,
                teacher_sft_repair_indices,
                teacher_sft_repair_stats,
            ) = apply_teacher_sft_repair_routing(
                completion_modes=completion_modes,
                teacher_traj_indices=set(teacher_trajs.keys()),
                teacher_correct_indices=teacher_correct_indices,
                group_has_correct=group_has_correct_list,
                num_generations=self.num_generations,
                config=repair_cfg,
            )
            if repair_cfg["mode"] in ("traj_sft", "refiner_sft") and teacher_sft_repair_indices:
                for repair_idx in teacher_sft_repair_indices:
                    raw_text = teacher_traj_texts.get(repair_idx, "")
                    prompt_idx = repair_idx // self.num_generations
                    sample_i = inputs[prompt_idx] if prompt_idx < len(inputs) else {}
                    reference_answer = answers[prompt_idx] if prompt_idx < len(answers) else sample_i.get("answer", "")
                    ids_i, mask_i, privileged_tag_present, target_audit = self._tokenize_teacher_sft_repair_target(
                        raw_text,
                        sample=sample_i,
                        reference_answer=reference_answer,
                        device=device,
                        max_tokens=repair_cfg["target_max_tokens"],
                        sanitize_privileged=repair_cfg["sanitize_privileged"],
                        target_constraint=repair_cfg["target_constraint"],
                        target_style=repair_cfg["target_style"],
                    )
                    teacher_sft_repairs[repair_idx] = (ids_i, mask_i)
                    if privileged_tag_present:
                        teacher_sft_privileged_tag_count += 1
                    teacher_sft_target_raw_full_hint_count += int(
                        bool(target_audit.get("raw_full_hint_format", False))
                    )
                    teacher_sft_target_full_hint_count += int(
                        bool(target_audit.get("full_hint_format", False))
                    )
                    teacher_sft_target_exact_answer_count += int(
                        bool(target_audit.get("exact_reference_answer_line", False))
                    )
                    teacher_sft_target_fallback_hint_count += int(
                        bool(target_audit.get("used_fallback_hint", False))
                    )
                    teacher_sft_target_raw_clipped_count += int(
                        bool(target_audit.get("raw_clipped", False))
                    )
                    teacher_sft_target_student_short_count += int(
                        bool(target_audit.get("student_short_format", False))
                    )
                    teacher_sft_target_answer_only_count += int(
                        bool(target_audit.get("answer_only_format", False))
                    )
                    self._teacher_sft_repaired_prompt_keys.add(
                        self._teacher_sft_repair_prompt_key(sample_i)
                    )
                teacher_trajs = {
                    idx: traj
                    for idx, traj in teacher_trajs.items()
                    if idx in kept_traj_indices
                }
            (
                completion_modes,
                kept_traj_indices,
                opd_cap_stats,
            ) = apply_opd_route_cap(
                completion_modes=completion_modes,
                teacher_traj_indices=set(teacher_trajs.keys()),
                group_has_correct=group_has_correct_list,
                num_generations=self.num_generations,
                global_step=global_step,
                max_steps=max_training_steps,
                config=(
                    self._adaptive_opd_route_cap_config()
                    or self._opd_route_cap_config()
                ),
            )
            if opd_cap_stats and opd_cap_stats.capped:
                teacher_trajs = {
                    idx: traj
                    for idx, traj in teacher_trajs.items()
                    if idx in kept_traj_indices
                }
            self._opsd_distributed_barrier("wait_for_everyone after teacher_probe routing")
            prompt_modes = [
                completion_modes[i * self.num_generations]
                for i in range(acc_rewards.shape[0])
            ]
            opsd_debug.log_mode_summary("opsd_router_post_probe", prompt_modes, completion_modes)
            gate_cfg = self.opsd_config.get("gate", {}) or {}
            route_guard_enabled = bool(
                gate_cfg.get("signal_aware_routing", False)
                or gate_cfg.get("degenerate_hard_override", False)
                or gate_cfg.get("clipped_hard_override", False)
            )
            if route_guard_enabled:
                if route_qualities is None:
                    route_qualities = build_route_qualities()
                completion_modes, kept_traj_indices, route_guard_stats = apply_signal_aware_routing(
                    completion_modes=completion_modes,
                    teacher_traj_indices=set(teacher_trajs.keys()),
                    qualities=route_qualities,
                    group_reward_std=group_reward_std_list,
                    num_generations=self.num_generations,
                    config=gate_cfg,
                )
                teacher_trajs = {
                    idx: traj
                    for idx, traj in teacher_trajs.items()
                    if idx in kept_traj_indices
                }
                prompt_modes = [
                    completion_modes[i * self.num_generations]
                    for i in range(acc_rewards.shape[0])
                ]
                opsd_debug.log_mode_summary("opsd_router_post_signal_guard", prompt_modes, completion_modes)

        format_rewards_flat = format_rewards.reshape(-1)
        acc_rewards_flat = acc_rewards.reshape(-1)
        context_rewards_flat = context_rewards.reshape(-1)
        opsd_diagnostics.log_reward_diagnostics(
            global_step=global_step,
            format_rewards=format_rewards_flat,
            acc_rewards=acc_rewards_flat,
            context_rewards=context_rewards_flat,
            all_rewards=all_rewards,
            advantages=advantages,
            reward_weights=self.reward_weights,
            num_generations=self.num_generations,
            answers=answers,
            completions=completions,
        )

        format_rewards = format_rewards_flat

        sft_slots = 0 if self.opsd_config.get("gate", {}).get("disable_online_sft_slots") else self._sft_slots_for_step()
        sft_check = []
        for i in range(batch_size):
            batch_id = i // self.num_generations
            gen_idx = i % self.num_generations
            sft_check.append((has_correct[batch_id] == 0) and (gen_idx < sft_slots))

        hints_before_refine = list(hints)
        refine_fn = (
            refine_context_sequential
            if getattr(self.refiner, "requires_sequential", False)
            else refine_context_in_parallel
        )
        hints = refine_fn(
            self.refiner,
            question_wo_prompts,
            hints,
            answers,
            task=self.task_name,
            gpu_id=gpu_id,
        )
        hint_changed = [
            (h or "").strip() != (o or "").strip()
            for h, o in zip(hints, hints_before_refine)
        ]
        opsd_debug.log("refine", "context refinement finished", num_hints=len(hints))

        sft_gt = self._build_online_sft_targets(hints, answers, inputs)

        sft_dt = self.processing_class.tokenizer(sft_gt, return_tensors="pt", padding=True,
                                                        padding_side="right")
        sft_padded_ids = sft_dt['input_ids'].to(device)
        sft_attn_masks = sft_dt['attention_mask'].to(device)
        sft_advantages = torch.ones_like(sft_attn_masks, device=device)

        final_completion_id_list = []
        final_completion_mask_list = []
        final_advantange_list = []
        opsd_mask_list = []
        sft_replaced_list = []
        opsd_skipped_degenerate = 0
        opsd_skipped_leakage = 0
        opsd_on_correct = 0
        grpo_on_correct = 0
        teacher_sft_repair_used_count = 0
        teacher_sft_repair_all_wrong_used_count = 0
        skip_degenerate_opsd = self._resolve_skip_degenerate_opsd()
        opsd_degenerate_require_answer_flag = self.opsd_config.get("gate", {}).get(
            "opsd_degenerate_require_answer_flag", True
        )
        threshold = self.opsd_config.get("gate", {}).get("correct_threshold", 0.5)

        for i in range(len(sft_padded_ids)):
            batch_id = i // self.num_generations
            cm = completion_modes[i] if completion_modes is not None else None
            use_grpo = opsd_active and cm == MODE_GRPO
            use_opsd = opsd_active and (
                cm == MODE_OPSD or self.opsd_config.get("mode") == "opsd_only"
            )
            use_sft = (not opsd_active) or cm == MODE_SFT
            joint_opsd = opsd_active and self.opsd_config.get("mode") == "grpo_opsd_joint" and has_correct[batch_id] > 0
            sft_replaced = False
            teacher_probe_correct = teacher_probe_route_confirmed(
                mode_name=str(self.opsd_config.get("mode", "")),
                completion_mode=cm,
                has_teacher_trajectory=i in teacher_trajs,
            )
            force_sft_replace = self._should_force_sft_replace(i, completions, answer_flag)
            gate_cfg = self.opsd_config.get("gate", {}) or {}
            route_guard_enabled = bool(
                gate_cfg.get("signal_aware_routing", False)
                or gate_cfg.get("degenerate_hard_override", False)
                or gate_cfg.get("clipped_hard_override", False)
            )
            if (
                self.opsd_config.get("mode") == "dyme_teacher_probe_opd"
                and teacher_probe_correct
                and not route_guard_enabled
            ):
                # User-selected rule: wrong/degenerate samples may still use OPD
                # when the no-leak teacher probe answers correctly.
                force_sft_replace = False

            repair_traj = teacher_sft_repairs.get(i)
            repair_requested = i in teacher_sft_repair_indices
            if cm == MODE_SKIP:
                completion_id_ = completion_ids[i]
                completion_mask_ = completion_mask[i]
                advantange_ = torch.zeros(
                    completion_mask[i].size(0),
                    device=device,
                    dtype=torch.float,
                )
                opsd_mask_list.append(False)
            elif repair_traj is not None:
                repair_ids, repair_mask = repair_traj
                completion_id_ = torch.cat([repair_ids, completion_ids[i][0:0]])
                completion_mask_ = torch.cat([repair_mask, completion_mask[i][0:0]])
                advantange_ = teacher_sft_repair_advantages(completion_mask_)
                opsd_mask_list.append(False)
                sft_replaced = True
                teacher_sft_repair_used_count += 1
                if has_correct[batch_id] == 0:
                    teacher_sft_repair_all_wrong_used_count += 1
            elif sft_check[i] or (opsd_active and use_sft) or force_sft_replace:
                completion_id_ = torch.cat([sft_padded_ids[i], completion_ids[i][0:0]])
                completion_mask_ = torch.cat([sft_attn_masks[i], completion_mask[i][0:0]])
                advantange_ = torch.cat([sft_advantages[i], advantages[i][0:0]])
                advantange_[:] = 1
                opsd_mask_list.append(False)
                sft_replaced = True
                if repair_requested:
                    teacher_sft_repair_used_count += 1
                    if has_correct[batch_id] == 0:
                        teacher_sft_repair_all_wrong_used_count += 1
            elif use_opsd:
                completion_id_ = completion_ids[i]
                completion_mask_ = completion_mask[i]
                advantange_ = torch.zeros(completion_mask[i].size(0), device=device, dtype=torch.float)
                run_opsd = True
                if skip_degenerate_opsd and not teacher_probe_correct:
                    eff_len = int(completion_mask_.sum().item())
                    if eff_len > 0:
                        ids_eff = completion_ids[i, :eff_len].tolist()
                        text_i = completions[i] if i < len(completions) else ""
                        if opsd_diagnostics.is_degenerate_completion(
                            ids_eff,
                            text_i,
                            answer_flag=answer_flag,
                            require_answer_flag=opsd_degenerate_require_answer_flag,
                        ):
                            run_opsd = False
                            opsd_skipped_degenerate += 1
                            advantange_[:] = 0
                if run_opsd:
                    source_idx = self._source_row_index(i, len(answers), batch_size)
                    gold = answers[source_idx] if source_idx < len(answers) else ""
                    text_i = completions[i] if i < len(completions) else ""
                    if completion_has_leakage_pattern(text_i, gold):
                        run_opsd = False
                        opsd_skipped_leakage += 1
                        advantange_[:] = 0
                if run_opsd and acc_rewards_flat[i].item() > threshold:
                    opsd_on_correct += 1
                    run_opsd = False
                    advantange_[:] = 0
                opsd_mask_list.append(run_opsd)
            elif use_grpo or (completion_modes is None and has_correct[batch_id] > 0):
                if use_grpo or acc_rewards_flat[i].item() > threshold:
                    grpo_on_correct += 1
                completion_id_ = torch.cat([completion_ids[i], sft_padded_ids[i][0:0]])
                completion_mask_ = torch.cat([completion_mask[i], sft_attn_masks[i][0:0]])
                advantange_ = torch.cat([advantages[i], sft_advantages[i][0:0]])
                advantange_ = advantange_.repeat_interleave(len(completion_id_))
                opsd_mask_list.append(joint_opsd)
            else:
                completion_id_ = torch.cat([completion_ids[i], completion_ids[i][0:0]])
                completion_mask_ = torch.cat([completion_mask[i], sft_attn_masks[i][0:0]])
                advantange_ = torch.cat([advantages[i], sft_advantages[i][0:0]])
                advantange_ = advantange_.repeat_interleave(len(completion_id_))
                advantange_[:] = 0
                opsd_mask_list.append(False)

            if has_correct[batch_id] == self.num_generations:
                advantange_[:] = 0

            final_completion_id_list.append(completion_id_)
            final_completion_mask_list.append(completion_mask_)
            final_advantange_list.append(advantange_)
            sft_replaced_list.append(sft_replaced)
            if hasattr(self.checker, "record_route_binding"):
                route_name = "sft_replaced" if sft_replaced else (
                    "grpo" if (use_grpo or (completion_modes is None and has_correct[batch_id] > 0)) else (
                        "opsd" if use_opsd else "other"
                    )
                )
                self.checker.record_route_binding(
                    sample_idx=i,
                    route=route_name,
                    checker_score=float(context_rewards_flat[i].item()),
                    answer_reward=float(acc_rewards_flat[i].item()),
                    format_reward=float(format_rewards_flat[i].item()),
                    refiner_changed=hint_changed[i] if i < len(hint_changed) else False,
                    consumed_refined_hint=bool(sft_replaced and i < len(hint_changed) and hint_changed[i]),
                )

        self._finish_visual_supervision_batch(global_step)

        opsd_debug.log(
            "opsd_mask",
            "completion routing finished",
            opsd_active=opsd_active,
            opsd_mask_true=sum(opsd_mask_list),
            opsd_mask_false=len(opsd_mask_list) - sum(opsd_mask_list),
            sft_replaced_count=sum(sft_replaced_list),
            opsd_skipped_degenerate=opsd_skipped_degenerate,
            has_correct=has_correct.tolist() if hasattr(has_correct, "tolist") else has_correct,
        )

        effective_filter_cfg = self._effective_group_filter_config()
        effective_keep_mask, effective_filter_stats = compute_effective_group_keep_mask(
            correct_counts=[int(x) for x in has_correct.detach().cpu().tolist()],
            num_generations=self.num_generations,
            global_step=global_step,
            config=effective_filter_cfg,
        )
        effective_filter_teacher_traj_removed = 0
        if effective_filter_stats.filtered_total > 0:
            effective_filter_teacher_traj_removed = apply_effective_group_filter_to_routes(
                keep_mask=effective_keep_mask,
                completion_masks=final_completion_mask_list,
                advantages=final_advantange_list,
                opsd_mask=opsd_mask_list,
                sft_replaced=sft_replaced_list,
                teacher_trajs=teacher_trajs,
            )
            self._metrics[mode].setdefault("filter/effective_group_filtered_rate", []).append(
                effective_filter_stats.filtered_total / max(effective_filter_stats.total, 1)
            )
            self._metrics[mode].setdefault("filter/all_wrong_overflow_filtered", []).append(
                float(effective_filter_stats.filtered_all_wrong)
            )
            self._metrics[mode].setdefault("filter/all_correct_filtered", []).append(
                float(effective_filter_stats.filtered_all_correct)
            )
            self._metrics[mode].setdefault("filter/teacher_traj_removed", []).append(
                float(effective_filter_teacher_traj_removed)
            )
        else:
            self._metrics[mode].setdefault("filter/effective_group_filtered_rate", []).append(0.0)
            self._metrics[mode].setdefault("filter/all_wrong_overflow_filtered", []).append(0.0)
            self._metrics[mode].setdefault("filter/all_correct_filtered", []).append(0.0)
            self._metrics[mode].setdefault("filter/teacher_traj_removed", []).append(0.0)

        if self._global_signal_logging_enabled():
            final_routes: list[str] = []
            for idx in range(len(sft_replaced_list)):
                if sft_replaced_list[idx]:
                    final_routes.append("sft")
                elif opsd_mask_list[idx]:
                    final_routes.append("opd")
                elif completion_modes is not None and completion_modes[idx] == MODE_SKIP:
                    final_routes.append("skip")
                else:
                    final_routes.append("grpo")

            eos_flags = [bool(is_eos[idx].any().item()) for idx in range(batch_size)]
            clipped_flags: list[bool] = []
            degenerate_flags: list[bool] = []
            for idx in range(batch_size):
                effective_length = int(completion_mask[idx].sum().item())
                clipped_flags.append(
                    not eos_flags[idx]
                    or (
                        self.max_completion_length is not None
                        and effective_length >= int(self.max_completion_length) - 1
                    )
                )
                ids_effective = (
                    completion_ids[idx, :effective_length].tolist()
                    if effective_length > 0
                    else []
                )
                text = completions[idx] if idx < len(completions) else ""
                degenerate_flags.append(
                    opsd_diagnostics.is_degenerate_completion(
                        ids_effective,
                        text,
                        answer_flag=answer_flag,
                        require_answer_flag=True,
                    )
                )

            total_zero_flags = (
                advantage_groups.abs().amax(dim=1) <= 1e-12
            ).detach().cpu().tolist()
            local_global_counts = counts_from_local_batch(
                correct_counts=[int(value) for value in has_correct.detach().cpu().tolist()],
                total_reward_zero_flags=[bool(value) for value in total_zero_flags],
                num_generations=self.num_generations,
                routes=final_routes,
                accuracy_rewards=[float(value) for value in acc_rewards.detach().reshape(-1).cpu().tolist()],
                clipped_flags=clipped_flags,
                eos_flags=eos_flags,
                degenerate_flags=degenerate_flags,
            )
            self._reduce_global_training_signal(
                mode=mode,
                counts=local_global_counts,
                global_step=int(global_step),
            )

        if self._health_monitor is not None:
            local_routing_n = max(len(sft_replaced_list), 1)
            local_prompt_n = max(int(has_correct.numel()), 1)
            all_wrong_groups = int((has_correct == 0).sum().item())
            mixed_groups = int(((has_correct > 0) & (has_correct < self.num_generations)).sum().item())
            reward_std_local = local_group_reward_std.detach().float()
            reward_std_denom = max(int(reward_std_local.numel()), 1)
            wrong_completion_count = int((acc_rewards <= threshold).sum().item())
            opd_route_count = int(sum(1 for value in opsd_mask_list if value))
            sft_route_count = int(sum(1 for value in sft_replaced_list if value))
            if completion_modes is not None:
                grpo_route_count = int(sum(1 for cm in completion_modes if cm == MODE_GRPO))
                skip_route_count = int(sum(1 for cm in completion_modes if cm == MODE_SKIP))
            else:
                grpo_route_count = int(
                    sum(1 for i in range(len(sft_replaced_list)) if has_correct[i // self.num_generations] > 0)
                )
                skip_route_count = 0
            probe_candidates = int(teacher_probe_stats.get("teacher_probe_candidates", 0) or 0)
            probe_correct = int(teacher_probe_stats.get("teacher_probe_correct", 0) or 0)
            probe_wrong = int(teacher_probe_stats.get("teacher_probe_wrong", 0) or 0)
            probe_probed = probe_correct + probe_wrong
            probe_candidate_denom = max(probe_candidates, 1)
            probe_probed_denom = max(probe_probed, 1)
            guard_degenerate = int(getattr(route_guard_stats, "degenerate_hard_overrides", 0) or 0)
            guard_clipped = int(getattr(route_guard_stats, "clipped_hard_overrides", 0) or 0)
            guard_teacher = int(getattr(route_guard_stats, "teacher_correct_overrides", 0) or 0)
            guard_signal_sft = int(getattr(route_guard_stats, "signal_aware_sft", 0) or 0)
            utility_candidates = int(getattr(utility_routing_stats, "candidate_count", 0) or 0)
            utility_denom = max(utility_candidates, 1)
            utility_reroute_grpo = int(getattr(utility_routing_stats, "rerouted_grpo", 0) or 0)
            utility_reroute_opd = int(getattr(utility_routing_stats, "rerouted_opd", 0) or 0)
            utility_reroute_sft = int(getattr(utility_routing_stats, "rerouted_sft", 0) or 0)
            utility_traj_removed = int(getattr(utility_routing_stats, "teacher_traj_removed", 0) or 0)
            utility_switches = int(getattr(utility_routing_stats, "switches", 0) or 0)
            utility_blocked_switches = int(getattr(utility_routing_stats, "blocked_switches", 0) or 0)
            utility_stable_holds = int(getattr(utility_routing_stats, "stable_holds", 0) or 0)
            utility_invalid_current_switches = int(
                getattr(utility_routing_stats, "invalid_current_switches", 0) or 0
            )
            opd_cap_capped = int(getattr(opd_cap_stats, "capped", 0) or 0)
            opd_cap_prompts = int(getattr(opd_cap_stats, "eligible_prompts", 0) or 0)
            opd_cap_kept = int(getattr(opd_cap_stats, "kept_opd", 0) or 0)
            opd_cap_traj_removed = int(getattr(opd_cap_stats, "teacher_traj_removed", 0) or 0)
            opd_cap_grpo = int(getattr(opd_cap_stats, "rerouted_grpo", 0) or 0)
            opd_cap_skip = int(getattr(opd_cap_stats, "skipped", 0) or 0)
            repair_slot_eligible = int(getattr(teacher_sft_repair_stats, "repair_slot_eligible", 0) or 0)
            repair_to_opd = int(getattr(teacher_sft_repair_stats, "teacher_correct_to_opd", 0) or 0)
            repair_to_sft = int(
                getattr(teacher_sft_repair_stats, "teacher_correct_to_sft_repair", 0) or 0
            )
            repair_teacher_denom = max(repair_to_opd + repair_to_sft, 1)
            repaired_prompt_denom = max(repaired_prompt_seen, 1)
            self._health_monitor.record_routing(
                global_step,
                {
                    "sft_replaced_ratio": sum(sft_replaced_list) / local_routing_n,
                    "opsd_skipped_degenerate": opsd_skipped_degenerate,
                    "opsd_skipped_leakage": opsd_skipped_leakage,
                    "opsd_on_correct_rate": opsd_on_correct / local_routing_n,
                    "grpo_on_correct_rate": grpo_on_correct / local_routing_n,
                    "opd_teacher_call_rate": sum(opsd_mask_list) / local_routing_n,
                    "grpo_route_rate": grpo_route_count / local_routing_n,
                    "opd_route_rate": opd_route_count / local_routing_n,
                    "sft_route_rate": sft_route_count / local_routing_n,
                    "skip_route_rate": skip_route_count / local_routing_n,
                    "total_completion_count": len(sft_replaced_list),
                    "wrong_completion_count": wrong_completion_count,
                    "probe_candidate_count": probe_candidates,
                    "teacher_correct_count": probe_correct,
                    "opd_route_count": opd_route_count,
                    "sft_route_count": sft_route_count,
                    "grpo_route_count": grpo_route_count,
                    "degenerate_hard_override_rate": guard_degenerate / local_routing_n,
                    "clipped_hard_override_rate": guard_clipped / local_routing_n,
                    "teacher_correct_overridden_rate": guard_teacher / local_routing_n,
                    "signal_aware_sft_rate": guard_signal_sft / local_routing_n,
                    "utility_candidate_rate": utility_candidates / local_routing_n,
                    "utility_grpo_mean": float(getattr(utility_routing_stats, "grpo_mean", 0.0) or 0.0),
                    "utility_opd_mean": float(getattr(utility_routing_stats, "opd_mean", 0.0) or 0.0),
                    "utility_sft_mean": float(getattr(utility_routing_stats, "sft_mean", 0.0) or 0.0),
                    "utility_margin_mean": float(getattr(utility_routing_stats, "margin_mean", 0.0) or 0.0),
                    "utility_reroute_grpo_rate": utility_reroute_grpo / utility_denom,
                    "utility_reroute_opd_rate": utility_reroute_opd / utility_denom,
                    "utility_reroute_sft_rate": utility_reroute_sft / utility_denom,
                    "utility_teacher_traj_removed_rate": utility_traj_removed / local_routing_n,
                    "utility_switch_rate": utility_switches / utility_denom,
                    "utility_blocked_switch_rate": utility_blocked_switches / utility_denom,
                    "utility_stable_hold_rate": utility_stable_holds / utility_denom,
                    "utility_invalid_current_switch_rate": utility_invalid_current_switches / utility_denom,
                    "utility_switch_gain_mean": float(getattr(utility_routing_stats, "switch_gain_mean", 0.0) or 0.0),
                    "utility_ema_grpo_mean": float(getattr(utility_routing_stats, "ema_grpo_mean", 0.0) or 0.0),
                    "utility_ema_opd_mean": float(getattr(utility_routing_stats, "ema_opd_mean", 0.0) or 0.0),
                    "utility_ema_sft_mean": float(getattr(utility_routing_stats, "ema_sft_mean", 0.0) or 0.0),
                    "utility_mode_stable_state_count": float(len(getattr(self, "_mode_stable_route_states", {}) or {})),
                    "opd_route_cap_rate": opd_cap_capped / local_routing_n,
                    "opd_route_cap_prompt_rate": opd_cap_prompts / local_prompt_n,
                    "opd_route_cap_kept_rate": opd_cap_kept / local_routing_n,
                    "opd_route_cap_teacher_traj_removed_rate": opd_cap_traj_removed / local_routing_n,
                    "opd_route_cap_grpo_rate": opd_cap_grpo / local_routing_n,
                    "opd_route_cap_skip_rate": opd_cap_skip / local_routing_n,
                    "effective_sampling_enabled": 1.0 if self._effective_signal_sampler is not None else 0.0,
                    "effective_sampling_mixed_update_rate": sampler_updates["mixed"] / local_prompt_n,
                    "effective_sampling_all_wrong_update_rate": sampler_updates["all_wrong"] / local_prompt_n,
                    "effective_sampling_all_correct_update_rate": sampler_updates["all_correct"] / local_prompt_n,
                    "effective_sampling_missing_index_rate": sampler_updates["missing_index"] / local_prompt_n,
                    "effective_group_filter_enabled": 1.0 if effective_filter_cfg.enabled else 0.0,
                    "effective_group_filtered_rate": (
                        effective_filter_stats.filtered_total / max(effective_filter_stats.total, 1)
                    ),
                    "effective_group_all_wrong_filtered_rate": (
                        effective_filter_stats.filtered_all_wrong / local_routing_n
                    ),
                    "effective_group_all_correct_filtered_rate": (
                        effective_filter_stats.filtered_all_correct / local_routing_n
                    ),
                    "effective_group_kept_all_wrong_rate": (
                        effective_filter_stats.kept_all_wrong / local_routing_n
                    ),
                    "effective_group_teacher_traj_removed_rate": (
                        effective_filter_teacher_traj_removed / local_routing_n
                    ),
                    "teacher_sft_repair_rate": teacher_sft_repair_used_count / local_routing_n,
                    "teacher_sft_repair_all_wrong_rate": (
                        teacher_sft_repair_all_wrong_used_count / local_routing_n
                    ),
                    "teacher_sft_repair_slot_utilization": (
                        teacher_sft_repair_used_count / max(repair_slot_eligible, 1)
                    ),
                    "teacher_correct_to_opd_rate": repair_to_opd / repair_teacher_denom,
                    "teacher_correct_to_sft_repair_rate": repair_to_sft / repair_teacher_denom,
                    "repaired_prompt_to_mixed_rate": repaired_prompt_to_mixed / repaired_prompt_denom,
                    "repaired_prompt_still_all_wrong_rate": (
                        repaired_prompt_still_all_wrong / repaired_prompt_denom
                    ),
                    "teacher_sft_privileged_tag_rate": (
                        teacher_sft_privileged_tag_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_raw_full_hint_format_rate": (
                        teacher_sft_target_raw_full_hint_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_full_hint_format_rate": (
                        teacher_sft_target_full_hint_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_exact_answer_line_rate": (
                        teacher_sft_target_exact_answer_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_fallback_hint_rate": (
                        teacher_sft_target_fallback_hint_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_raw_clipped_rate": (
                        teacher_sft_target_raw_clipped_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_student_short_rate": (
                        teacher_sft_target_student_short_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "teacher_sft_target_answer_only_rate": (
                        teacher_sft_target_answer_only_count / max(teacher_sft_repair_used_count, 1)
                    ),
                    "group_all_wrong_rate": all_wrong_groups / local_prompt_n,
                    "group_mixed_rate": mixed_groups / local_prompt_n,
                    "reward_std_lt_0_01_rate": float((reward_std_local < 0.01).sum().item()) / reward_std_denom,
                    "reward_std_lt_0_05_rate": float((reward_std_local < 0.05).sum().item()) / reward_std_denom,
                    "reward_std_lt_0_10_rate": float((reward_std_local < 0.10).sum().item()) / reward_std_denom,
                    "teacher_probe_candidate_rate": probe_candidates / local_routing_n,
                    "teacher_probe_correct_rate": probe_correct / local_routing_n,
                    "teacher_probe_wrong_rate": probe_wrong / local_routing_n,
                    "teacher_probe_skipped_no_evidence_rate": teacher_probe_stats.get("teacher_probe_skipped_no_evidence", 0) / local_routing_n,
                    "teacher_probe_skipped_budget_rate": teacher_probe_stats.get("teacher_probe_skipped_budget", 0) / local_routing_n,
                    "teacher_probe_candidate_accuracy": probe_correct / probe_candidate_denom,
                    "teacher_probe_probed_accuracy": probe_correct / probe_probed_denom,
                    "teacher_probe_evidence_present_rate": teacher_probe_stats.get("teacher_probe_evidence_present", 0) / probe_candidate_denom,
                    "teacher_probe_deplot_placeholder_rate": teacher_probe_stats.get("teacher_probe_deplot_placeholder", 0) / probe_candidate_denom,
                    "teacher_probe_deplot_real_rate": teacher_probe_stats.get("teacher_probe_deplot_real", 0) / probe_candidate_denom,
                    "teacher_probe_visual_fact_used_rate": teacher_probe_stats.get("teacher_probe_visual_fact_used", 0) / probe_candidate_denom,
                    "teacher_probe_answer_flag_rate": teacher_probe_stats.get("teacher_probe_answer_flag", 0) / probe_probed_denom,
                    "teacher_probe_parse_fail_rate": teacher_probe_stats.get("teacher_probe_parse_failed", 0) / probe_probed_denom,
                    "teacher_probe_gold_suffix_rate": teacher_probe_stats.get("teacher_probe_gold_suffix", 0) / probe_probed_denom,
                    "teacher_probe_generated_tokens_mean": teacher_probe_stats.get("teacher_probe_generated_tokens_mean", 0.0),
                    "teacher_probe_generated_tokens_p95": teacher_probe_stats.get("teacher_probe_generated_tokens_p95", 0.0),
                    "teacher_probe_clipped_rate": teacher_probe_stats.get("teacher_probe_clipped_rate", 0.0),
                    "format_mean": float(format_rewards.mean().item()),
                    "accuracy_mean": float(acc_rewards.mean().item()),
                    "perception_reward_mean": float(perception_stats.get("mean", 0.0)),
                    "perception_reward_skipped_rate": float(perception_stats.get("skipped_rate", 0.0)),
                    "perception_judge_parse_fail_rate": float(perception_stats.get("judge_parse_fail_rate", 0.0)),
                    "diagnostic_deplot_overlap_mean": float(perception_stats.get("diagnostic_deplot_overlap_mean", 0.0)),
                },
            )

        raw_completion_ids = completion_ids.clone()
        raw_completion_shape = tuple(completion_ids.shape)
        opsd_diagnostics.log_generation_diagnostics(
            global_step=global_step,
            completions=completions,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            is_eos=is_eos,
            max_completion_length=self.max_completion_length,
            num_generations=self.num_generations,
        )

        completion_ids = pad_sequence(final_completion_id_list, batch_first=True,
                                      padding_value=self.processing_class.tokenizer.pad_token_id).long()
        completion_mask = pad_sequence(final_completion_mask_list, batch_first=True, padding_value=0)
        completion_advantange = pad_sequence(final_advantange_list, batch_first=True, padding_value=0)
        completion_ids = completion_ids.to(device)
        completion_mask = completion_mask.to(device)
        completion_advantange = completion_advantange.to(device)

        teacher_traj_mask_list = []
        teacher_traj_id_list = []
        teacher_traj_attn_list = []
        for i in range(completion_ids.shape[0]):
            traj = teacher_trajs.get(i)
            if traj is None:
                teacher_traj_id_list.append(completion_ids[i][0:0])
                teacher_traj_attn_list.append(completion_mask[i][0:0])
                teacher_traj_mask_list.append(False)
            else:
                ids_i, mask_i = traj
                teacher_traj_id_list.append(ids_i)
                teacher_traj_attn_list.append(mask_i)
                teacher_traj_mask_list.append(True)

        opsd_diagnostics.log_routed_completion_probe(
            global_step=global_step,
            trainer_step=self._step,
            raw_completion_shape=raw_completion_shape,
            final_completion_ids=completion_ids,
            final_completion_mask=completion_mask,
            opsd_mask_list=opsd_mask_list,
            sample_count=self._opsd_probe_sample_count,
            tokenizer=self.processing_class.tokenizer,
            sft_replaced_list=sft_replaced_list,
            raw_completion_ids=raw_completion_ids,
        )

        opsd_diagnostics.log_routing_diagnostics(
            global_step=global_step,
            opsd_active=opsd_active,
            opsd_mask_list=opsd_mask_list,
            has_correct=has_correct,
            completion_modes=completion_modes,
            recoverable_flags=recoverable_flags,
            completion_advantages=completion_advantange,
            completion_mask=completion_mask,
        )

        input_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1).long()
        attention_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        if (
            self.accelerator.device.index == 0
            and opsd_debug.should_log_detail(global_step)
        ):
            completion_id = completion_ids[0]
            completion_id_pos = completion_id[
                (completion_advantange[0] > 0) & (completion_mask[0] > 0)
            ]
            completion_id_neg = completion_id[
                (completion_advantange[0] < 0) & (completion_mask[0] > 0)
            ]

            show = self.processing_class.decode(completion_id_pos, skip_special_tokens=False)
            show_neg = self.processing_class.decode(completion_id_neg, skip_special_tokens=False)
            prediction = completions[0] if completions else ""
            opsd_debug.log_detail(
                "routing",
                "completion routing debug sample",
                global_step=global_step,
                has_correct=has_correct.tolist() if hasattr(has_correct, "tolist") else has_correct,
                prediction_preview=prediction[:800] if isinstance(prediction, str) else str(prediction)[:800],
                pos_decode_preview=show[:800] if show else "",
                neg_decode_preview=show_neg[:800] if show_neg else "",
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        local_batch_size = prompt_ids.size(0)
        logps_micro_batch = (
            self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                opsd_debug.log_sync_point(
                    "logps",
                    "before _get_per_token_logps in generate path",
                    input_shape=tuple(input_completion_ids.shape),
                )
                with opsd_debug.timed("logps", "_get_per_token_logps"):
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_completion_ids,
                        attention_completion_mask,
                        pixel_values,
                        image_sizes,
                        logits_to_keep,
                        logps_micro_batch,
                    )
            else:
                old_per_token_logps = None
                opsd_debug.log("logps", "skip old_per_token_logps because num_iterations == 1")

        self._opsd_distributed_barrier("wait_for_everyone before generate-path metric gathers")

        # Log the metrics
        if mode == "train":
            opsd_debug.log_sync_point("dist", "before gather_for_metrics(attention_mask.sum())")
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()

        # log completion lengths, mean, min, max
        opsd_debug.log_sync_point("dist", "before gather_for_metrics(completion_mask.sum(1))")
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())

        # identify sequences that terminated with EOS and log their lengths
        opsd_debug.log_sync_point("dist", "before gather_for_metrics(is_eos.any(dim=1))")
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        # completion_advantange: (batch_size, seq_len) 或 (batch_size, n)
        mask_pos = completion_advantange > 0 
        row_min = completion_advantange.min(dim=1, keepdim=True).values.abs()  # (batch, 1)

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": completion_advantange,
            "old_per_token_logps": old_per_token_logps,
            "img_sizes": image_sizes,
            "acc_rewards": acc_rewards_flat.to(device),
            "reward_std_mean": reward_std_mean.to(device),
            "group_mixed_rate": float(
                ((has_correct > 0) & (has_correct < self.num_generations)).float().mean().item()
            ),
        }

        if opsd_active:
            opsd_mask = torch.tensor(opsd_mask_list, dtype=torch.bool, device=device)
            result["opsd_mask"] = opsd_mask
            traj_cfg = self._teacher_trajectory_config()
            if traj_cfg["enabled"]:
                result["teacher_traj_mask"] = torch.tensor(
                    teacher_traj_mask_list,
                    dtype=torch.bool,
                    device=device,
                )
                result["teacher_traj_completion_ids"] = pad_sequence(
                    teacher_traj_id_list,
                    batch_first=True,
                    padding_value=self.processing_class.tokenizer.pad_token_id,
                ).long().to(device)
                result["teacher_traj_completion_mask"] = pad_sequence(
                    teacher_traj_attn_list,
                    batch_first=True,
                    padding_value=0,
                ).to(device)
            opsd_indices = [i for i, m in enumerate(opsd_mask_list) if m]
            opsd_debug.hang_probe(
                "teacher_build_decision",
                local_opsd_count=len(opsd_indices),
                opsd_indices=opsd_indices,
            )
            opsd_debug.log_sync_point(
                "teacher_prompt",
                "before build_teacher_prompt_batch",
                local_batch_size=local_batch_size,
                opsd_indices=opsd_indices,
                provider_names=self.opsd_config.get("privileged_providers", ["text"]),
            )
            if opsd_indices:
                build_indices = sorted(opsd_indices)
                with opsd_debug.timed("teacher_prompt", "build_teacher_prompt_batch"):
                    teacher_tensors = build_teacher_prompt_batch(
                        self.processing_class,
                        inputs,
                        build_indices,
                        self.opsd_config.get("privileged_providers", ["text"]),
                        device,
                        opsd_config=self.opsd_config,
                        global_step=getattr(self.state, "global_step", self._step),
                        output_dir=self.args.output_dir,
                        expanded_count=local_batch_size,
                        num_generations=self.num_generations,
                    )
                teacher_tensors = expand_teacher_tensors_to_full_batch(
                    teacher_tensors,
                    build_indices,
                    local_batch_size,
                )
                result.update(teacher_tensors)
                opsd_debug.hang_probe(
                    "teacher_build_done",
                    build_indices=build_indices,
                    teacher_prompt_shape=tuple(teacher_tensors["teacher_prompt_ids"].shape),
                )
                for key, value in teacher_tensors.items():
                    opsd_debug.log_tensor("teacher_prompt", key, value)
                teacher_stats = teacher_tensors.get("teacher_stats")
                if self._health_monitor is not None and teacher_stats:
                    self._health_monitor.record_data(
                        getattr(self.state, "global_step", self._step),
                        teacher_stats,
                    )
            else:
                opsd_debug.log(
                    "teacher_prompt",
                    "skip teacher prompt build (no local OPSD samples)",
                    opsd_indices=opsd_indices,
                )
                opsd_debug.hang_probe("teacher_build_skipped", reason="no_local_opsd")
            if opsd_indices:
                mode = "train" if self.model.training else "eval"
                self._metrics[mode].setdefault("opsd/mask_ratio", []).append(
                    len(opsd_indices) / max(local_batch_size, 1)
                )
                opsd_debug.log(
                    "teacher_prompt",
                    "teacher tensors attached to batch",
                    opsd_indices=opsd_indices,
                    opsd_mask_ratio=len(opsd_indices) / max(local_batch_size, 1),
                )
        else:
            opsd_debug.log("teacher_prompt", "OPSD inactive, skip teacher prompt build")

        opsd_debug.log(
            "generate",
            "exit _generate_and_score_completions",
            result_keys=list(result.keys()),
            batch_size=result["prompt_ids"].shape[0],
            opsd_active=opsd_active,
        )
        if opsd_active and "opsd_mask" in result:
            opsd_debug.hang_probe_force(
                "generate_and_score_return",
                opsd_mask_true=int(result["opsd_mask"].sum().item()),
                has_teacher_prompt=result.get("teacher_prompt_ids") is not None,
            )
        return result

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = None

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"][:, 0],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs["old_per_token_logps"],
            ref_per_token_logps=ref_per_token_logps,
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def _compute_positive_replay_loss(self, model, *, global_step: int) -> tuple[torch.Tensor, float, int] | None:
        mode = "train" if self.model.training else "eval"
        buffer = self._positive_replay_buffer
        if mode != "train" or buffer is None:
            return None
        cfg = buffer.config
        available = bool(buffer.available)
        self._metrics[mode].setdefault("replay/positive_available", []).append(1.0 if available else 0.0)
        self._metrics[mode].setdefault("loss/positive_replay_weight", []).append(float(cfg.weight if available else 0.0))
        if not available or not buffer.enabled_for_step(global_step) or self.processing_func is None:
            self._metrics[mode].setdefault("replay/positive_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/positive_batch_size", []).append(0.0)
            return None

        replay_samples = buffer.sample(global_step=global_step)
        if not replay_samples:
            self._metrics[mode].setdefault("replay/positive_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/positive_batch_size", []).append(0.0)
            return None

        replay_batch = self.processing_func(replay_samples)
        replay_inputs = super(DyMETrainer, self)._prepare_inputs(replay_batch)
        labels = replay_inputs.get("labels")
        if labels is None:
            self._metrics[mode].setdefault("replay/positive_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/positive_batch_size", []).append(0.0)
            return None

        outputs = model(**replay_inputs)
        replay_loss = outputs.loss
        replay_tokens = int((labels != -100).sum().detach().item())
        self._metrics[mode].setdefault("loss/positive_replay", []).append(float(replay_loss.detach().item()))
        self._metrics[mode].setdefault("replay/positive_skipped_rate", []).append(0.0)
        self._metrics[mode].setdefault("replay/positive_batch_size", []).append(float(len(replay_samples)))
        self._metrics[mode].setdefault("replay/positive_tokens", []).append(float(replay_tokens))
        return replay_loss, float(cfg.weight), len(replay_samples)

    def _pad_rollout_replay_rows(self, rows: list[torch.Tensor], *, pad_value: int | float = 0) -> torch.Tensor:
        return pad_sequence(
            [row.to(self.accelerator.device) for row in rows],
            batch_first=True,
            padding_value=pad_value,
        )

    def _stack_optional_rollout_tensors(self, rows: list[torch.Tensor | None]) -> torch.Tensor | None:
        return stack_optional_compatible_tensors(rows, device=self.accelerator.device)

    def _compute_rollout_replay_loss(self, model, *, global_step: int) -> tuple[torch.Tensor, float, int] | None:
        mode = "train" if self.model.training else "eval"
        buffer = self._rollout_replay_buffer
        if mode != "train" or buffer is None:
            return None
        cfg = buffer.config
        available = bool(buffer.available)
        self._metrics[mode].setdefault("replay/rollout_available", []).append(1.0 if available else 0.0)
        self._metrics[mode].setdefault("loss/rollout_replay_weight", []).append(float(cfg.weight if available else 0.0))
        if not available or not buffer.enabled_for_step(global_step):
            self._metrics[mode].setdefault("replay/rollout_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/rollout_batch_size", []).append(0.0)
            return None

        entries = buffer.sample(global_step=global_step)
        if not entries:
            self._metrics[mode].setdefault("replay/rollout_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/rollout_batch_size", []).append(0.0)
            return None

        pad_id = int(self.processing_class.tokenizer.pad_token_id)
        replay_prompt_ids = self._pad_rollout_replay_rows([entry.prompt_ids for entry in entries], pad_value=pad_id).long()
        replay_prompt_mask = self._pad_rollout_replay_rows([entry.prompt_mask for entry in entries], pad_value=0).long()
        replay_completion_ids = self._pad_rollout_replay_rows(
            [entry.completion_ids for entry in entries],
            pad_value=pad_id,
        ).long()
        replay_completion_mask = self._pad_rollout_replay_rows(
            [entry.completion_mask for entry in entries],
            pad_value=0,
        ).long()
        replay_old_logps = self._pad_rollout_replay_rows(
            [entry.old_per_token_logps for entry in entries],
            pad_value=0.0,
        ).float()
        replay_advantages = torch.tensor(
            [float(entry.advantage) for entry in entries],
            dtype=torch.float32,
            device=self.accelerator.device,
        )
        replay_input_ids = torch.cat([replay_prompt_ids, replay_completion_ids], dim=1)
        replay_attention_mask = torch.cat([replay_prompt_mask, replay_completion_mask], dim=1)
        replay_pixel_values = self._stack_optional_rollout_tensors([entry.pixel_values for entry in entries])
        replay_image_sizes = self._stack_optional_rollout_tensors([entry.image_sizes for entry in entries])
        has_pixel_values = any(isinstance(entry.pixel_values, torch.Tensor) for entry in entries)
        has_image_sizes = any(isinstance(entry.image_sizes, torch.Tensor) for entry in entries)
        if (has_pixel_values and replay_pixel_values is None) or (has_image_sizes and replay_image_sizes is None):
            self._metrics[mode].setdefault("replay/rollout_skipped_rate", []).append(1.0)
            self._metrics[mode].setdefault("replay/rollout_batch_size", []).append(0.0)
            self._metrics[mode].setdefault("replay/rollout_skipped_incompatible_vision", []).append(1.0)
            return None

        replay_logps = self._get_per_token_logps(
            model,
            replay_input_ids,
            replay_attention_mask,
            replay_pixel_values,
            replay_image_sizes,
            replay_completion_ids.size(1),
        )
        coef_1 = torch.exp(replay_logps - replay_old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(
            coef_1 * replay_advantages.unsqueeze(1),
            coef_2 * replay_advantages.unsqueeze(1),
        )
        replay_loss = (per_token_loss * replay_completion_mask).sum() / replay_completion_mask.sum().clamp(min=1.0)
        self._metrics[mode].setdefault("loss/rollout_replay", []).append(float(replay_loss.detach().item()))
        self._metrics[mode].setdefault("replay/rollout_skipped_rate", []).append(0.0)
        self._metrics[mode].setdefault("replay/rollout_batch_size", []).append(float(len(entries)))
        self._metrics[mode].setdefault("replay/rollout_tokens", []).append(
            float(replay_completion_mask.detach().float().sum().item())
        )
        self._metrics[mode].setdefault("replay/rollout_advantage_mean", []).append(
            float(replay_advantages.detach().float().mean().item())
        )
        return replay_loss, float(cfg.weight), len(entries)

    def _update_rollout_replay_buffer(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        advantages: torch.Tensor,
        acc_rewards: torch.Tensor | None,
        global_step: int,
        pixel_values: torch.Tensor | None,
        image_sizes: torch.Tensor | None,
    ) -> None:
        mode = "train" if self.model.training else "eval"
        buffer = self._rollout_replay_buffer
        if mode != "train" or buffer is None or not buffer.available:
            return
        stats = buffer.add_batch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            acc_rewards=acc_rewards,
            global_step=global_step,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        self._metrics[mode].setdefault("replay/rollout_buffer_size", []).append(float(len(buffer)))
        self._metrics[mode].setdefault("replay/rollout_added", []).append(float(stats.added))
        self._metrics[mode].setdefault("replay/rollout_skipped_not_positive", []).append(
            float(stats.skipped_not_positive)
        )
        self._metrics[mode].setdefault("replay/rollout_skipped_low_advantage", []).append(
            float(stats.skipped_low_advantage)
        )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.training_stage == "opd_only":
            return self._compute_loss(model, inputs)
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_opd_only_loss(self, model, inputs):
        """Compute an isolated OPD objective for the explicit opd_only stage.

        This path intentionally never constructs a GRPO or SFT objective.  The
        rollout builder marks every completion as an OPD sample and supplies a
        teacher prompt for every row.  ``acc_gate`` is disabled here so reward
        correctness cannot silently remove gradients from otherwise valid
        student states.
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        pixel_values = inputs.get("pixel_values")
        image_sizes = inputs.get("img_sizes")
        opsd_mask = inputs.get("opsd_mask")
        if opsd_mask is None:
            opsd_mask = torch.ones(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
        opsd_indices = [i for i, flag in enumerate(opsd_mask.detach().bool().cpu().tolist()) if flag]
        global_step = getattr(self.state, "global_step", self._step)
        cfg = self.opsd_config.get("loss", {}) or {}
        opsd_weight = float(cfg.get("opsd_weight", 1.0) or 0.0)
        beta = float(cfg.get("beta", 0.5) or 0.5)
        loss_type = str(cfg.get("loss_type", "jsd") or "jsd")
        srkl_alpha = float(cfg.get("srkl_alpha", 0.1) or 0.1)
        if self.teacher_model is None:
            raise RuntimeError("opd_only requires a loaded frozen teacher model")

        if float(cfg.get("grpo_weight", 0.0) or 0.0) != 0.0:
            raise RuntimeError("opd_only invariant violated: grpo_weight must be zero")
        if float(cfg.get("sft_weight", 0.0) or 0.0) != 0.0:
            raise RuntimeError("opd_only invariant violated: sft_weight must be zero")
        if cfg.get("acc_gate") is not False:
            raise RuntimeError("opd_only invariant violated: acc_gate must be false")

        # One student forward provides logits with gradient for OPD.  The
        # selective log-probability result is deliberately ignored.
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        _, student_completion_logits = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
            completion_ids.size(1),
            return_completion_logits=True,
        )
        if not opsd_indices:
            opd_loss_tensor = student_completion_logits.sum() * 0.0
        else:
            opd_loss_tensor = compute_vlm_opsd_loss_masked_batch(
                model,
                opsd_indices,
                list(range(prompt_ids.size(0))),
                inputs,
                beta=beta,
                processor=self.processing_class,
                teacher_model=self.teacher_model,
                acc_gate=False,
                global_step=global_step,
                tokenizer=self.processing_class.tokenizer,
                student_completion_logits=student_completion_logits,
                loss_type=loss_type,
                srkl_alpha=srkl_alpha,
            )
            opd_loss_tensor = opd_loss_tensor * opsd_weight
        loss = opd_loss_tensor

        # Optional teacher trajectory supervision is an auxiliary term, not a
        # routing decision.  Probe-generated trajectories are attached to the
        # same full batch; only rows with valid trajectories contribute.
        traj_cfg = self._teacher_trajectory_config()
        traj_mask = inputs.get("teacher_traj_mask")
        traj_loss = None
        traj_indices: list[int] = []
        if traj_cfg["enabled"] and isinstance(traj_mask, torch.Tensor):
            traj_indices = [
                i for i, flag in enumerate(traj_mask.detach().bool().cpu().tolist()) if flag
            ]
        if traj_indices:
            traj_inputs = dict(inputs)
            traj_inputs["completion_ids"] = inputs["teacher_traj_completion_ids"]
            traj_inputs["completion_mask"] = inputs["teacher_traj_completion_mask"]
            traj_loss = compute_vlm_opsd_loss_masked_batch(
                model,
                traj_indices,
                list(range(prompt_ids.size(0))),
                traj_inputs,
                beta=beta,
                processor=self.processing_class,
                teacher_model=self.teacher_model,
                acc_gate=False,
                global_step=global_step,
                tokenizer=self.processing_class.tokenizer,
                loss_type=traj_cfg["loss_type"],
                srkl_alpha=srkl_alpha,
            )
            loss = loss + float(traj_cfg["weight"]) * traj_loss

        if opsd_indices and opsd_debug.should_log_detail(global_step):
            # The cache is also useful for SRKL: it records aligned logits
            # already produced by the OPD forward and never launches another
            # student/teacher pass.
            opsd_diagnostics.log_opsd_jsd_diagnostics(global_step=global_step)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("loss/opsd", []).append(float(opd_loss_tensor.detach().item()))
        self._metrics[mode].setdefault("loss/combined", []).append(float(loss.detach().item()))
        self._metrics[mode].setdefault("loss/grpo", []).append(0.0)
        self._metrics[mode].setdefault("loss/sft", []).append(0.0)
        self._metrics[mode].setdefault("loss/teacher_traj_fkl", []).append(0.0)
        self._metrics[mode]["loss/teacher_traj_fkl"][-1] = float(
            traj_loss.detach().item() if traj_loss is not None else 0.0
        )
        self._metrics[mode].setdefault("loss/teacher_traj_effective_weight", []).append(
            float(traj_cfg["weight"]) if traj_loss is not None else 0.0
        )
        self._metrics[mode].setdefault("teacher_traj/rows", []).append(float(len(traj_indices)))
        self._metrics[mode].setdefault("routing/opd_only", []).append(1.0)
        self._metrics[mode].setdefault("routing/opd_route_count", []).append(float(len(opsd_indices)))
        self._metrics[mode].setdefault("routing/grpo_route_count", []).append(0.0)
        self._metrics[mode].setdefault("routing/sft_route_count", []).append(0.0)
        if self._health_monitor is not None:
            self._health_monitor.record_loss(
                global_step,
                {
                    "training_stage": "opd_only",
                    "combined_loss_scalar": float(loss.detach().item()),
                    "grpo_loss_scalar": 0.0,
                    "grpo_zero_loss_rate": 1.0,
                    "advantages_abs_mean": 0.0,
                    "opsd_loss_scalar": float(opd_loss_tensor.detach().item()),
                    "opd_loss_scalar": float(opd_loss_tensor.detach().item()),
                    "teacher_traj_loss_scalar": float(traj_loss.detach().item()) if traj_loss is not None else 0.0,
                },
            )
        return loss

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        if self.training_stage == "opd_only":
            return self._compute_opd_only_loss(model, inputs)
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["img_sizes"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        if inputs.get("sft_cold_start"):
            per_token_logps = self._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                pixel_values,
                image_sizes,
                logits_to_keep,
            )
            sft_loss = -(per_token_logps * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            mode = "train" if self.model.training else "eval"
            self._metrics[mode].setdefault("loss/sft", []).append(float(sft_loss.detach().item()))
            self._metrics[mode].setdefault("phase/sft_cold_start", []).append(1.0)
            global_step = getattr(self.state, "global_step", self._step)
            if self._health_monitor is not None:
                self._health_monitor.record_loss(
                    global_step,
                    {
                        "combined_loss_scalar": float(sft_loss.detach().item()),
                        "grpo_loss_scalar": 0.0,
                        "grpo_zero_loss_rate": 1.0,
                        "advantages_abs_mean": 0.0,
                    },
                )
            return sft_loss

        # Compute the per-token log probabilities for the model
        opsd_active = (
            self.opsd_config.get("enabled", False)
            and inputs.get("opsd_mask") is not None
            and self.teacher_model is not None
        )
        local_opsd_for_cache = 0
        if opsd_active and inputs.get("opsd_mask") is not None:
            local_opsd_for_cache = int(inputs["opsd_mask"].sum().item())
        cache_logits_for_opsd = (
            opsd_active
            and deepspeed_requires_single_student_forward()
            and local_opsd_for_cache > 0
        )
        opsd_debug.hang_probe(
            "student_forward_start",
            opsd_active=opsd_active,
            cache_logits_for_opsd=cache_logits_for_opsd,
            completion_tokens=int(completion_mask.sum().item()),
            batch_size=int(prompt_ids.shape[0]),
        )
        try:
            if cache_logits_for_opsd:
                per_token_logps, student_completion_logits = self._get_per_token_logps(
                    model,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_sizes,
                    logits_to_keep,
                    return_completion_logits=True,
                )
            else:
                per_token_logps = self._get_per_token_logps(
                    model, input_ids, attention_mask, pixel_values, image_sizes, logits_to_keep
                )
                student_completion_logits = None
        except Exception as e:
            print(f"Error in _get_per_token_logps: {e}")
            raise e
        opsd_debug.hang_probe("student_forward_done", cache_logits_for_opsd=cache_logits_for_opsd)

        # sft_loss = -(per_token_logps * completion_mask).sum(-1) / completion_mask.sum(-1)
        advantages = inputs["advantages"][:, 0]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        # loss = (has_correct > 0) * loss + sft_loss
        # loss = (has_correct > 0) * loss
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        global_step = getattr(self.state, "global_step", self._step)
        max_training_steps = self._max_training_steps()
        phase_mode = str((self.opsd_config.get("phase_schedule") or {}).get("mode", "step") or "step")
        opsd_debug.set_detail_step(global_step)
        grpo_loss_tensor = loss.detach()
        opsd_loss_tensor = None
        combined_loss_tensor = None
        opsd_loss_cfg = self.opsd_config.get("loss", {})
        base_opsd_weight = opsd_loss_cfg.get("opsd_weight", 1.0)
        opsd_decay_cfg = opsd_loss_cfg.get("weight_decay") or {}
        traj_cfg = self._teacher_trajectory_config()
        traj_decay_cfg = traj_cfg.get("weight_decay", {})
        scheduled_opsd_weight = effective_linear_weight(
            base_weight=base_opsd_weight,
            global_step=global_step,
            decay_enabled=bool(opsd_decay_cfg.get("enabled", False)),
            decay_start_step=int(opsd_decay_cfg.get("start_step", 294) or 294),
            decay_end_step=int(opsd_decay_cfg.get("end_step", 441) or 441),
            final_weight=float(opsd_decay_cfg.get("final_weight", base_opsd_weight) or base_opsd_weight),
            max_steps=max_training_steps,
            schedule_mode=phase_mode,
            decay_start_progress=float(opsd_decay_cfg.get("start_progress", 0.50)),
            decay_end_progress=float(opsd_decay_cfg.get("end_progress", 0.75)),
        )
        adaptive_weights = self._adaptive_loss_weights()
        if adaptive_weights is not None:
            scheduled_opsd_weight = adaptive_weights[0]
        reward_std_mean_value = inputs.get("reward_std_mean")
        if isinstance(reward_std_mean_value, torch.Tensor):
            reward_std_mean_scalar = float(reward_std_mean_value.detach().float().mean().item())
        elif reward_std_mean_value is None:
            reward_std_mean_scalar = float(opsd_loss_cfg.get("adaptive_std_target", 0.25))
        else:
            reward_std_mean_scalar = float(reward_std_mean_value)
        opsd_effective_weight, opsd_adaptive_multiplier = effective_opsd_weight(
            scheduled_opsd_weight,
            reward_std_mean_scalar,
            enabled=bool(opsd_loss_cfg.get("variance_adaptive", False)),
            std_target=opsd_loss_cfg.get("adaptive_std_target", 0.25),
            max_mult=opsd_loss_cfg.get("adaptive_max_mult", 2.0),
        )

        if self.opsd_config.get("enabled", False) and inputs.get("opsd_mask") is not None:
            opsd_mask = inputs["opsd_mask"]
            opsd_debug.log(
                "opsd_loss",
                "enter OPSD loss branch in compute_loss",
                opsd_mask_true=int(opsd_mask.sum().item()),
                batch_size=prompt_ids.size(0),
            )
            beta = opsd_loss_cfg.get("beta", 0.5)
            opsd_weight = scheduled_opsd_weight
            grpo_weight = opsd_loss_cfg.get("grpo_weight", 1.0)
            opsd_loss_type = opsd_loss_cfg.get("loss_type", "jsd")
            srkl_alpha = opsd_loss_cfg.get("srkl_alpha", 0.1)
            opsd_indices: list[int] = []
            if opsd_mask.any():
                opsd_indices = opsd_mask.nonzero(as_tuple=True)[0].tolist()

            local_opsd_count = len(opsd_indices)
            global_opsd_count = sync_global_sum_count(
                local_opsd_count,
                loss.device,
                self.accelerator.num_processes,
            )
            opsd_debug.hang_probe_force(
                "opsd_branch_enter",
                local_opsd_count=local_opsd_count,
                global_opsd_count=global_opsd_count,
                opsd_indices=opsd_indices,
                has_teacher_prompt=inputs.get("teacher_prompt_ids") is not None,
            )

            if global_opsd_count > 0 and local_opsd_count > 0:
                opsd_debug.log(
                    "opsd_loss",
                    "compute_vlm_opsd_loss_masked_batch args",
                    opsd_indices=opsd_indices,
                    beta=beta,
                    loss_type=opsd_loss_type,
                    srkl_alpha=srkl_alpha,
                    opsd_weight=opsd_weight,
                    effective_opsd_weight=opsd_effective_weight,
                    opsd_adaptive_multiplier=opsd_adaptive_multiplier,
                    reward_std_mean=reward_std_mean_scalar,
                    grpo_weight=grpo_weight,
                    grpo_loss=float(loss.detach().item()),
                    local_opsd_count=local_opsd_count,
                )
                opsd_debug.hang_probe("opsd_loss_compute_start", local_opsd_count=local_opsd_count)
                with opsd_debug.timed("opsd_loss", "compute_vlm_opsd_loss_masked_batch"):
                    acc_gate = self.opsd_config.get("loss", {}).get("acc_gate", True)
                    opsd_loss = compute_vlm_opsd_loss_masked_batch(
                        model,
                        opsd_indices,
                        list(range(prompt_ids.size(0))),
                        inputs,
                        beta=beta,
                        processor=self.processing_class,
                        teacher_model=self.teacher_model,
                        acc_gate=acc_gate,
                        global_step=global_step,
                        tokenizer=self.processing_class.tokenizer,
                        student_completion_logits=student_completion_logits,
                        loss_type=opsd_loss_type,
                        srkl_alpha=srkl_alpha,
                    )
                opsd_debug.hang_probe(
                    "opsd_loss_compute_done",
                    opsd_loss=float(opsd_loss.detach().item()),
                )
                opsd_loss_tensor = opsd_loss
                loss = grpo_weight * loss + opsd_effective_weight * opsd_loss
                combined_loss_tensor = loss
                opsd_debug.log(
                    "opsd_loss",
                    "combined GRPO + OPSD loss",
                    opsd_loss=float(opsd_loss.detach().item()),
                    loss_type=opsd_loss_type,
                    base_opsd_weight=base_opsd_weight,
                    scheduled_opsd_weight=scheduled_opsd_weight,
                    effective_opsd_weight=opsd_effective_weight,
                    opsd_adaptive_multiplier=opsd_adaptive_multiplier,
                    reward_std_mean=reward_std_mean_scalar,
                    combined_loss=float(loss.detach().item()),
                )
            else:
                if global_opsd_count == 0:
                    opsd_debug.log("opsd_loss", "no OPSD samples on any rank, skip teacher/OPSD")
                else:
                    opsd_debug.log("opsd_loss", "no local OPSD samples, skip teacher/OPSD on this rank")
                opsd_debug.hang_probe_force(
                    "opsd_loss_skip_local",
                    local_opsd_count=local_opsd_count,
                    global_opsd_count=global_opsd_count,
                )

            # Every rank must enter this collective; barrier keeps ranks aligned first.
            opsd_metric_value = (
                opsd_loss_tensor.detach()
                if opsd_loss_tensor is not None
                else torch.zeros((), device=loss.device, dtype=loss.dtype)
            )
            opsd_debug.hang_probe("barrier_before_gather_opsd_loss", local_opsd_count=local_opsd_count)
            self._opsd_distributed_barrier("wait_for_everyone before gather_for_metrics(opsd_loss)")
            opsd_debug.hang_probe("gather_opsd_loss_start")
            opsd_debug.log_sync_point("dist", "before gather_for_metrics(opsd_loss)")
            self._metrics[mode].setdefault("loss/opsd", []).append(
                self.accelerator.gather_for_metrics(opsd_metric_value).mean().item()
            )
            self._metrics[mode].setdefault("signal/reward_std_mean", []).append(reward_std_mean_scalar)
            self._metrics[mode].setdefault("loss/opsd_scheduled_base_weight", []).append(scheduled_opsd_weight)
            self._metrics[mode].setdefault("loss/opsd_effective_weight", []).append(opsd_effective_weight)
            self._metrics[mode].setdefault("loss/opsd_adaptive_multiplier", []).append(
                opsd_adaptive_multiplier
            )
            opsd_debug.hang_probe("gather_opsd_loss_done")
            teacher_traj_effective_weight = effective_teacher_traj_weight(
                base_weight=traj_cfg["weight"],
                global_step=global_step,
                decay_enabled=bool(traj_decay_cfg.get("enabled", False)),
                decay_start_step=int(traj_decay_cfg.get("start_step", 294)),
                decay_end_step=int(traj_decay_cfg.get("end_step", 441)),
                final_weight=float(traj_decay_cfg.get("final_weight", 0.0)),
                max_steps=max_training_steps,
                schedule_mode=phase_mode,
                decay_start_progress=float(traj_decay_cfg.get("start_progress", 0.25)),
                decay_end_progress=float(traj_decay_cfg.get("end_progress", 0.50)),
            )
            if adaptive_weights is not None:
                teacher_traj_effective_weight = adaptive_weights[1]
            self._metrics[mode].setdefault("loss/teacher_traj_effective_weight", []).append(
                teacher_traj_effective_weight
            )
            if traj_cfg["enabled"] and inputs.get("teacher_traj_mask") is not None:
                teacher_traj_mask = inputs["teacher_traj_mask"]
                teacher_traj_indices = local_teacher_traj_indices(
                    teacher_traj_mask=teacher_traj_mask.detach().bool().cpu().tolist(),
                    has_teacher_prompt_ids=inputs.get("teacher_prompt_ids") is not None,
                )
                local_traj_count = len(teacher_traj_indices)
                global_traj_count = sync_global_sum_count(
                    local_traj_count,
                    loss.device,
                    self.accelerator.num_processes,
                )
                opsd_debug.hang_probe_force(
                    "teacher_traj_branch",
                    local_traj_count=local_traj_count,
                    global_traj_count=global_traj_count,
                )
                teacher_traj_loss_tensor = None
                if local_traj_count > 0:
                    traj_inputs = dict(inputs)
                    traj_inputs["completion_ids"] = inputs["teacher_traj_completion_ids"]
                    traj_inputs["completion_mask"] = inputs["teacher_traj_completion_mask"]
                    opsd_debug.hang_probe("teacher_traj_compute_start", local_traj_count=local_traj_count)
                    teacher_traj_start = self._perf_start()
                    with opsd_debug.timed("opsd_loss", "compute_teacher_traj_fkl_loss"):
                        teacher_traj_loss_tensor = compute_vlm_opsd_loss_masked_batch(
                            model,
                            teacher_traj_indices,
                            list(range(prompt_ids.size(0))),
                            traj_inputs,
                            beta=beta,
                            processor=self.processing_class,
                            teacher_model=self.teacher_model,
                            acc_gate=False,
                            global_step=global_step,
                            tokenizer=self.processing_class.tokenizer,
                            loss_type=traj_cfg["loss_type"],
                            srkl_alpha=srkl_alpha,
                        )
                    self._perf_metric(mode, "teacher_traj_loss_s", self._perf_elapsed(teacher_traj_start))
                    opsd_debug.hang_probe(
                        "teacher_traj_compute_done",
                        teacher_traj_loss=float(teacher_traj_loss_tensor.detach().item()),
                    )
                    loss = loss + teacher_traj_effective_weight * teacher_traj_loss_tensor
                    combined_loss_tensor = loss
                    opsd_debug.log(
                        "opsd_loss",
                        "added teacher trajectory distillation loss",
                        teacher_traj_loss=float(teacher_traj_loss_tensor.detach().item()),
                        teacher_traj_weight=teacher_traj_effective_weight,
                        teacher_traj_base_weight=traj_cfg["weight"],
                        teacher_traj_loss_type=traj_cfg["loss_type"],
                        combined_loss=float(loss.detach().item()),
                    )
                elif global_traj_count > 0:
                    opsd_debug.log(
                        "opsd_loss",
                        "no local teacher traj on this rank; wait for traj sync",
                        global_traj_count=global_traj_count,
                    )
                metric_value = (
                    teacher_traj_loss_tensor.detach()
                    if teacher_traj_loss_tensor is not None
                    else torch.zeros((), device=loss.device, dtype=loss.dtype)
                )
                opsd_debug.hang_probe("barrier_before_gather_teacher_traj", local_traj_count=local_traj_count)
                self._opsd_distributed_barrier("wait_for_everyone before gather_for_metrics(teacher_traj_fkl)")
                opsd_debug.hang_probe("gather_teacher_traj_start")
                self._metrics[mode].setdefault("loss/teacher_traj_fkl", []).append(
                    self.accelerator.gather_for_metrics(metric_value).mean().item()
                )
                opsd_debug.hang_probe("gather_teacher_traj_done")
            if opsd_indices and opsd_debug.should_log_detail(global_step):
                opsd_diagnostics.log_opsd_jsd_diagnostics(global_step=global_step)
            self._opsd_distributed_barrier("wait_for_everyone after OPSD compute_loss")

        positive_replay = self._compute_positive_replay_loss(model, global_step=global_step)
        if positive_replay is not None:
            positive_replay_loss, positive_replay_weight, positive_replay_batch = positive_replay
            loss = loss + positive_replay_weight * positive_replay_loss
            combined_loss_tensor = loss
            opsd_debug.log(
                "positive_replay",
                "added positive replay auxiliary CE loss",
                replay_loss=float(positive_replay_loss.detach().item()),
                replay_weight=positive_replay_weight,
                replay_batch=positive_replay_batch,
                combined_loss=float(loss.detach().item()),
            )

        rollout_replay = self._compute_rollout_replay_loss(model, global_step=global_step)
        if rollout_replay is not None:
            rollout_replay_loss, rollout_replay_weight, rollout_replay_batch = rollout_replay
            loss = loss + rollout_replay_weight * rollout_replay_loss
            combined_loss_tensor = loss
            opsd_debug.log(
                "rollout_replay",
                "added rollout replay clipped PG loss",
                replay_loss=float(rollout_replay_loss.detach().item()),
                replay_weight=rollout_replay_weight,
                replay_batch=rollout_replay_batch,
                combined_loss=float(loss.detach().item()),
            )

        self._update_rollout_replay_buffer(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_per_token_logps=old_per_token_logps.detach(),
            advantages=advantages.detach().view(-1, 1),
            acc_rewards=inputs.get("acc_rewards"),
            global_step=global_step,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        opsd_diagnostics.log_loss_diagnostics(
            global_step=global_step,
            grpo_loss=grpo_loss_tensor,
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            completion_mask=completion_mask,
            advantages=advantages,
            coef_1=coef_1,
            per_token_loss=per_token_loss,
            opsd_loss=opsd_loss_tensor,
            combined_loss=combined_loss_tensor if combined_loss_tensor is not None else loss,
            opsd_mask=inputs.get("opsd_mask"),
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
        )

        if self._health_monitor is not None:
            loss_health = opsd_diagnostics.summarize_loss_health(
                grpo_loss=grpo_loss_tensor,
                per_token_logps=per_token_logps,
                completion_mask=completion_mask,
                advantages=advantages,
                per_token_loss=per_token_loss,
                opsd_loss=opsd_loss_tensor,
                combined_loss=combined_loss_tensor if combined_loss_tensor is not None else loss,
            )
            loss_health.update(
                {
                    "reward_std_mean": reward_std_mean_scalar,
                    "opsd_effective_weight": opsd_effective_weight,
                    "opsd_adaptive_multiplier": opsd_adaptive_multiplier,
                }
            )
            self._health_monitor.record_loss(global_step, loss_health)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _checkpoint_evaluation_preflight_should_stop(
        self, resume_from_checkpoint: Optional[Union[str, bool]]
    ) -> bool:
        """Restore a terminal checkpoint-eval sidecar before HF enters its loop.

        Transformers restores callback state and calls ``on_train_begin`` from
        inside ``Trainer.train``.  In the current HF lifecycle that is not an
        early enough point to guarantee that a resumed terminal run will not
        reach an optimizer step.  The checkpoint-eval sidecar is the durable
        source for terminal decisions (the terminal comparison intentionally
        creates no native checkpoint), so inspect it before delegating to the
        parent training loop.

        Every distributed rank performs the same call.  Only rank zero reads
        the sidecar; the callback broadcasts either the restored policy or its
        rank-zero load error, so corrupted metadata cannot strand workers in a
        later collective.
        """
        if not resume_from_checkpoint or not self.checkpoint_eval_config.get("enabled", False):
            return False

        callback = next(
            (
                candidate
                for candidate in getattr(getattr(self, "callback_handler", None), "callbacks", ())
                if isinstance(candidate, CheckpointEvaluationTriggerCallback)
            ),
            None,
        )
        if callback is None:
            return False

        # ``on_train_begin`` normally initializes these callback fields, but
        # this preflight deliberately runs before that HF hook.  Use the live
        # Trainer rank rather than the stale/default TrainerState value.
        callback._set_output_dir_from_training_args(self.args)
        callback._is_world_process_zero = bool(self.is_world_process_zero())
        callback._load_and_broadcast_policy_state_from_world_zero()
        callback._policy_stop_requested = bool(callback.policy.state.stop_requested)

        # HF can replace callback instances during a normal resume.  Bind the
        # active instance now as well, so terminal reporting and the main
        # entrypoint see the recovered best score even though no later save
        # event will occur to rebind it.
        self.checkpoint_eval_policy = callback.policy
        self.checkpoint_eval_state = callback.policy.state
        policy_state = callback.policy.state
        self.best_checkpoint_score = policy_state.best_score
        self.best_checkpoint_step = policy_state.best_step
        self.best_checkpoint_path = (
            os.path.join(self.args.output_dir, f"checkpoint-{policy_state.best_step}")
            if policy_state.best_step is not None
            else None
        )
        return bool(policy_state.stop_requested)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> TrainOutput:
        """Train normally, unless a resumed checkpoint already exhausted patience."""
        normalized_resume_checkpoint = resume_from_checkpoint
        if normalized_resume_checkpoint is True:
            # Match Trainer.train(resume_from_checkpoint=True): resolve the
            # retained native checkpoint before reading its output sidecar.
            # Import locally so the surrounding trainer module stays aligned
            # with the project's existing lightweight import surface.
            from transformers.trainer_utils import get_last_checkpoint

            normalized_resume_checkpoint = get_last_checkpoint(self.args.output_dir)
            if normalized_resume_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )
        elif normalized_resume_checkpoint is False:
            normalized_resume_checkpoint = None

        if self._checkpoint_evaluation_preflight_should_stop(normalized_resume_checkpoint):
            policy_state = self.checkpoint_eval_policy.state
            # The retained checkpoint is the last native checkpoint for this
            # policy.  Returning its step gives callers a meaningful, stable
            # TrainOutput without entering HF's model/optimizer lifecycle.
            global_step = int(
                policy_state.best_step
                if policy_state.best_step is not None
                else getattr(self.state, "global_step", 0)
            )
            metrics = {
                "train_loss": 0.0,
                "checkpoint_eval_terminal_resume": 1.0,
                "checkpoint_eval_best_score": float(policy_state.best_score),
                "checkpoint_eval_best_step": float(policy_state.best_step),
                "checkpoint_eval_lower_score_streak": float(policy_state.lower_score_streak),
                "checkpoint_eval_evaluation_count": float(policy_state.evaluation_count),
            }
            return TrainOutput(global_step, 0.0, metrics)
        return super().train(
            resume_from_checkpoint=normalized_resume_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def _consume_checkpoint_evaluation_request(self) -> bool:
        """Consume the save-trigger flag installed by the checkpoint callback."""
        callback_handler = getattr(self, "callback_handler", None)
        for callback in getattr(callback_handler, "callbacks", ()):
            if isinstance(callback, CheckpointEvaluationTriggerCallback):
                if callback.consume_checkpoint_evaluation():
                    # HF can replace exportable callback instances while
                    # restoring a checkpoint.  Rebind on consumption so the
                    # resumed policy state, not the constructor's empty one,
                    # governs the next comparison.
                    self.checkpoint_eval_policy = callback.policy
                    self.checkpoint_eval_state = callback.policy.state
                    return True
        return False

    def _refresh_checkpoint_evaluation_callback_state(self, *, persist_sidecar: bool) -> None:
        """Refresh callback state; persist only decisions with no native save.

        A new-best result is included in native Trainer checkpoint state during
        ``_save_checkpoint``.  Writing its sidecar before that point can make
        a crash recover an unwritten best step.  Lower/tied results veto
        native saving, so their streak must instead be persisted now.
        """
        stateful_callbacks = getattr(self.state, "stateful_callbacks", None)
        for callback in getattr(getattr(self, "callback_handler", None), "callbacks", ()):
            if isinstance(callback, CheckpointEvaluationTriggerCallback):
                if isinstance(stateful_callbacks, dict):
                    stateful_callbacks[callback.__class__.__name__] = callback.state()
                if persist_sidecar:
                    # Lower-scoring evaluations do not create a native Trainer
                    # checkpoint, so persist the updated streak/best state to
                    # the callback sidecar as well.  All ranks enter this
                    # branch, but only global rank zero may write it.  The
                    # callback broadcasts a write failure before raising, so
                    # a filesystem error cannot leave workers advancing toward
                    # a later collective while rank zero exits.
                    # ``on_train_begin`` normally initializes this cache. Set
                    # it from the live accelerator here too so this helper
                    # remains correct for direct/atypical Trainer lifecycles.
                    callback._is_world_process_zero = bool(
                        getattr(self.accelerator, "is_main_process", True)
                    )
                    callback._run_on_rank_zero_and_broadcast_failure(
                        operation="non-saving checkpoint evaluation sidecar persistence",
                        action=lambda: callback.persist(is_world_process_zero=True),
                    )
                return

    @contextmanager
    def _checkpoint_evaluation_generation_context(self):
        """Expose the live, distributed student safely for ChartQA generation."""
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                yield unwrapped_model

    def _broadcast_checkpoint_evaluation_payload(
        self,
        payload: Optional[dict[str, Any]],
        *,
        rank_zero_error: Exception | None = None,
    ) -> dict[str, Any]:
        """Share a rank-zero policy result, including an observation failure.

        The policy is intentionally mutated only on global rank zero.  A
        malformed score or terminal-state mismatch can make ``observe`` raise;
        it must still enter the one evaluation-result broadcast so workers do
        not hang waiting for a payload that rank zero never sends.
        """
        is_main = bool(getattr(self.accelerator, "is_main_process", True))
        objects: list[Optional[dict[str, Any]]] = [None]
        if is_main:
            if rank_zero_error is None:
                objects[0] = {"ok": True, "payload": payload}
            else:
                objects[0] = {
                    "ok": False,
                    "error_type": type(rank_zero_error).__name__,
                    "error": str(rank_zero_error),
                }
        if int(getattr(self.accelerator, "num_processes", 1)) > 1:
            broadcast_object_list(objects, from_process=0)
        outcome = objects[0]
        if not isinstance(outcome, dict) or not isinstance(outcome.get("ok"), bool):
            raise RuntimeError("checkpoint evaluation did not receive a valid rank-zero outcome")
        if not outcome["ok"]:
            error_type = outcome.get("error_type", "RuntimeError")
            detail = outcome.get("error", "unknown policy observation error")
            message = (
                "checkpoint evaluation policy observation failed on global rank zero "
                f"({error_type}): {detail}"
            )
            if rank_zero_error is not None:
                raise RuntimeError(message) from rank_zero_error
            raise RuntimeError(message)
        result = outcome.get("payload")
        if not isinstance(result, dict):
            raise RuntimeError("checkpoint evaluation did not receive a valid rank-zero decision")
        return result

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate the resident student for checkpoint selection on ChartQA.

        The normal GRPO ``prediction_step`` computes rewards and may invoke the
        teacher/refiner.  It is therefore deliberately bypassed here: this path
        runs greedy ChartQA inference through the existing GPU model, on every
        rank, and evaluates the exact global accuracy only once per save event.
        """
        if self.checkpoint_eval_policy is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                **kwargs,
            )

        # Only the callback-transformed native save event participates in
        # best-checkpoint selection.  A separately configured Trainer eval
        # cadence must neither consume patience nor overwrite the score state.
        save_time_evaluation = self._consume_checkpoint_evaluation_request()
        if not save_time_evaluation:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                **kwargs,
            )

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(dataset, dict):
            if len(dataset) != 1:
                raise ValueError(
                    "checkpoint_eval requires one ChartQA evaluation dataset, not a split mapping"
                )
            dataset = next(iter(dataset.values()))
        if dataset is None:
            raise ValueError("checkpoint_eval requires an evaluation dataset")

        eval_config = ChartQAEvaluationConfig(
            batch_size=int(self.checkpoint_eval_config.get("batch_size", 1)),
            max_new_tokens=int(self.checkpoint_eval_config.get("max_new_tokens", 1024)),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=float(self.checkpoint_eval_config.get("repetition_penalty", 1.0)),
            device=getattr(self.accelerator, "device", None),
            synced_gpus=self.checkpoint_eval_config.get("synced_gpus"),
        )
        result = evaluate_chartqa_in_memory(
            model=self.model,
            processor=self.processing_class,
            accelerator=self.accelerator,
            dataset=dataset,
            generation_context=self._checkpoint_evaluation_generation_context,
            config=eval_config,
        )
        score = float(result["checkpoint_score"])
        step = int(self.state.global_step)

        # The evaluator all-reduces accuracy, but only rank zero mutates the
        # retention policy.  Broadcasting its complete state prevents a later
        # save/early-stop branch from diverging across distributed workers.
        payload: Optional[dict[str, Any]] = None
        rank_zero_error: Exception | None = None
        if getattr(self.accelerator, "is_main_process", True):
            try:
                decision = self.checkpoint_eval_policy.observe(score, step=step)
                payload = {
                    "decision": decision.to_dict(),
                    "policy_state": self.checkpoint_eval_policy.state_dict(),
                }
            except Exception as exc:
                # Do not re-raise before the collective below: all workers
                # must consume this failure rather than wait for a decision.
                rank_zero_error = exc
        payload = self._broadcast_checkpoint_evaluation_payload(
            payload,
            rank_zero_error=rank_zero_error,
        )
        self.checkpoint_eval_policy.load_state_dict(payload["policy_state"])
        self.checkpoint_eval_state = self.checkpoint_eval_policy.state
        decision = payload["decision"]

        # Let the callback distinguish a no-save result that must be made
        # durable immediately from an improving score that must first pass
        # native Trainer checkpoint writing and rotation.
        for callback in getattr(getattr(self, "callback_handler", None), "callbacks", ()):
            if isinstance(callback, CheckpointEvaluationTriggerCallback):
                callback.record_checkpoint_evaluation_decision(decision)
                break
        self._refresh_checkpoint_evaluation_callback_state(
            persist_sidecar=not bool(decision["should_save"])
        )

        self.best_checkpoint_score = float(decision["best_score"])
        self.best_checkpoint_step = int(decision["best_step"])
        if bool(decision["should_save"]):
            self.best_checkpoint_path = os.path.join(
                self.args.output_dir, f"checkpoint-{self.best_checkpoint_step}"
            )

        control = self.control
        # The callback has converted the native save event into evaluation.
        # Restore it only for the initial/improved result; all other scores
        # retain the previously written checkpoint.
        apply_checkpoint_evaluation_decision(control, decision)

        prefix = f"{metric_key_prefix}_"
        metrics: dict[str, float] = {
            f"{prefix}checkpoint_score": score,
            f"{prefix}chartqa_accuracy": float(result["chartqa_accuracy"]),
            f"{prefix}checkpoint_best_score": self.best_checkpoint_score,
            f"{prefix}checkpoint_lower_score_streak": float(decision["lower_score_streak"]),
            f"{prefix}checkpoint_evaluation_count": float(decision["evaluation_count"]),
            f"{prefix}checkpoint_sample_count": float(result["sample_count"]),
        }
        # Call the base implementation directly.  DyMETrainer.log aggregates
        # and clears GRPO metric buffers, which is wrong for this independent
        # checkpoint score event.
        Trainer.log(self, metrics)
        return metrics

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        if mode == "train" and self._perf_timing_enabled and self._perf_step_start_s is not None:
            self._perf_metric(mode, "step_wall_s", self._perf_elapsed(self._perf_step_start_s))
            self._perf_step_start_s = None
        if self._health_monitor is not None and mode == "train" and self.accelerator.is_main_process:
            step = getattr(self.state, "global_step", self._step)
            self._health_monitor.record_optimizer(
                step,
                logs.get("grad_norm"),
                logs.get("learning_rate"),
            )
            health_metrics = self._health_monitor.finish_step(step)
            self._record_dynamic_trigger_metrics(
                mode=mode,
                global_step=int(step),
                health_metrics=health_metrics,
            )
            for key, value in health_metrics.items():
                self._metrics[mode].setdefault(key, []).append(value)
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})
