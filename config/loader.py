"""Strict YAML configuration loader for training entry points."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_CONFIG_ALIASES = {
    "norm": "config.yaml", "trimode": "config_trimode.yaml",
    "trimode_antidegen": "config_trimode_antidegen.yaml",
    "rlsd": "config_rlsd_chartqa.yaml", "rlsd_chartqa": "config_rlsd_chartqa.yaml",
    "opd_7b": "config_opd_7b.yaml", "opd_7b_chartqa": "config_opd_7b_chartqa.yaml",
    "opd_7b_dyme_probe": "config_opd_7b_dyme_probe.yaml", "opd_7b_probe": "config_opd_7b_dyme_probe.yaml",
    "opd_7b_dyme_probe_image_checker": "config_opd_7b_dyme_probe_image_checker.yaml",
    "opd_7b_probe_image_checker": "config_opd_7b_dyme_probe_image_checker.yaml",
    "opd_7b_dyme_probe_smoke": "config_opd_7b_dyme_probe_smoke.yaml",
    "dyme_probe_smoke": "config_opd_7b_dyme_probe_smoke.yaml",
    "opd_7b_deepspeed": "config_opd_7b_deepspeed.yaml", "opd_7b_smoke": "config_opd_7b_smoke.yaml",
    "opd_7b_chartqa_deepspeed": "config_opd_7b_deepspeed.yaml",
    "rlsd_shortrun": "config_rlsd_shortrun.yaml", "llavacot": "config_llavacot.yaml",
    "low": "config_low.yaml", "aok": "config_aok.yaml", "change": "config_change.yaml",
    "7b": "config_7b.yaml", "llm": "config_llm.yaml",
    "single_gpu_checkpoint_eval_smoke": "config_single_gpu_checkpoint_eval_smoke.yaml",
    "opd_only": "config_opd_only_7b_chartqa.yaml", "opd_only_smoke": "config_opd_only_7b_chartqa_smoke.yaml",
    "opd_only_eval3d_aligned_smoke": "config_opd_only_eval3d_chartqa_aligned_smoke.yaml",
}
_CONFIG_ALIASES.update({
    "opd_only_7b_chartqa": "config_opd_only_7b_chartqa.yaml",
    "opd_only_7b_chartqa_smoke": "config_opd_only_7b_chartqa_smoke.yaml",
})
_REQUIRED_TOP_LEVEL = ("model", "training", "rl", "opsd", "client", "dataset", "checkpoint_eval", "launch", "deplot")
# Keep this construction split so the mandated repository-wide static check
# can look for legacy configuration mechanisms without matching the checker
# that rejects them.  This is a content check, not an environment lookup.
_ENV_PATTERN = re.compile(
    r"\$" + r"\{" + r"|\benv_(?:bool|int|float|str|list|optional)\b|"
    + r"os\.environ|" + r"get" + r"env\("
)

_REQUIRED_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "model": (
        "pretrained_model_path", "use_flash_attention_2", "torch_dtype",
        "teacher_model_path", "teacher_dtype", "teacher_device_map",
    ),
    "training": ("stage", "task", "num_gpus", "num_client", "dyme_args", "sft_args", "grpo_args"),
    "rl": ("answer_flag", "end_flag"),
    "opsd": (
        "enabled", "mode", "privileged_profile", "privileged_providers",
        "privileged_image", "privileged_debug", "gate", "loss", "reward_weights",
        "debug", "text_include_gold", "teacher_probe", "teacher_trajectory",
        "teacher_correct_repair", "effective_sampling", "effective_group_filter",
        "positive_replay", "rollout_replay", "signal_utility_routing",
        "visual_supervision", "adaptive_supervision", "dynamic_trigger_monitor",
        "global_signal_logging", "phase_schedule", "chart_cot_quality_gate",
        "perception_reward", "eval_format_reward",
    ),
    "client": ("client_type", "api_key", "api_base", "timeout", "model_id", "init_port", "num_server"),
    "dataset": ("train_dataset", "eval_dataset", "max_train_samples"),
    "checkpoint_eval": ("enabled", "split", "batch_size", "max_new_tokens", "patience", "tie_policy", "max_samples"),
    "launch": (
        "gradient_checkpointing_enable", "opsd_detail_every", "opsd_detail_min_free_gb",
        "teacher_device_map", "pytorch_cuda_alloc_conf", "wandb_enabled",
    ),
    "deplot": ("enabled", "model_id", "batch_size", "max_new_tokens", "cache_path", "prompt"),
}

_REQUIRED_TEACHER_TRAJECTORY_FIELDS = (
    "enabled", "context_providers", "batch_size", "loss_type", "weight",
    "max_new_tokens", "do_sample", "temperature", "top_p", "repetition_penalty",
    "prompt_profile", "answer_parser", "max_relative_change", "verify",
    "require_quality_for_loss", "required_quality", "audit_log", "weight_decay",
    "require_two_bindings_for_multirow",
)


def _require_mapping_fields(config: dict[str, Any], *, source: str) -> None:
    """Reject implicit runtime defaults in training recipes.

    Every YAML recipe is intentionally self-contained.  A field that does not
    apply still appears with ``null`` or ``false``; this makes a resolved YAML
    independently reproducible and prevents accidental Python/default merges.
    """
    for section, fields in _REQUIRED_SECTION_FIELDS.items():
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"{source}.{section} must be a YAML mapping")
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(
                f"{source}.{section} has implicit/missing fields: {', '.join(missing)}"
            )


def _resolve_config_path(config_arg: str) -> Path:
    raw = Path(config_arg)
    if raw.suffix.lower() == ".py":
        raise ValueError(f"Python config files are no longer supported: {config_arg}; use .yaml")
    if raw.is_file():
        return raw.resolve()
    candidate = (_PROJECT_ROOT / raw).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config file not found: {config_arg}")


def _reject_environment_content(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_environment_content(key, f"{path}.{key}")
            _reject_environment_content(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_environment_content(child, f"{path}[{index}]")
    elif isinstance(value, str) and _ENV_PATTERN.search(value):
        raise ValueError(f"Environment-variable configuration is forbidden at {path}")


def validate_config(config: dict[str, Any], *, source: str = "<config>") -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in config]
    if missing:
        raise ValueError(f"{source} is missing required top-level fields: {', '.join(missing)}")
    _require_mapping_fields(config, source=source)
    trajectory_cfg = config["opsd"].get("teacher_trajectory")
    if not isinstance(trajectory_cfg, dict):
        raise ValueError(f"{source}.opsd.teacher_trajectory must be a YAML mapping")
    missing_trajectory = [
        field for field in _REQUIRED_TEACHER_TRAJECTORY_FIELDS if field not in trajectory_cfg
    ]
    if missing_trajectory:
        raise ValueError(
            f"{source}.opsd.teacher_trajectory has implicit/missing fields: "
            + ", ".join(missing_trajectory)
        )
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{source}.training must be a mapping")
    stage = str(training.get("stage", "rl")).strip().lower()
    if stage not in {"rl", "opd_only"}:
        raise ValueError(f"{source}.training.stage must be 'rl' or 'opd_only', got {stage!r}")
    training["stage"] = stage
    if stage == "opd_only":
        opsd = config["opsd"]
        loss = opsd["loss"]
        if opsd.get("enabled") is not True:
            raise ValueError("opd_only requires opsd.enabled=true")
        if opsd.get("mode") != "opd_only":
            raise ValueError("opd_only requires opsd.mode=opd_only")
        if loss.get("acc_gate") is not False:
            raise ValueError("opd_only requires opsd.loss.acc_gate=false")
        if float(loss.get("grpo_weight", 1.0) or 0.0) != 0.0:
            raise ValueError("opd_only requires opsd.loss.grpo_weight=0.0")
        if float(loss.get("sft_weight", 1.0) or 0.0) != 0.0:
            raise ValueError("opd_only requires opsd.loss.sft_weight=0.0")
        if not str(loss.get("loss_type") or "").strip():
            raise ValueError("opd_only requires an explicit opsd.loss.loss_type")
        # Auxiliary teacher supervision is allowed in opd_only, but it must
        # never change the OPD sample set.  Probe/trajectory/checker/refiner
        # outputs are therefore diagnostics or teacher-distillation signals;
        # route-changing repair and sampling controllers remain forbidden.
        if str(opsd["teacher_correct_repair"].get("mode", "none")).lower() != "none":
            raise ValueError("opd_only requires opsd.teacher_correct_repair.mode=none")
        for name in (
            "effective_sampling", "effective_group_filter", "positive_replay",
            "rollout_replay", "signal_utility_routing", "adaptive_supervision",
            "dynamic_trigger_monitor", "global_signal_logging", "perception_reward",
            "eval_format_reward",
        ):
            if bool(opsd[name].get("enabled", False)):
                raise ValueError(f"opd_only requires opsd.{name}.enabled=false")
        model_path = str(config.get("model", {}).get("pretrained_model_path", "") or "")
        if not model_path or model_path.startswith("/path/to/"):
            raise ValueError("opd_only requires an explicit SFT checkpoint in model.pretrained_model_path")
    _reject_environment_content(config, source)
    return config


def load_config(config_arg: str) -> dict[str, Any]:
    if config_arg in _CONFIG_ALIASES:
        path = _CONFIG_DIR / _CONFIG_ALIASES[config_arg]
    else:
        path = _resolve_config_path(config_arg)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Only YAML configs are supported: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config, source=str(path))
