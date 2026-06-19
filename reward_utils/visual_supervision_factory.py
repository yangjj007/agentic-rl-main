"""Factory for Visual Supervision checker / refiner."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from reward_utils.checker import RewardCalculatorLocal
from reward_utils.refiner import ContextRefinerLocal
from reward_utils.template_pool import TemplatePool
from reward_utils.visual_checker_teacher import TeacherVisualChecker
from reward_utils.visual_refiner_teacher import TeacherVisualRefiner


def _visual_enabled(visual_config: dict) -> bool:
    if visual_config.get("enabled") is True:
        return True
    if visual_config.get("enabled") is False:
        return False
    checker_on = visual_config.get("checker", {}).get("enabled", False)
    refiner_on = visual_config.get("refiner", {}).get("enabled", False)
    return bool(checker_on or refiner_on)


def visual_supervision_needs_teacher(opsd_config: dict) -> bool:
    visual_config = dict(opsd_config.get("visual_supervision") or {})
    if not _visual_enabled(visual_config):
        return False
    checker_cfg = visual_config.get("checker", {})
    refiner_cfg = visual_config.get("refiner", {})
    use_teacher_checker = checker_cfg.get("enabled", True) and checker_cfg.get(
        "model_source", "loaded_teacher"
    ) == "loaded_teacher"
    use_teacher_refiner = refiner_cfg.get("enabled", True) and refiner_cfg.get(
        "model_source", "loaded_teacher"
    ) == "loaded_teacher"
    return use_teacher_checker or use_teacher_refiner


def build_visual_supervision(
    rl_config: dict,
    client_config: dict,
    opsd_config: dict,
    *,
    gpu_id: int = 0,
    teacher_model=None,
    processor=None,
) -> Tuple[Any, Any, dict]:
    """
    Returns (checker, refiner, meta).
    meta keys: enabled, needs_teacher, template_pool
    """
    visual_config = dict(opsd_config.get("visual_supervision") or {})
    if not _visual_enabled(visual_config):
        return (
            RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=gpu_id),
            ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=gpu_id),
            {"enabled": False, "needs_teacher": False},
        )

    pool_cfg = visual_config.get("template_pool", {})
    template_pool = TemplatePool(
        template_path=pool_cfg.get("path", "best_template.txt"),
        refresh_interval_sec=float(pool_cfg.get("refresh_interval_sec", 60)),
    )

    checker_cfg = visual_config.get("checker", {})
    refiner_cfg = visual_config.get("refiner", {})
    use_teacher_checker = checker_cfg.get("enabled", True) and checker_cfg.get(
        "model_source", "loaded_teacher"
    ) == "loaded_teacher"
    use_teacher_refiner = refiner_cfg.get("enabled", True) and refiner_cfg.get(
        "model_source", "loaded_teacher"
    ) == "loaded_teacher"
    needs_teacher = use_teacher_checker or use_teacher_refiner

    if use_teacher_checker:
        checker = TeacherVisualChecker(
            rl_config,
            client_config.copy(),
            gpu_id=gpu_id,
            visual_config=visual_config,
            template_pool=template_pool,
        )
    else:
        checker = RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=gpu_id)

    if use_teacher_refiner:
        refiner = TeacherVisualRefiner(
            rl_config,
            client_config.copy(),
            gpu_id=gpu_id,
            visual_config=visual_config,
            template_pool=template_pool,
        )
    else:
        refiner = ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=gpu_id)

    if teacher_model is not None and processor is not None:
        if hasattr(checker, "bind_teacher"):
            checker.bind_teacher(teacher_model, processor)
        if hasattr(refiner, "bind_teacher"):
            refiner.bind_teacher(teacher_model, processor)

    return checker, refiner, {
        "enabled": True,
        "needs_teacher": needs_teacher,
        "template_pool": template_pool,
        "visual_config": visual_config,
    }
