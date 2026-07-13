import importlib
import os
from collections.abc import Callable, Mapping


def maybe_allow_trusted_torch_load(
    *,
    resume_from_checkpoint: str | None,
    env: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Allow trusted local Trainer checkpoints on torch<2.6.

    Transformers 4.57 blocks torch.load for optimizer/scheduler `.pt` files
    unless torch>=2.6. Our DeepSpeed resume checkpoints are produced locally;
    require an explicit env opt-in before bypassing that guard.
    """

    env = env or os.environ
    if not resume_from_checkpoint:
        return False
    if env.get("DYME_TRUST_LOCAL_TORCH_LOAD", "").strip() != "1":
        return False

    def _trusted_local_checkpoint_is_safe() -> None:
        return None

    trainer_module = importlib.import_module("transformers.trainer")
    import_utils_module = importlib.import_module("transformers.utils.import_utils")
    trainer_module.check_torch_load_is_safe = _trusted_local_checkpoint_is_safe
    import_utils_module.check_torch_load_is_safe = _trusted_local_checkpoint_is_safe
    log(
        "[DyME] DYME_TRUST_LOCAL_TORCH_LOAD=1: allowing torch.load for "
        f"trusted local resume checkpoint: {resume_from_checkpoint}"
    )
    return True
