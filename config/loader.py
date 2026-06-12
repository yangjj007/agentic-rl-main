import importlib
import importlib.util
import os
from typing import Any

_CONFIG_ALIASES: dict[str, str] = {
    "norm": "config.config",
    "trimode": "config.config_trimode",
    "trimode_antidegen": "config.config_trimode_antidegen",
    "rlsd": "config.config_rlsd_chartqa",
    "rlsd_chartqa": "config.config_rlsd_chartqa",
    "opd_7b": "config.config_opd_7b_chartqa",
    "opd_7b_chartqa": "config.config_opd_7b_chartqa",
    "llavacot": "config.config_llavacot",
    "low": "config.config_low",
    "aok": "config.config_aok",
    "change": "config.config_change",
    "7b": "config.config_7B",
    "llm": "config.config_llm",
}


def _resolve_config_path(config_arg: str) -> str:
    if os.path.isfile(config_arg):
        return os.path.abspath(config_arg)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(project_root, config_arg)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def load_config(config_arg: str) -> dict[str, Any]:
    """
    Load CONFIG from a Python config file path or shorthand alias.

    Examples:
        load_config("config/config.py")
        load_config("config/config_trimode.py")
        load_config("norm")
        load_config("trimode")
    """
    if config_arg.endswith(".py"):
        path = _resolve_config_path(config_arg)
        module_name = f"_dyme_runtime_config_{abs(hash(path)) % 10 ** 8}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "CONFIG"):
            raise ValueError(f"{path} must define a CONFIG dict")
        return module.CONFIG

    module_path = _CONFIG_ALIASES.get(config_arg)
    if module_path is None:
        raise ValueError(
            f"Unknown config alias '{config_arg}'. "
            f"Use a .py path (e.g. config/config.py) or one of: {', '.join(sorted(_CONFIG_ALIASES))}"
        )
    module = importlib.import_module(module_path)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"{module_path} must define a CONFIG dict")
    return module.CONFIG
