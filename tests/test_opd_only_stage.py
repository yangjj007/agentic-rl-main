from pathlib import Path

import pytest
import yaml

from config.loader import load_config


def test_opd_only_yaml_is_explicit_and_isolated():
    cfg = load_config("opd_only")
    assert cfg["training"]["stage"] == "opd_only"
    assert cfg["opsd"]["enabled"] is True
    assert cfg["opsd"]["mode"] == "opd_only"
    assert cfg["opsd"]["loss"]["acc_gate"] is False
    assert cfg["opsd"]["loss"]["grpo_weight"] == 0.0
    assert cfg["opsd"]["loss"]["sft_weight"] == 0.0
    assert cfg["opsd"]["teacher_probe"]["enabled"] is True
    assert cfg["opsd"]["teacher_trajectory"]["enabled"] is True


def test_opd_only_allows_visual_diagnostics_without_route_changes():
    cfg = load_config("opd_only")
    cfg["opsd"]["visual_supervision"] = {
        "enabled": True,
        "checker": {"enabled": True},
        "refiner": {"enabled": False},
    }
    from config.loader import validate_config

    assert validate_config(cfg)["training"]["stage"] == "opd_only"


def test_python_config_paths_are_rejected():
    with pytest.raises(ValueError, match="Python config files are no longer supported"):
        load_config("config/config.py")


def test_all_yaml_aliases_load_without_environment_overrides():
    from config.loader import _CONFIG_ALIASES

    for alias in _CONFIG_ALIASES:
        cfg = load_config(alias)
        assert isinstance(cfg, dict)
        assert "training" in cfg


def test_all_yaml_recipes_are_complete_and_do_not_contain_legacy_env_syntax():
    from config.loader import _REQUIRED_SECTION_FIELDS

    config_dir = Path(__file__).resolve().parents[1] / "config"
    forbidden_fragments = ("${", "env_bool", "env_int", "env_float", "env_str", "os.environ", "getenv(")
    for path in sorted(config_dir.glob("*.yaml")):
        raw = path.read_text(encoding="utf-8")
        assert not any(fragment in raw for fragment in forbidden_fragments), path
        cfg = yaml.safe_load(raw)
        for section, fields in _REQUIRED_SECTION_FIELDS.items():
            assert isinstance(cfg[section], dict), (path, section)
            assert not (set(fields) - set(cfg[section])), (path, section)
