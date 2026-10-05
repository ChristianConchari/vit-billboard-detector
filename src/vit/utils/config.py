"""YAML configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Raises FileNotFoundError if the path does not exist, so misconfigured
    pipelines fail fast instead of silently running with defaults.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Apply `section.key=value` overrides in place; values are parsed as YAML.

    So `training.freeze_backbone=false` sets a bool and `training.learning_rate=5e-5`
    a float. Only existing keys can be overridden, which catches typos early.
    """
    for override in overrides:
        dotted_key, sep, raw_value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value: {override!r}")
        *parents, key = dotted_key.split(".")
        section = config
        for parent in parents:
            section = section[parent]
        if key not in section:
            raise KeyError(f"Unknown config key: {dotted_key}")
        section[key] = _parse_value(raw_value)


def _parse_value(raw_value: str) -> Any:
    # PyYAML follows YAML 1.1, which reads "5e-5" (no dot) as a string.
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value
