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
