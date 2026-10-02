"""MLflow helpers: run setup, reproducibility tags and config flattening."""
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def start_run(mlflow_config: dict[str, Any], run_name: str) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    return mlflow.start_run(run_name=run_name)


def require_active_run() -> None:
    """Fail instead of letting MLflow silently open a stray run in the default experiment."""
    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run: wrap the call in vit.utils.tracking.start_run")


def git_tags() -> dict[str, str]:
    """Commit the code ran from, and whether the working tree had uncommitted changes."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"git.commit": "unknown", "git.dirty": "unknown"}
    return {"git.commit": commit, "git.dirty": str(dirty).lower()}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config sections into dotted keys for MLflow params."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
