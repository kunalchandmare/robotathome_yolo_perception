from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow


DEFAULT_TRACKING_DB = "mlflow.db"


def sqlite_tracking_uri(db_path: str | Path) -> str:
    """Build a Windows-safe SQLite tracking URI for an absolute database path."""
    return f"sqlite:///{Path(db_path).resolve().as_posix()}"


def configure_project_mlflow(
    project_root: str | Path,
    experiment_name: Optional[str] = None,
    tracking_db_name: str = DEFAULT_TRACKING_DB,
) -> tuple[str, Optional[str]]:
    """Configure MLflow so every project entry point logs to the repo-root tracking DB."""
    project_root = Path(project_root).resolve()
    tracking_uri = sqlite_tracking_uri(project_root / tracking_db_name)

    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    resolved_experiment = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if resolved_experiment:
        mlflow.set_experiment(resolved_experiment)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = resolved_experiment

    return tracking_uri, resolved_experiment

