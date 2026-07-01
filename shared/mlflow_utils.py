from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

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


@contextmanager
def start_run_with_fallback(logger: logging.Logger | None = None) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run and recover if inherited run-id context is invalid."""
    inherited_run_id = os.environ.get("MLFLOW_RUN_ID")
    if inherited_run_id is not None:
        normalized_run_id = inherited_run_id.strip()
        if normalized_run_id:
            os.environ["MLFLOW_RUN_ID"] = normalized_run_id
        else:
            os.environ.pop("MLFLOW_RUN_ID", None)

    try:
        with mlflow.start_run() as active_run:
            yield active_run
        return
    except Exception as exc:
        exc_text = str(exc).lower()
        if "not found" in exc_text or "parameter 'run_id'" in exc_text or "run with id=" in exc_text:
            inherited_run_id = os.environ.get("MLFLOW_RUN_ID")
            if logger is not None:
                logger.warning(
                    "Inherited MLFLOW_RUN_ID=%s is invalid for current tracking context; starting a new run.",
                    inherited_run_id or "<unset>",
                )
            os.environ.pop("MLFLOW_RUN_ID", None)
            os.environ.pop("MLFLOW_RUN_CONTEXT", None)
            os.environ.pop("MLFLOW_PARENT_RUN_ID", None)
            with mlflow.start_run() as active_run:
                yield active_run
            return
        raise


