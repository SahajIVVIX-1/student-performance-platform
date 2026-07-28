"""
MLflow experiment tracking utilities.

Provides helpers for starting runs, logging params/metrics/artifacts,
and retrieving run history for the dashboard.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """Configure MLflow tracking URI (defaults to file:///.../mlruns)."""
    raw_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", str(_PROJECT_ROOT / "mlruns"))
    if not (raw_uri.startswith("http") or raw_uri.startswith("file:") or raw_uri.startswith("sqlite")):
        uri = Path(raw_uri).resolve().as_uri()
    else:
        uri = raw_uri
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")


def get_or_create_experiment(name: str = "student-performance") -> str:
    """Get experiment ID, creating it if it doesn't exist."""
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        logger.info(f"Created MLflow experiment '{name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def log_model_run(
    model_name: str,
    task_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    model_obj: Any = None,
    experiment_name: str = "student-performance",
) -> str:
    """
    Log a complete model training run to MLflow.

    Args:
        model_name: E.g. "random_forest", "xgboost"
        task_type: "binary" | "regression" | "multiclass"
        params: Hyperparameters dict
        metrics: Evaluation metrics dict
        artifacts: Dict of {label: local_path} for files to log
        model_obj: sklearn-compatible model to log (optional)
        experiment_name: MLflow experiment name

    Returns:
        run_id
    """
    setup_mlflow()
    experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:
        # Tags
        mlflow.set_tags({
            "model_name": model_name,
            "task_type": task_type,
            "platform": "student-perf-platform",
        })

        # Params
        mlflow.log_params(params)

        # Metrics
        mlflow.log_metrics(metrics)

        # Artifacts (figures, reports, etc.)
        if artifacts:
            for label, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=label)
                else:
                    logger.warning(f"Artifact not found, skipping: {path}")

        # Model
        if model_obj is not None:
            try:
                mlflow.sklearn.log_model(model_obj, artifact_path="model")
            except Exception:
                # For PyTorch or non-sklearn models, just log the file path
                logger.debug("sklearn log_model failed; model object may not be sklearn-compatible")

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {model_name} → run_id={run_id}")
        return run_id


def get_all_runs(experiment_name: str = "student-performance") -> "pd.DataFrame":  # noqa: F821
    """Return all MLflow runs as a DataFrame for the dashboard."""
    import pandas as pd
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    return runs
