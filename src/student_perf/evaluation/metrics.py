"""
Model evaluation metrics.

Supports: binary classification, multiclass classification, regression.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    task_type: str = "binary",
) -> dict:
    """
    Compute all task-appropriate metrics.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Predicted labels/values.
        y_proba: Predicted probabilities (for classification tasks).
        task_type: "binary" | "multiclass" | "regression"

    Returns:
        Dict of metric name → value.
    """
    metrics: dict = {}

    if task_type == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    elif task_type == "binary":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        if y_proba is not None:
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    elif task_type == "multiclass":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if y_proba is not None:
            try:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            except Exception:
                pass

    return metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_seed)
    scores = []
    n = len(y_true)
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = float(np.percentile(scores, (1 - ci) / 2 * 100))
    upper = float(np.percentile(scores, (1 + ci) / 2 * 100))
    return lower, upper
