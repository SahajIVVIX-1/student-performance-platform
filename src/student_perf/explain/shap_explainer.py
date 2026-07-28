"""
SHAP explainability module.

Selects the appropriate explainer type based on model class:
  - Tree models (RF, XGB, CatBoost, LGBM): TreeExplainer (fast, exact)
  - Linear models (LR): LinearExplainer
  - Neural nets (MLP, LSTM): KernelExplainer / DeepExplainer
  - AutoML: KernelExplainer (model-agnostic)

SHAP values are cached to disk keyed by model_name + data hash.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_config() -> dict:
    with open(_PROJECT_ROOT / "configs" / "model_config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _data_hash(X: np.ndarray) -> str:
    return hashlib.md5(X.tobytes()).hexdigest()[:12]


def _cache_path(model_name: str, data_hash: str) -> Path:
    cache_dir = _PROJECT_ROOT / "data" / "processed" / "shap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_name}_{data_hash}.pkl"


def _load_cache(path: Path) -> np.ndarray | None:
    if path.exists():
        logger.info(f"Loading SHAP cache: {path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(path: Path, values: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(values, f)
    logger.info(f"SHAP cache saved: {path.name}")


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def _is_tree_model(model_obj) -> bool:
    tree_types = (
        "RandomForest", "XGB", "xgb", "CatBoost", "LightGBM", "lgbm",
        "DecisionTree",
    )
    cls_name = type(model_obj).__name__
    return any(t in cls_name for t in tree_types)


def _is_linear_model(model_obj) -> bool:
    return type(model_obj).__name__ in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso")


def get_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    model_name: str,
    task_type: str = "binary",
    use_cache: bool = True,
) -> tuple[Any, np.ndarray]:
    """
    Compute SHAP values for X_explain.

    Args:
        model: BaseModel instance.
        X_background: Background dataset for KernelExplainer.
        X_explain: Instances to explain.
        model_name: Name for caching.
        task_type: Task type string.
        use_cache: Whether to use disk cache.

    Returns:
        (explainer, shap_values)
    """
    data_hash = _data_hash(X_explain)
    cache_file = _cache_path(model_name, data_hash)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return None, cached

    model_obj = model.get_sklearn_model()

    try:
        if _is_tree_model(model_obj):
            logger.info(f"Using TreeExplainer for {model_name}")
            explainer = shap.TreeExplainer(model_obj)
            shap_vals = explainer.shap_values(X_explain)
        elif _is_linear_model(model_obj):
            logger.info(f"Using LinearExplainer for {model_name}")
            explainer = shap.LinearExplainer(model_obj, X_background)
            shap_vals = explainer.shap_values(X_explain)
        else:
            # Neural nets, AutoML — use KernelExplainer
            logger.info(f"Using KernelExplainer for {model_name} (may be slow)")
            background = shap.sample(X_background, min(50, len(X_background)))

            def predict_fn(x):
                proba = model.predict_proba(x)
                if proba.ndim == 2:
                    return proba[:, 1]
                return proba

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_vals = explainer.shap_values(X_explain[:min(20, len(X_explain))], nsamples=100)
            X_explain = X_explain[:min(20, len(X_explain))]

    except Exception as e:
        logger.warning(f"TreeExplainer failed for {model_name}: {e}. Falling back to KernelExplainer.")
        background = shap.sample(X_background, min(50, len(X_background)))
        def predict_fn(x):
            try:
                proba = model.predict_proba(x)
                return proba[:, 1] if proba.ndim == 2 else proba
            except Exception:
                return model.predict(x).astype(float)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_explain[:min(20, len(X_explain))], nsamples=100)

    if use_cache:
        _save_cache(cache_file, shap_vals)

    return explainer, shap_vals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_summary(
    shap_values,
    X: np.ndarray,
    feature_names: list[str],
    model_name: str,
    output_dir: Path,
) -> str:
    """Generate and save beeswarm summary plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = str(output_dir / f"{model_name}_shap_summary.png")

    # Handle both binary (2D) and multiclass (list) SHAP values
    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[1]  # positive class

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved: {path}")
    return path


def plot_feature_importance_bar(
    shap_values,
    feature_names: list[str],
    model_name: str,
    output_dir: Path,
) -> str:
    """Generate and save SHAP feature importance bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = str(output_dir / f"{model_name}_shap_importance.png")

    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[1]

    mean_abs = np.abs(vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],
        mean_abs[top_idx[::-1]],
        color="steelblue",
    )
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    logger.info(f"SHAP importance bar saved: {path}")
    return path


def plot_waterfall(
    shap_values,
    X: np.ndarray,
    feature_names: list[str],
    sample_idx: int,
    model_name: str,
    label: str,
    output_dir: Path,
    expected_value=None,
) -> str:
    """Generate waterfall plot for one sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = str(output_dir / f"{model_name}_waterfall_{label}.png")

    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[1]

    sample_vals = vals[sample_idx] if len(vals) > sample_idx else vals[0]

    plt.figure(figsize=(9, 5))
    # Use bar chart as waterfall approximation if shap.waterfall fails
    try:
        if expected_value is not None:
            shap_exp = shap.Explanation(
                values=sample_vals,
                base_values=expected_value if np.isscalar(expected_value) else expected_value[1],
                data=X[sample_idx],
                feature_names=feature_names,
            )
            shap.plots.waterfall(shap_exp, show=False)
        else:
            raise ValueError("No expected value")
    except Exception:
        # Fallback: simple bar
        sorted_idx = np.argsort(np.abs(sample_vals))[::-1][:10]
        ax = plt.gca()
        colors = ["red" if v < 0 else "green" for v in sample_vals[sorted_idx]]
        ax.barh(
            [feature_names[i] for i in sorted_idx[::-1]],
            sample_vals[sorted_idx[::-1]],
            color=colors[::-1],
        )
        ax.set_title(f"SHAP — {model_name} ({label} student)")
        ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP waterfall ({label}) saved: {path}")
    return path


def explain_all_models(
    loaded_models: dict,
    X_test: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
) -> dict:
    """
    Generate SHAP explanations for all models.

    Args:
        loaded_models: {model_name: BaseModel instance}
        X_test: Test set for explanations
        feature_names: Feature name list
        task_type: Task type

    Returns:
        Dict of {model_name: {"shap_values": ..., "plots": [...]}}
    """
    output_dir = _PROJECT_ROOT / "data" / "processed" / "shap_plots"
    results = {}

    # Pick 3 representative samples
    # (find one high performer, one low performer, one borderline)
    # We use indices 0, len//2, -1 as approximations
    sample_labels = ["low_performer", "borderline", "high_performer"]
    n = len(X_test)
    sample_indices = [0, n // 2, n - 1]

    for name, model in loaded_models.items():
        logger.info(f"Computing SHAP for {name}...")
        try:
            _, shap_vals = get_shap_values(
                model, X_test, X_test, name, task_type
            )
            plots = []
            plots.append(plot_summary(shap_vals, X_test, feature_names, name, output_dir))
            plots.append(plot_feature_importance_bar(shap_vals, feature_names, name, output_dir))

            for idx, label in zip(sample_indices, sample_labels):
                if idx < len(X_test):
                    plots.append(plot_waterfall(shap_vals, X_test, feature_names, idx, name, label, output_dir))

            results[name] = {"shap_values": shap_vals, "plots": plots}
        except Exception as e:
            logger.error(f"SHAP failed for {name}: {e}")
            results[name] = {"error": str(e)}

    return results
