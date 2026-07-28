"""
LIME explainability module.

Uses LimeTabularExplainer for all model types (model-agnostic).
Generates local explanations for 3 representative students and saves
them as HTML (LIME native) + normalized JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_lime_explainer(
    X_train: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
    class_names: list[str] | None = None,
):
    """Build a LimeTabularExplainer."""
    from lime.lime_tabular import LimeTabularExplainer

    mode = "regression" if task_type == "regression" else "classification"
    if class_names is None:
        class_names = ["Fail", "Pass"] if task_type == "binary" else None

    return LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        random_state=42,
        discretize_continuous=True,
    )


def explain_instance(
    explainer,
    model,
    X_instance: np.ndarray,
    task_type: str = "binary",
    num_features: int = 10,
    num_samples: int = 500,
) -> dict:
    """
    Generate LIME explanation for one instance.

    Returns:
        dict with 'explanation' (list of (feature, weight)) and 'html'
    """
    if task_type == "regression":
        predict_fn = lambda x: model.predict(x)  # noqa: E731
    else:
        def predict_fn(x):
            proba = model.predict_proba(x)
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba

    exp = explainer.explain_instance(
        X_instance,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    return {
        "explanation": exp.as_list(),
        "html": exp.as_html(),
        "local_pred": exp.local_pred,
        "score": exp.score,
    }


def explain_samples(
    model,
    explainer,
    X_test: np.ndarray,
    model_name: str,
    task_type: str = "binary",
    num_features: int = 10,
    sample_indices: list[int] | None = None,
    sample_labels: list[str] | None = None,
) -> dict:
    """
    Explain multiple samples and save HTML + JSON outputs.

    Returns:
        Dict of {label: explanation_dict}
    """
    output_dir = _PROJECT_ROOT / "data" / "processed" / "lime_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(X_test)
    if sample_indices is None:
        sample_indices = [0, n // 2, n - 1]
    if sample_labels is None:
        sample_labels = ["low_performer", "borderline", "high_performer"]

    results = {}
    for idx, label in zip(sample_indices, sample_labels):
        if idx >= n:
            continue
        logger.info(f"LIME explanation: {model_name} — {label} (idx={idx})")
        try:
            exp = explain_instance(explainer, model, X_test[idx], task_type, num_features)

            # Save HTML
            html_path = output_dir / f"{model_name}_{label}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(exp["html"])

            # Save JSON (normalized weights)
            features_weights = [
                {"feature": feat, "weight": float(weight)}
                for feat, weight in exp["explanation"]
            ]
            json_path = output_dir / f"{model_name}_{label}.json"
            with open(json_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "sample_index": idx,
                    "label": label,
                    "features": features_weights,
                    "local_pred": float(exp["local_pred"][0]) if exp["local_pred"] is not None else None,
                    "score": float(exp["score"]),
                }, f, indent=2)

            results[label] = {
                "explanation": exp["explanation"],
                "html_path": str(html_path),
                "json_path": str(json_path),
            }
        except Exception as e:
            logger.error(f"LIME failed for {model_name}/{label}: {e}")
            results[label] = {"error": str(e)}

    return results


def explain_all_models(
    loaded_models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
) -> dict:
    """
    Run LIME for all models. Reuse same explainer (training distribution).

    Returns:
        {model_name: {label: explanation_dict}}
    """
    explainer = get_lime_explainer(X_train, feature_names, task_type)
    all_results = {}

    for name, model in loaded_models.items():
        logger.info(f"LIME explaining: {name}")
        try:
            all_results[name] = explain_samples(model, explainer, X_test, name, task_type)
        except Exception as e:
            logger.error(f"LIME failed for {name}: {e}")
            all_results[name] = {"error": str(e)}

    return all_results
