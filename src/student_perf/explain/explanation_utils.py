"""
Shared explainability utilities.

Helpers for normalizing feature weights, ranking top features,
and computing SHAP vs LIME agreement metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def normalize_weights(weights: list[tuple[str, float]], top_n: int = 10) -> list[dict]:
    """
    Normalize feature weights to [-1, 1] range and return top N.

    Args:
        weights: List of (feature_name, weight) tuples.
        top_n: Number of top features to return.

    Returns:
        List of {'feature': str, 'weight': float, 'weight_normalized': float}
    """
    if not weights:
        return []
    ws = [(f, w) for f, w in weights]
    ws_sorted = sorted(ws, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    max_abs = max(abs(w) for _, w in ws_sorted) or 1.0
    return [
        {
            "feature": f,
            "weight": float(w),
            "weight_normalized": float(w / max_abs),
        }
        for f, w in ws_sorted
    ]


def shap_lime_rank_correlation(
    shap_values: np.ndarray,
    lime_weights: list[tuple[str, float]],
    feature_names: list[str],
) -> float | None:
    """
    Compute Spearman rank correlation between SHAP and LIME feature importance.

    Args:
        shap_values: 1D array of SHAP values for one instance.
        lime_weights: LIME (feature, weight) list.
        feature_names: All feature names.

    Returns:
        Spearman r or None if computation fails.
    """
    try:
        shap_rank = {f: abs(float(shap_values[i])) for i, f in enumerate(feature_names)}
        lime_dict = {f: abs(float(w)) for f, w in lime_weights}
        common = [f for f in shap_rank if f in lime_dict]
        if len(common) < 3:
            return None
        shap_vals = [shap_rank[f] for f in common]
        lime_vals = [lime_dict[f] for f in common]
        r, _ = spearmanr(shap_vals, lime_vals)
        return float(r)
    except Exception:
        return None


def top_n_features(importances: dict[str, float], n: int = 10) -> dict[str, float]:
    """Return top N features by absolute importance."""
    return dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:n])
