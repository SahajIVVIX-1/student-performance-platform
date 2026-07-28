"""
Inference engine: model loading, preprocessing, prediction, and explanation assembly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def discover_model_files() -> dict[str, Path]:
    """Discover trained model files in the models/ directory."""
    if not _MODELS_DIR.exists():
        return {}

    patterns = {
        "logistic_regression": "logistic_regression.joblib",
        "decision_tree": "decision_tree.joblib",
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost.json",
        "catboost": "catboost.cbm",
        "lightgbm": "lightgbm.txt",
        "mlp_torch": "mlp_torch.pt",
        "lstm_torch": "lstm_torch.pt",
        "automl_flaml": "automl_flaml.joblib",
    }

    found = {}
    for name, filename in patterns.items():
        path = _MODELS_DIR / filename
        if path.exists():
            found[name] = path
    return found


def load_model(name: str, path: Path) -> Any:
    """Load a model by name using its appropriate loader."""
    logger.info(f"Loading model: {name} from {path}")

    if name == "logistic_regression":
        from student_perf.models.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel.load(path)
    elif name == "decision_tree":
        from student_perf.models.decision_tree import DecisionTreeModel
        return DecisionTreeModel.load(path)
    elif name == "random_forest":
        from student_perf.models.random_forest import RandomForestModel
        return RandomForestModel.load(path)
    elif name == "xgboost":
        from student_perf.models.xgboost_model import XGBoostModel
        return XGBoostModel.load(path)
    elif name == "catboost":
        from student_perf.models.catboost_model import CatBoostModel
        return CatBoostModel.load(path)
    elif name == "lightgbm":
        from student_perf.models.lightgbm_model import LightGBMModel
        m = LightGBMModel.__new__(LightGBMModel)
        import lightgbm as lgb
        m._model = lgb.Booster(model_file=str(path))
        m._is_fitted = True
        # Wrap predict_proba
        def _predict(x):
            p = m._model.predict(x)
            if p.ndim == 1:
                return np.column_stack([1 - p, p])
            return p
        m.predict_proba = _predict
        m.predict = lambda x: (m._model.predict(x) > 0.5).astype(int)
        return m
    elif name == "mlp_torch":
        from student_perf.models.mlp_torch import MLPModel
        return MLPModel.load(path)
    elif name == "lstm_torch":
        from student_perf.models.lstm_torch import LSTMModel
        return LSTMModel.load(path)
    elif name == "automl_flaml":
        from student_perf.models.automl_runner import AutoMLModel
        return AutoMLModel.load(path)
    else:
        raise ValueError(f"Unknown model: {name}")


def load_all_models() -> tuple[dict, dict]:
    """
    Load all available trained models at startup.

    Returns:
        (models_dict, metrics_dict)
    """
    model_files = discover_model_files()
    models = {}
    for name, path in model_files.items():
        try:
            models[name] = load_model(name, path)
            logger.info(f"✅ Loaded: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")

    metrics = load_metrics()
    return models, metrics


def load_preprocessor():
    """Load the fitted sklearn preprocessor."""
    path = _PROCESSED_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run training pipeline first.")
    return joblib.load(path)


def load_feature_names() -> list[str]:
    """Load feature names."""
    path = _PROCESSED_DIR / "feature_names.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_metrics() -> dict:
    """Load metrics summary."""
    path = _PROCESSED_DIR / "metrics_summary.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return df.set_index("model").to_dict(orient="index")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Feature engineering for single student
# ---------------------------------------------------------------------------

def features_to_dataframe(features_dict: dict) -> pd.DataFrame:
    """Convert student features dict to a DataFrame with engineered features."""
    df = pd.DataFrame([features_dict])

    # Engineered features
    df["grade_trend"] = df["G2"] - df["G1"]
    df["avg_prior_grade"] = (df["G1"] + df["G2"]) / 2.0
    df["study_efficiency"] = df["studytime"] / (df["absences"] + 1)
    df["support_index"] = (
        (df["famsup"] == "yes").astype(int)
        + (df["schoolsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )
    df["parent_edu"] = (df["Medu"] + df["Fedu"]) / 2.0
    df["alcohol_index"] = (df["Dalc"] + df["Walc"]) / 2.0

    return df


def preprocess_single(features_dict: dict, preprocessor) -> np.ndarray:
    """Preprocess a single student feature dict for inference."""
    df = features_to_dataframe(features_dict)
    return preprocessor.transform(df)


# ---------------------------------------------------------------------------
# Explanation assembly
# ---------------------------------------------------------------------------

def get_shap_for_instance(
    model,
    model_name: str,
    X_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
) -> list[dict]:
    """Return SHAP contributions for one instance."""
    from student_perf.explain.shap_explainer import get_shap_values
    from student_perf.explain.explanation_utils import normalize_weights

    try:
        _, shap_vals = get_shap_values(
            model, X_background, X_instance, model_name, task_type, use_cache=False
        )
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
        pairs = list(zip(feature_names, vals))
        return normalize_weights(pairs, top_n=10)
    except Exception as e:
        logger.error(f"SHAP error for {model_name}: {e}")
        return []


def get_lime_for_instance(
    model,
    model_name: str,
    X_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
) -> list[dict]:
    """Return LIME contributions for one instance."""
    from student_perf.explain.lime_explainer import get_lime_explainer, explain_instance
    from student_perf.explain.explanation_utils import normalize_weights

    try:
        explainer = get_lime_explainer(X_background, feature_names, task_type)
        result = explain_instance(explainer, model, X_instance[0], task_type)
        return normalize_weights(result["explanation"], top_n=10)
    except Exception as e:
        logger.error(f"LIME error for {model_name}: {e}")
        return []
