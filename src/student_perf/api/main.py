"""
FastAPI application for the Student Performance Prediction Platform.

Endpoints:
  GET  /health                - Liveness/readiness check
  GET  /models                - List available models + metrics
  POST /predict               - Predict with default (best) model
  POST /predict/{model_name}  - Predict with specific model
  POST /explain               - Get SHAP + LIME explanations
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from student_perf.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictRequest,
    PredictionResponse,
    StudentFeatures,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state — populated at startup
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "models": {},
    "metrics": {},
    "preprocessor": None,
    "feature_names": [],
    "X_test_sample": None,
    "default_model": None,
}

LABEL_MAP = {0: "Fail", 1: "Pass"}


def _pick_default_model(metrics: dict, models: dict) -> str | None:
    """Pick the best model by F1 score."""
    best_name = None
    best_f1 = -1.0
    for name in models:
        if name in metrics and "f1" in metrics[name]:
            f1 = float(metrics[name].get("f1", 0))
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
    return best_name or (list(models.keys())[0] if models else None)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models + preprocessor at startup."""
    logger.info("🚀 Loading models at startup...")
    try:
        from student_perf.api.inference import (
            load_all_models,
            load_preprocessor,
            load_feature_names,
        )
        import numpy as np
        from pathlib import Path

        models, metrics = load_all_models()
        preprocessor = load_preprocessor()
        feature_names = load_feature_names()

        # Load a small X_test sample for SHAP background
        x_test_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "X_test.npy"
        X_test = np.load(x_test_path, allow_pickle=True) if x_test_path.exists() else None

        _STATE["models"] = models
        _STATE["metrics"] = metrics
        _STATE["preprocessor"] = preprocessor
        _STATE["feature_names"] = feature_names
        _STATE["X_test_sample"] = X_test
        _STATE["default_model"] = _pick_default_model(metrics, models)

        logger.info(f"✅ Loaded {len(models)} models. Default: {_STATE['default_model']}")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "Predict student outcomes and explain predictions using SHAP + LIME. "
        "Supports binary (pass/fail), multiclass (Low/Medium/High), and regression (G3) tasks."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model(model_name: str | None):
    models = _STATE["models"]
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded. Run training pipeline first.")

    name = model_name or _STATE["default_model"]
    if name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {list(models.keys())}",
        )
    return name, models[name]


def _preprocess(features: StudentFeatures) -> np.ndarray:
    from student_perf.api.inference import preprocess_single
    preprocessor = _STATE["preprocessor"]
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded.")
    return preprocess_single(features.model_dump(), preprocessor)


def _make_prediction(model, X: np.ndarray, model_name: str) -> dict:
    pred = model.predict(X)
    label = int(pred[0]) if hasattr(pred[0], "__int__") else float(pred[0])
    result = {
        "prediction": label,
        "prediction_label": LABEL_MAP.get(label, str(label)),
    }
    try:
        proba = model.predict_proba(X)
        if proba is not None and proba.ndim == 2:
            result["confidence"] = float(proba[0, label]) if label < proba.shape[1] else None
            result["probabilities"] = {
                LABEL_MAP.get(i, str(i)): float(p)
                for i, p in enumerate(proba[0])
            }
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Liveness and readiness check."""
    return HealthResponse(
        status="ok",
        models_loaded=len(_STATE["models"]),
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """List all available trained models and their metrics."""
    model_infos = []
    metrics = _STATE["metrics"]
    for name in _STATE["models"]:
        model_infos.append(ModelInfo(
            name=name,
            task_type="binary",
            metrics=metrics.get(name, {}),
            available=True,
        ))
    return ModelsListResponse(
        models=model_infos,
        default_model=_STATE["default_model"] or "",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictRequest):
    """Predict student outcome using default (or specified) model."""
    model_name, model = _get_model(request.model_name)
    X = _preprocess(request.features)
    result = _make_prediction(model, X, model_name)
    return PredictionResponse(model_used=model_name, **result)


@app.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
async def predict_with_model(
    features: StudentFeatures,
    model_name: str = Path(..., description="Model name to use for prediction"),
):
    """Predict using a specific model."""
    model_name_key, model = _get_model(model_name)
    X = _preprocess(features)
    result = _make_prediction(model, X, model_name_key)
    return PredictionResponse(model_used=model_name_key, **result)


@app.post("/explain", response_model=ExplanationResponse, tags=["Explanations"])
async def explain(request: PredictRequest):
    """Get SHAP + LIME explanations for a prediction."""
    from student_perf.api.inference import get_shap_for_instance, get_lime_for_instance
    from student_perf.explain.explanation_utils import shap_lime_rank_correlation
    import numpy as np

    model_name, model = _get_model(request.model_name)
    X = _preprocess(request.features)
    feature_names = _STATE["feature_names"]
    X_bg = _STATE["X_test_sample"]
    if X_bg is None:
        X_bg = X  # fallback

    result = _make_prediction(model, X, model_name)

    shap_contribs = get_shap_for_instance(model, model_name, X, X_bg, feature_names)
    lime_contribs = get_lime_for_instance(model, model_name, X, X_bg, feature_names)

    # Compute SHAP-LIME correlation
    corr = None
    if shap_contribs and lime_contribs:
        shap_pairs = [(c["feature"], c["weight"]) for c in shap_contribs]
        lime_pairs = [(c["feature"], c["weight"]) for c in lime_contribs]
        shap_dict = dict(shap_pairs)
        lime_dict = dict(lime_pairs)
        common = [f for f in shap_dict if f in lime_dict]
        if len(common) >= 3:
            from scipy.stats import spearmanr
            s_vals = [abs(shap_dict[f]) for f in common]
            l_vals = [abs(lime_dict[f]) for f in common]
            r, _ = spearmanr(s_vals, l_vals)
            corr = float(r) if not np.isnan(r) else None

    return ExplanationResponse(
        model_used=model_name,
        prediction=result["prediction"],
        shap_contributions=[FeatureContribution(**c) for c in shap_contribs],
        lime_contributions=[FeatureContribution(**c) for c in lime_contribs],
        shap_lime_rank_correlation=corr,
    )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
