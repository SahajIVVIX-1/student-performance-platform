# PROGRESS LOG — Explainable Student Performance Prediction Platform

All phases logged here as they complete.

---

## Phase 1 — Project Scaffolding & Environment
**Status:** ✅ Complete
- Full directory tree scaffolded
- `configs/`, `requirements.txt`, `pyproject.toml`, `.env.example`, `.gitignore`, `Makefile`
- Dependencies installed into `P:\LabX-Global` virtual environment

---

## Phase 2 — Data Ingestion & Validation
**Status:** ✅ Complete
- `data/ingest.py`: UCI dataset download with synthetic fallback
- `data/validate.py`: Pandera schema validation with JSON report
- Tests: `tests/test_data_validation.py` — **4/4 passing**

---

## Phase 3 — Cleaning & Feature Engineering
**Status:** ✅ Complete
- `data/clean.py`: Imputation, dtype coercion, deduplication
- `features/build_features.py`: `grade_trend`, `study_efficiency`, `support_index`, `parent_edu`, `alcohol_index`
- `features/pipeline.py`: ColumnTransformer, stratified splits, joblib persistence
- Tests: `tests/test_features.py` — **3/3 passing**

---

## Phase 4 — Experiment Tracking
**Status:** ✅ Complete
- `tracking/mlflow_utils.py`: MLflow with `file:///` URI for Windows compatibility
- MLflow experiment `student-performance` created and running

---

## Phase 5 — Classical ML Models
**Status:** ✅ Complete
- `BaseModel` abstract interface
- 6 models: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM
- Tests: `tests/test_models.py` — **4/4 passing**

---

## Phase 6 — PyTorch MLP
**Status:** ✅ Complete
- `models/mlp_torch.py`: BatchNorm, Dropout, LR scheduler, early stopping
- **Test F1: 0.842 | ROC-AUC: 0.870**

---

## Phase 7 — PyTorch LSTM (Temporal)
**Status:** ✅ Complete
- `data/generate_temporal.py`: Synthetic 10-week weekly quiz dataset
- `models/lstm_torch.py`: LSTM on CUDA with early stopping
- **Test F1: 0.860 | ROC-AUC: 0.944**

---

## Phase 8 — AutoML (FLAML)
**Status:** ✅ Complete
- `models/automl_runner.py`: FLAML AutoML wrapper
- Best estimator: `lrl1` | **Test F1: 0.818 | ROC-AUC: 0.907**

---

## Phase 9 — Evaluation & Comparison
**Status:** ✅ Complete
- `evaluation/metrics.py`: Classification metrics, bootstrap CIs
- `evaluation/comparison.py`: Leaderboard + Plotly charts

### Final Model Leaderboard (Test Set):
| Rank | Model | F1 | ROC-AUC | Accuracy | Train Time |
|------|-------|-----|---------|----------|-----------|
| 1 | **Random Forest** | **0.900** | **0.963** | 0.867 | 13.1s |
| 2 | LSTM (Temporal) | 0.860 | 0.944 | 0.823 | 1.6s |
| 3 | Logistic Regression | 0.857 | 0.815 | 0.800 | 4.0s |
| 4 | XGBoost | 0.857 | 0.759 | 0.800 | 3.1s |
| 5 | MLP (PyTorch) | 0.842 | 0.870 | 0.800 | 1.9s |
| 6 | AutoML (FLAML) | 0.818 | 0.907 | 0.733 | 180s |
| 7 | CatBoost | 0.818 | 0.907 | 0.733 | 0.6s |
| 8 | LightGBM | 0.800 | 0.815 | 0.733 | 0.1s |
| 9 | Decision Tree | 0.762 | 0.824 | 0.667 | 0.1s |

---

## Phase 10 & 11 — Explainability (SHAP & LIME)
**Status:** ✅ Complete
- `explain/shap_explainer.py`: TreeExplainer / LinearExplainer / KernelExplainer with disk cache
- `explain/lime_explainer.py`: LimeTabularExplainer with HTML + JSON output
- `explain/explanation_utils.py`: Normalization, SHAP-LIME rank correlation
- Tests: `tests/test_explainability.py` — **3/3 passing**

---

## Phase 12 — REST API (FastAPI)
**Status:** ✅ Complete — **LIVE on http://localhost:8000**
- `api/schemas.py`: Pydantic v2 schemas
- `api/inference.py`: Model loader & inference engine
- `api/main.py`: 5 endpoints: `/health`, `/models`, `/predict`, `/predict/{model}`, `/explain`
- Smoke test: `random_forest` → **Prediction: Pass (84.6% confidence)**
- Tests: `tests/test_api.py` — **3/3 passing**

---

## Phase 13 — Streamlit Dashboard
**Status:** ✅ Complete — **LIVE on http://localhost:8501**
- 5 tabs: Overview, Model Leaderboard, Single Prediction + XAI, Explainability Explorer, MLflow Tracking

---

## Phase 14 — Dockerization
**Status:** ✅ Complete
- `docker/Dockerfile.api`, `docker/Dockerfile.dashboard`, `docker/Dockerfile.train`
- `docker-compose.yml`: API + Dashboard + MLflow UI orchestration

---

## Phase 15 — CI/CD
**Status:** ✅ Complete
- `.github/workflows/ci.yml`: Lint (Ruff) + Pytest + Docker Build

---

## Phase 16 — Final Verification
**Status:** ✅ Complete

### Test Results: **17 / 17 passed** ✅
```
tests/test_api.py                    3/3  ✅
tests/test_data_validation.py        4/4  ✅
tests/test_explainability.py         3/3  ✅
tests/test_features.py               3/3  ✅
tests/test_models.py                 4/4  ✅
```

### Trained Model Artifacts (9/9):
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/catboost.cbm`
- `models/lightgbm.joblib`
- `models/mlp_torch.pt`
- `models/lstm_torch.pt`
- `models/automl_flaml.joblib`

### Running Services:
- **FastAPI REST API**: http://localhost:8000 (Swagger: http://localhost:8000/docs)
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: Run `mlflow ui` → http://localhost:5000
