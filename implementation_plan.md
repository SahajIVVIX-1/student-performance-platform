# Explainable Student Performance Prediction Platform — Implementation Plan

## Background

Build a production-grade, end-to-end ML platform for predicting and explaining student outcomes. The platform covers the full lifecycle: data ingestion → cleaning → feature engineering → multi-model training → SHAP/LIME explainability → REST API → Streamlit dashboard → Docker deployment → MLflow experiment tracking.

**Working directory:** `p:\03_Projects\student-performance-platform`
**Python environment:** `P:\LabX-Global` (Python 3.11.9)

---

## User Review Required

> [!IMPORTANT]
> **Missing packages must be installed.** The LabX-Global environment has: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `torch`, `streamlit`, `pydantic`, `pytest`, `joblib`. The following **critical packages are NOT installed** and will be pip-installed during Phase 1:
> - `xgboost`, `catboost`, `lightgbm` — gradient boosting models
> - `shap`, `lime` — explainability
> - `mlflow` — experiment tracking
> - `plotly` — visualization
> - `flaml` — AutoML
> - `pandera` — data validation
> - `category_encoders` — categorical encoding
> - `optuna` — hyperparameter tuning
> - `pytest-cov`, `httpx` (for TestClient) — testing
> - `ruff` — linting

> [!WARNING]
> **CatBoost & LightGBM on Windows** can sometimes have native library issues. If either fails to install, it will be noted in PROGRESS.md and excluded from training.

> [!IMPORTANT]
> **LSTM Phase (Phase 7)**: The UCI Student Performance dataset has NO genuine time-series structure. Phase 7 will generate a **synthetic longitudinal extension** (weekly quiz scores over 10 weeks per student) to justify LSTM training. This is clearly documented as a "temporal extension" module.

> [!IMPORTANT]
> **Docker**: The plan assumes Docker Desktop is installed and running. If not, Phases 14–16 will scaffold the Dockerfiles but actual `docker-compose up` won't be verified in this environment.

---

## Open Questions

> [!NOTE]
> **Task type default**: All three targets (regression G3, binary pass/fail, multiclass bands) will be supported via `configs/model_config.yaml`. The primary leaderboard will use **binary classification (pass/fail)** as the default — easiest to compare across models. This can be changed in the config.

> [!NOTE]
> **AutoML time budget**: FLAML will run with a 3-minute budget (180s) to keep Phase 8 tractable in an interactive session. The config makes this adjustable.

---

## Proposed Changes

### Phase 1 — Project Scaffolding & Environment

#### [NEW] Full directory tree
All directories and stub files from Section 3 of the brief, plus:

#### [NEW] `requirements.txt`
Pinned versions for all project dependencies.

#### [NEW] `pyproject.toml`
Project metadata, pytest config, ruff config.

#### [NEW] `configs/data_config.yaml`
Dataset schema documentation, paths, target variable settings.

#### [NEW] `configs/model_config.yaml`
Model hyperparameters, task type, random seed, train/val/test split ratios.

#### [NEW] `configs/logging_config.yaml`
Standard Python logging configuration.

#### [NEW] `Makefile`
Targets: `setup`, `lint`, `test`, `train`, `serve-api`, `serve-dashboard`, `docker-build`, `docker-up`.

#### [NEW] `.env.example`, `.gitignore`, `PROGRESS.md`

---

### Phase 2 — Data Ingestion & Validation

#### [NEW] `src/student_perf/data/ingest.py`
- Downloads UCI Student Performance dataset (`student-mat.csv`, `student-por.csv`) via URL or generates a realistic synthetic dataset if offline.
- Logs row/column counts.
- Saves raw file to `data/raw/`.

#### [NEW] `src/student_perf/data/validate.py`
- Pandera schema: column types, ranges (age 15–22, G1/G2/G3 0–20), categorical value sets.
- Saves `data/interim/validation_report.json`.

#### [NEW] `tests/test_data_validation.py`

---

### Phase 3 — Cleaning & Feature Engineering

#### [NEW] `src/student_perf/data/clean.py`
- Missing value imputation, deduplication, dtype correction.

#### [NEW] `src/student_perf/features/build_features.py`
- `grade_trend`, `avg_prior_grade`, `study_efficiency`, `support_index` + encodings.

#### [NEW] `src/student_perf/features/pipeline.py`
- `ColumnTransformer` + `Pipeline`, joblib persistence.
- Stratified train/val/test split → `data/processed/`.

#### [NEW] `tests/test_features.py`

---

### Phase 4 — MLflow Setup

#### [NEW] `src/student_perf/tracking/mlflow_utils.py`
- Helper: start run, log params/metrics/artifacts/model, tag by model + task type.

---

### Phase 5 — Classical ML Models

#### [NEW] `src/student_perf/models/base.py`
Abstract `BaseModel` with `fit`, `predict`, `predict_proba`, `save`, `load`, `get_feature_importance`.

#### [NEW] Individual model files:
- `logistic_regression.py`, `decision_tree.py`, `random_forest.py`
- `xgboost_model.py`, `catboost_model.py`, `lightgbm_model.py`

Each logs to MLflow, saves artifact, writes to `data/processed/metrics_summary.csv`.

#### [NEW] `tests/test_models.py`

---

### Phase 6 — MLP (PyTorch)

#### [NEW] `src/student_perf/models/mlp_torch.py`
- Input → [256, 128, 64] hidden layers (BatchNorm + Dropout 0.3) → output.
- Custom Dataset/DataLoader, early stopping, LR scheduler.
- Saves `.pt` + `BaseModel` wrapper.

---

### Phase 7 — LSTM (Temporal Extension)

#### [NEW] `src/student_perf/data/generate_temporal.py`
- Generates synthetic weekly quiz scores (10 weeks) per student anchored to G1/G2/G3 trend.
- Saves to `data/raw/student_temporal.csv`.

#### [NEW] `src/student_perf/models/lstm_torch.py`
- LSTM → dense head for pass/fail prediction from sequential input.

---

### Phase 8 — AutoML (FLAML)

#### [NEW] `src/student_perf/models/automl_runner.py`
- FLAML run with 180s budget.
- Logs best model to MLflow + `metrics_summary.csv`.

---

### Phase 9 — Evaluation & Comparison

#### [NEW] `src/student_perf/evaluation/metrics.py`
- Regression: RMSE, MAE, R². Classification: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.

#### [NEW] `src/student_perf/evaluation/comparison.py`
- Unified leaderboard sorted by F1/RMSE. Bootstrap CIs. Plotly chart.

---

### Phase 10 — SHAP Explainability

#### [NEW] `src/student_perf/explain/shap_explainer.py`
- `TreeExplainer`, `LinearExplainer`, `KernelExplainer`/`DeepExplainer` by model type.
- Summary plot, importance bar, 3 individual force/waterfall plots.
- Disk cache keyed by model + data version.

---

### Phase 11 — LIME Explainability

#### [NEW] `src/student_perf/explain/lime_explainer.py`
- `LimeTabularExplainer` for all models.
- Same 3 example students as SHAP.
- HTML output + normalized JSON.

#### [NEW] `src/student_perf/explain/explanation_utils.py`
- Shared helpers: normalize feature weights, top-N selection, SHAP vs LIME correlation.

#### [NEW] `tests/test_explainability.py`

---

### Phase 12 — REST API

#### [NEW] `src/student_perf/api/main.py`
- FastAPI with lifespan startup (model loading).
- Routes: `GET /health`, `GET /models`, `POST /predict`, `POST /predict/{model_name}`, `POST /explain`.

#### [NEW] `src/student_perf/api/schemas.py`
- Pydantic v2 request/response models with descriptions and examples.

#### [NEW] `src/student_perf/api/inference.py`
- Model loading, preprocessing, prediction, explanation assembly.

#### [NEW] `tests/test_api.py`

---

### Phase 13 — Streamlit Dashboard

#### [NEW] `src/student_perf/dashboard/app.py`
- 5 pages/tabs: Overview, Model Leaderboard, Single Prediction, Explainability Explorer, Experiment Tracking.
- `st.cache_data` / `st.cache_resource` throughout.

---

### Phase 14 — Dockerization

#### [NEW] `docker/Dockerfile.api`
#### [NEW] `docker/Dockerfile.dashboard`
#### [NEW] `docker/Dockerfile.train`
#### [NEW] `docker-compose.yml`
Shared volumes for `mlruns/` and `data/processed/`. Healthchecks. `depends_on`.

---

### Phase 15 — Deployment Prep

#### [MODIFY] `README.md`
Full deployment section: local docker-compose, cloud options (Render, Railway, AWS ECS, GCP Cloud Run, managed MLflow).

#### [NEW] `.github/workflows/ci.yml`
On push: ruff lint → pytest → docker build.

---

### Phase 16 — Documentation & Polish

#### [MODIFY] `README.md`
Full docs: motivation, architecture diagram (mermaid), setup, screenshots, model comparison table, known limitations.

#### [MODIFY] `PROGRESS.md`
Clean phase-by-phase changelog.

---

## Execution Order & Key Decisions

| # | Phase | Key decision |
|---|---|---|
| 1 | Scaffold + env | Install missing packages into LabX-Global venv |
| 2 | Ingest + validate | Try UCI download first; fall back to synthetic if offline |
| 3 | Clean + features | Use `category_encoders` for ordinal/target encoding |
| 4 | MLflow setup | Local file store (`./mlruns`), SQLite backend |
| 5 | Classical ML (6 models) | GridSearchCV for RF; Optuna for XGB; native cats for CatBoost/LGBM |
| 6 | MLP PyTorch | Use existing torch 2.5.1+cu121 |
| 7 | LSTM | Synthetic temporal data generated in `generate_temporal.py` |
| 8 | AutoML FLAML | 180s budget, binary classification default |
| 9 | Evaluation | Primary metric F1 (binary); bootstrap CIs (1000 samples) |
| 10 | SHAP | Cache SHAP values as `.pkl` in `data/processed/shap_cache/` |
| 11 | LIME | HTML + JSON outputs in `data/processed/lime_outputs/` |
| 12 | API | Models loaded at startup via lifespan; async endpoints |
| 13 | Dashboard | `requests` calls to API for predictions; direct file access for comparisons |
| 14 | Docker | Python 3.11-slim base images |
| 15–16 | Docs | README + CI workflow |

## Verification Plan

### Automated Tests
```
& "P:\LabX-Global\Scripts\pytest.exe" tests/ -v --cov=src --cov-report=term-missing
```
Target: >70% coverage across all test files.

### Manual Verification
- API: `curl http://localhost:8000/health` and `curl -X POST http://localhost:8000/predict`
- Dashboard: Open `http://localhost:8501` in browser
- MLflow UI: `mlflow ui` → `http://localhost:5000`
- Docker: `docker-compose up --build` → all services healthy
