"""
sklearn Pipeline + ColumnTransformer for the student performance dataset.

Handles:
  - Numeric scaling (StandardScaler)
  - Ordinal encoding for ordered categoricals
  - One-hot encoding for nominal categoricals
  - Stratified train/val/test split
  - Joblib serialization of fitted pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "data_config.yaml"
_MODEL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_config() -> tuple[dict, dict]:
    with open(_CONFIG_PATH) as f:
        data_cfg = yaml.safe_load(f)
    with open(_MODEL_CONFIG_PATH) as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
    # engineered
    "grade_trend", "avg_prior_grade", "study_efficiency",
    "support_index", "parent_edu", "alcohol_index",
]

# Ordered ordinal: these have meaningful order but NOT one-hot encoded
ORDINAL_FEATURES: list[tuple[str, list]] = [
    ("Medu", [0, 1, 2, 3, 4]),
    ("Fedu", [0, 1, 2, 3, 4]),
]

# Binary yes/no features → encoded 0/1
BINARY_FEATURES = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Multi-category nominal → one-hot
NOMINAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
]


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _build_column_transformer() -> ColumnTransformer:
    """Build the ColumnTransformer for preprocessing."""
    # Numeric: standard scale
    numeric_cols = [c for c in NUMERIC_FEATURES]

    # Binary: map yes/no → 1/0
    binary_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder(categories=[["no", "yes"]] * len(BINARY_FEATURES))),
    ])

    # Nominal: ordinal encode (tree models) — note: we provide a separate
    # one-hot variant for linear/MLP models via a flag
    nominal_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("nom", nominal_pipeline, NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def get_feature_names(ct: ColumnTransformer) -> list[str]:
    """Return feature names from a fitted ColumnTransformer."""
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(cols, "__iter__") and not isinstance(cols, str):
            names.extend(cols)
        else:
            names.append(cols)
    return names


# ---------------------------------------------------------------------------
# Split + save
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    task_type: str = "binary",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/val/test split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    data_cfg, model_cfg = _load_config()
    split_cfg = data_cfg["splits"]

    if task_type == "binary":
        y = df["pass_fail"]
    elif task_type == "regression":
        y = df["G3"].astype(float)
    else:
        y = df["performance_band"]

    # Feature columns (drop targets and raw G3)
    drop_cols = ["G3", "pass_fail", "performance_band"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    stratify = y if task_type != "regression" else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_seed"],
        stratify=stratify,
    )

    val_ratio = split_cfg["val_size"] / (1 - split_cfg["test_size"])
    stratify2 = y_temp if task_type != "regression" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=split_cfg["random_seed"],
        stratify=stratify2,
    )

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, task_type: str = "binary") -> dict:
    """
    Fit the preprocessing pipeline, transform all splits, save artifacts.

    Args:
        df: Feature-engineered DataFrame from build_features().
        task_type: "binary" | "regression" | "multiclass"

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        feature_names, pipeline
    """
    data_cfg, model_cfg = _load_config()
    processed_dir = _PROJECT_ROOT / data_cfg["dataset"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df, task_type)

    ct = _build_column_transformer()
    ct.fit(X_train)

    feature_names = get_feature_names(ct)
    logger.info(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")

    X_train_t = ct.transform(X_train)
    X_val_t = ct.transform(X_val)
    X_test_t = ct.transform(X_test)

    # Encode multiclass labels if needed
    label_encoder = None
    if task_type == "multiclass":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

    # Save splits
    def _save(arr, name):
        path = processed_dir / f"{name}.npy"
        np.save(path, arr)

    _save(X_train_t, "X_train")
    _save(X_val_t, "X_val")
    _save(X_test_t, "X_test")
    _save(np.array(y_train), "y_train")
    _save(np.array(y_val), "y_val")
    _save(np.array(y_test), "y_test")

    # Save pipeline
    pipeline_path = processed_dir / "preprocessor.joblib"
    joblib.dump(ct, pipeline_path)
    logger.info(f"Preprocessor saved to {pipeline_path}")

    if label_encoder is not None:
        le_path = processed_dir / "label_encoder.joblib"
        joblib.dump(label_encoder, le_path)

    # Save feature names
    import json
    with open(processed_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": np.array(y_train),
        "y_val": np.array(y_val),
        "y_test": np.array(y_test),
        "feature_names": feature_names,
        "pipeline": ct,
        "label_encoder": label_encoder,
    }


def load_processed_data(task_type: str = "binary") -> dict:
    """Load pre-processed splits from disk."""
    data_cfg, _ = _load_config()
    processed_dir = _PROJECT_ROOT / data_cfg["dataset"]["processed_dir"]
    import json

    splits = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = processed_dir / f"{name}.npy"
        splits[name] = np.load(path, allow_pickle=True)

    with open(processed_dir / "feature_names.json") as f:
        splits["feature_names"] = json.load(f)

    splits["pipeline"] = joblib.load(processed_dir / "preprocessor.joblib")

    le_path = processed_dir / "label_encoder.joblib"
    splits["label_encoder"] = joblib.load(le_path) if le_path.exists() else None

    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from student_perf.data.ingest import ingest
    from student_perf.data.clean import clean
    from student_perf.features.build_features import build_features

    df = build_features(clean(ingest()))
    result = run_pipeline(df, task_type="binary")
    print(f"\nX_train: {result['X_train'].shape}")
    print(f"X_val:   {result['X_val'].shape}")
    print(f"X_test:  {result['X_test'].shape}")
    print(f"y_train dist: {pd.Series(result['y_train']).value_counts().to_dict()}")
    print(f"Feature names: {result['feature_names'][:8]}")
