"""
Feature engineering module.

Builds derived features from the raw/cleaned student performance data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "data_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def add_grade_trend(df: pd.DataFrame) -> pd.DataFrame:
    """grade_trend = G2 - G1: positive means improving."""
    df = df.copy()
    df["grade_trend"] = df["G2"] - df["G1"]
    return df


def add_avg_prior_grade(df: pd.DataFrame) -> pd.DataFrame:
    """avg_prior_grade = mean(G1, G2)."""
    df = df.copy()
    df["avg_prior_grade"] = (df["G1"] + df["G2"]) / 2.0
    return df


def add_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """study_efficiency = studytime / (absences + 1): high = efficient."""
    df = df.copy()
    df["study_efficiency"] = df["studytime"] / (df["absences"] + 1)
    return df


def add_support_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    support_index: count of support factors
    (famsup == 'yes') + (schoolsup == 'yes') + (paid == 'yes')
    """
    df = df.copy()
    df["support_index"] = (
        (df["famsup"] == "yes").astype(int)
        + (df["schoolsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )
    return df


def add_parental_education(df: pd.DataFrame) -> pd.DataFrame:
    """Combined parental education level."""
    df = df.copy()
    df["parent_edu"] = (df["Medu"] + df["Fedu"]) / 2.0
    return df


def add_alcohol_index(df: pd.DataFrame) -> pd.DataFrame:
    """Combined weekday + weekend alcohol consumption."""
    df = df.copy()
    df["alcohol_index"] = (df["Dalc"] + df["Walc"]) / 2.0
    return df


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------

def add_target_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add binary and multiclass target columns."""
    df = df.copy()
    task_cfg = cfg["task"]
    threshold = task_cfg["binary_threshold"]

    # Binary: pass=1, fail=0
    df["pass_fail"] = (df["G3"] >= threshold).astype(int)

    # Multiclass: Low/Medium/High
    bands = task_cfg["multiclass_bands"]
    def assign_band(g3):
        for label, (lo, hi) in bands.items():
            if lo <= g3 <= hi:
                return label
        return "Low"
    df["performance_band"] = df["G3"].apply(assign_band)

    return df


# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to cleaned DataFrame.

    Args:
        df: Cleaned DataFrame from clean.py.

    Returns:
        DataFrame with all engineered features + target columns added.
    """
    cfg = _load_config()
    fe_cfg = cfg["feature_engineering"]

    if fe_cfg.get("grade_trend"):
        df = add_grade_trend(df)
    if fe_cfg.get("avg_prior_grade"):
        df = add_avg_prior_grade(df)
    if fe_cfg.get("study_efficiency"):
        df = add_study_efficiency(df)
    if fe_cfg.get("support_index"):
        df = add_support_index(df)

    # Always add these
    df = add_parental_education(df)
    df = add_alcohol_index(df)
    df = add_target_columns(df, cfg)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from student_perf.data.ingest import ingest
    from student_perf.data.clean import clean
    df = build_features(clean(ingest()))
    print(f"\nFeatures shape: {df.shape}")
    print(f"New columns: {[c for c in df.columns if c not in ['school','sex','age']][-10:]}")
