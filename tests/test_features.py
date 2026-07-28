"""Unit tests for feature engineering and preprocessing pipeline."""

import pytest
import numpy as np
import pandas as pd
from student_perf.data.ingest import generate_synthetic_dataset
from student_perf.data.clean import clean
from student_perf.features.build_features import build_features
from student_perf.features.pipeline import run_pipeline, make_splits, load_processed_data

def test_feature_engineering():
    raw_df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    clean_df = clean(raw_df)
    feat_df = build_features(clean_df)

    assert "grade_trend" in feat_df.columns
    assert "avg_prior_grade" in feat_df.columns
    assert "study_efficiency" in feat_df.columns
    assert "support_index" in feat_df.columns
    assert "pass_fail" in feat_df.columns
    assert "performance_band" in feat_df.columns

    # No missing values after cleaning & features
    assert feat_df.isnull().sum().sum() == 0

def test_pipeline_execution():
    raw_df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    clean_df = clean(raw_df)
    feat_df = build_features(clean_df)

    res = run_pipeline(feat_df, task_type="binary")

    assert "X_train" in res
    assert "X_val" in res
    assert "X_test" in res
    assert res["X_train"].shape[0] > 0
    assert not np.isnan(res["X_train"]).any()
    assert len(res["feature_names"]) == res["X_train"].shape[1]

def test_reproducibility():
    raw_df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    clean_df = clean(raw_df)
    feat_df = build_features(clean_df)

    res1 = run_pipeline(feat_df, task_type="binary")
    res2 = run_pipeline(feat_df, task_type="binary")

    np.testing.assert_array_equal(res1["X_train"], res2["X_train"])
    np.testing.assert_array_equal(res1["y_train"], res2["y_train"])
