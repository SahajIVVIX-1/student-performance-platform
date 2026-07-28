"""Unit tests for data ingestion and validation."""

import pytest
import pandas as pd
from student_perf.data.ingest import generate_synthetic_dataset, ingest
from student_perf.data.validate import validate, build_schema

def test_generate_synthetic_dataset():
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    assert len(df) == 50
    assert "G3" in df.columns
    assert "age" in df.columns
    assert df["G3"].between(0, 20).all()
    assert df["age"].between(15, 22).all()

def test_data_validation_pass():
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    passed, report = validate(df, save_report=False)
    assert passed is True
    assert report["validation_passed"] is True
    assert len(report["errors"]) == 0

def test_data_validation_out_of_range():
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    df.loc[0, "age"] = 99  # Invalid age
    passed, report = validate(df, save_report=False)
    assert passed is False
    assert report["validation_passed"] is False

def test_data_validation_missing_column():
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    df = df.drop(columns=["age"])
    passed, report = validate(df, save_report=False)
    assert passed is False
    assert "Missing required columns" in report["errors"][0]
