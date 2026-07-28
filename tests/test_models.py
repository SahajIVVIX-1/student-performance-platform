"""Unit tests for ML models implementation."""

import pytest
import numpy as np
from student_perf.models.logistic_regression import LogisticRegressionModel
from student_perf.models.decision_tree import DecisionTreeModel
from student_perf.models.random_forest import RandomForestModel
from student_perf.models.mlp_torch import MLPModel

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X_train = np.random.randn(80, 10)
    y_train = np.random.choice([0, 1], size=80)
    X_val = np.random.randn(20, 10)
    y_val = np.random.choice([0, 1], size=20)
    X_test = np.random.randn(20, 10)
    return X_train, y_train, X_val, y_val, X_test

def test_logistic_regression(dummy_data, tmp_path):
    X_tr, y_tr, X_v, y_v, X_te = dummy_data
    model = LogisticRegressionModel(task_type="binary")
    model.fit(X_tr, y_tr)
    
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)
    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 2)
    
    save_path = tmp_path / "lr.joblib"
    model.save(save_path)
    loaded = LogisticRegressionModel.load(save_path)
    np.testing.assert_array_equal(loaded.predict(X_te), preds)

def test_decision_tree(dummy_data):
    X_tr, y_tr, X_v, y_v, X_te = dummy_data
    model = DecisionTreeModel(task_type="binary")
    model.fit(X_tr, y_tr, X_v, y_v)
    preds = model.predict(X_te)
    assert len(preds) == len(X_te)

def test_random_forest(dummy_data):
    X_tr, y_tr, X_v, y_v, X_te = dummy_data
    model = RandomForestModel(task_type="binary")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    assert len(preds) == len(X_te)

def test_mlp_torch(dummy_data, tmp_path):
    X_tr, y_tr, X_v, y_v, X_te = dummy_data
    model = MLPModel(task_type="binary", max_epochs=5, patience=2)
    model.fit(X_tr, y_tr, X_v, y_v)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)
    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 2)

    save_path = tmp_path / "mlp.pt"
    model.save(save_path)
    loaded = MLPModel.load(save_path)
    np.testing.assert_array_equal(loaded.predict(X_te), preds)
