"""Unit tests for SHAP and LIME explainability modules."""

import pytest
import numpy as np
from student_perf.models.logistic_regression import LogisticRegressionModel
from student_perf.explain.shap_explainer import get_shap_values
from student_perf.explain.lime_explainer import get_lime_explainer, explain_instance
from student_perf.explain.explanation_utils import normalize_weights

@pytest.fixture
def fitted_lr_data():
    np.random.seed(42)
    X_tr = np.random.randn(50, 5)
    y_tr = np.random.choice([0, 1], size=50)
    X_te = np.random.randn(5, 5)
    feature_names = [f"feat_{i}" for i in range(5)]
    model = LogisticRegressionModel(task_type="binary")
    model.fit(X_tr, y_tr)
    return model, X_tr, X_te, feature_names

def test_shap_explainer(fitted_lr_data):
    model, X_tr, X_te, f_names = fitted_lr_data
    _, shap_vals = get_shap_values(
        model, X_tr, X_te, "test_lr", task_type="binary", use_cache=False
    )
    assert shap_vals is not None

def test_lime_explainer(fitted_lr_data):
    model, X_tr, X_te, f_names = fitted_lr_data
    explainer = get_lime_explainer(X_tr, f_names, task_type="binary")
    res = explain_instance(explainer, model, X_te[0], task_type="binary", num_features=3)
    assert "explanation" in res
    assert len(res["explanation"]) <= 3

def test_normalize_weights():
    weights = [("feat1", 0.5), ("feat2", -1.0), ("feat3", 0.2)]
    norm = normalize_weights(weights, top_n=2)
    assert len(norm) == 2
    assert norm[0]["feature"] == "feat2"
    assert norm[0]["weight_normalized"] == -1.0
