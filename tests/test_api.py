"""Unit tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from student_perf.api.main import app
from student_perf.models.logistic_regression import LogisticRegressionModel
import numpy as np

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_list_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data

def test_predict_validation_error():
    payload = {
        "features": {
            "age": 999  # invalid age > 22
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
