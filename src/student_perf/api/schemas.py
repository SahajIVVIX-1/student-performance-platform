"""
Pydantic v2 request/response models for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class StudentFeatures(BaseModel):
    """Input features for a student prediction request."""

    school: str = Field("GP", description="School code: 'GP' or 'MS'", examples=["GP"])
    sex: str = Field("F", description="Student sex: 'F' or 'M'", examples=["F"])
    age: int = Field(17, ge=15, le=22, description="Age (15–22)", examples=[17])
    address: str = Field("U", description="Address type: 'U' (urban) or 'R' (rural)", examples=["U"])
    famsize: str = Field("GT3", description="Family size: 'LE3' or 'GT3'", examples=["GT3"])
    Pstatus: str = Field("T", description="Parent cohabitation: 'T' (together) or 'A' (apart)", examples=["T"])
    Medu: int = Field(3, ge=0, le=4, description="Mother's education level (0–4)", examples=[3])
    Fedu: int = Field(2, ge=0, le=4, description="Father's education level (0–4)", examples=[2])
    Mjob: str = Field("other", description="Mother's job", examples=["teacher"])
    Fjob: str = Field("other", description="Father's job", examples=["other"])
    reason: str = Field("course", description="Reason for school choice", examples=["course"])
    guardian: str = Field("mother", description="Student guardian", examples=["mother"])
    traveltime: int = Field(2, ge=1, le=4, description="Travel time to school (1–4)", examples=[2])
    studytime: int = Field(2, ge=1, le=4, description="Weekly study time (1=<2h, 4=>10h)", examples=[2])
    failures: int = Field(0, ge=0, le=4, description="Number of past failures", examples=[0])
    schoolsup: str = Field("no", description="Extra school support: 'yes'/'no'", examples=["no"])
    famsup: str = Field("yes", description="Family educational support: 'yes'/'no'", examples=["yes"])
    paid: str = Field("no", description="Extra paid classes: 'yes'/'no'", examples=["no"])
    activities: str = Field("no", description="Extra-curricular activities: 'yes'/'no'", examples=["no"])
    nursery: str = Field("yes", description="Attended nursery school: 'yes'/'no'", examples=["yes"])
    higher: str = Field("yes", description="Wants higher education: 'yes'/'no'", examples=["yes"])
    internet: str = Field("yes", description="Internet access at home: 'yes'/'no'", examples=["yes"])
    romantic: str = Field("no", description="In a romantic relationship: 'yes'/'no'", examples=["no"])
    famrel: int = Field(4, ge=1, le=5, description="Family relationship quality (1–5)", examples=[4])
    freetime: int = Field(3, ge=1, le=5, description="Free time after school (1–5)", examples=[3])
    goout: int = Field(2, ge=1, le=5, description="Going out with friends (1–5)", examples=[2])
    Dalc: int = Field(1, ge=1, le=5, description="Workday alcohol consumption (1–5)", examples=[1])
    Walc: int = Field(2, ge=1, le=5, description="Weekend alcohol consumption (1–5)", examples=[2])
    health: int = Field(4, ge=1, le=5, description="Current health status (1–5)", examples=[4])
    absences: int = Field(2, ge=0, le=93, description="Number of school absences", examples=[2])
    G1: int = Field(12, ge=0, le=20, description="First period grade (0–20)", examples=[12])
    G2: int = Field(13, ge=0, le=20, description="Second period grade (0–20)", examples=[13])


class PredictRequest(BaseModel):
    features: StudentFeatures
    model_name: str | None = Field(None, description="Model to use (None = default best model)")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    model_used: str
    prediction: int | float
    prediction_label: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class FeatureContribution(BaseModel):
    feature: str
    weight: float
    weight_normalized: float


class ExplanationResponse(BaseModel):
    model_used: str
    prediction: int | float
    shap_contributions: list[FeatureContribution]
    lime_contributions: list[FeatureContribution]
    shap_lime_rank_correlation: float | None = None


class ModelInfo(BaseModel):
    name: str
    task_type: str
    metrics: dict[str, Any]
    available: bool


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
