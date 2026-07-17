"""Pydantic request/response models for the surgical risk prediction API.

Field names and value domains mirror the raw dataset schema
(data/raw/mock_surgery_data.csv) exactly, since the saved pipeline
(src/train.build_pipeline) does its own encoding/imputation/scaling and
expects these raw values as input.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Patient age in years")
    BMI: float = Field(..., ge=10, le=60, description="Body mass index")
    ASA_Score: Literal["I", "II", "III", "IV", "V"] = Field(
        ..., description="ASA physical status classification"
    )
    Has_Diabetes: int = Field(..., ge=0, le=1, description="1 if diabetic, else 0")
    Has_HTN: int = Field(..., ge=0, le=1, description="1 if hypertensive, else 0")
    Surgery_Type: Literal["CT", "CC", "Chương trình", "Cấp cứu"] = Field(
        ..., description="CT/Chương trình = elective, CC/Cấp cứu = emergency"
    )
    PreOp_WBC: Optional[float] = Field(
        None, ge=0, le=50, description="Pre-op white blood cell count (10^9/L); omit if not yet available"
    )
    PreOp_Albumin: Optional[float] = Field(
        None, ge=0, le=10, description="Pre-op serum albumin (g/dL); omit if not yet available"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 62,
                "BMI": 22.0,
                "ASA_Score": "I",
                "Has_Diabetes": 0,
                "Has_HTN": 0,
                "Surgery_Type": "CT",
                "PreOp_WBC": 10.1,
                "PreOp_Albumin": None,
            }
        }
    }


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Predicted probability of a severe (CDC Grade IV/V) complication")
    high_risk_flag: int = Field(..., description="1 if risk_probability >= 0.5, else 0")
    model_version: str
    mlflow_run_id: Optional[str] = None
    predicted_at: str


class BatchPredictionRequest(BaseModel):
    patients: list[PatientFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class FeatureContribution(BaseModel):
    feature: str
    value: float = Field(..., description="Preprocessed (encoded/scaled) feature value seen by the model")
    shap_value: float = Field(..., description="Contribution to the model's raw output; sign/magnitude drive the risk score")


class ExplainResponse(BaseModel):
    risk_probability: float
    high_risk_flag: int
    base_value: float = Field(..., description="Expected model output over the background sample, before feature contributions")
    contributions: list[FeatureContribution] = Field(
        ..., description="Per-feature SHAP contributions, sorted by |shap_value| descending"
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    mlflow_run_id: Optional[str] = None
    metrics: dict[str, float]
    feature_schema: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
