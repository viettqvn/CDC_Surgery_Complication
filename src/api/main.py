"""FastAPI service exposing the surgical risk pipeline for prediction,
explanation, health/metrics (ops), and a static demo web UI.

Run locally with:
    uvicorn src.api.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
Web demo:   http://localhost:8000/
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.model_service import model_service
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WEB_DIR = ROOT_DIR / "web"

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made, by risk flag", ["high_risk_flag"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="CDC/CCI Surgical Risk Prediction API",
    description=(
        "Predicts, from pre-operative patient data, the probability of a "
        "severe (Clavien-Dindo Grade IV/V) post-surgical complication -- "
        "an early-warning signal computed before surgery rather than the "
        "retrospective CDC grading itself."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNT.labels(
        endpoint=endpoint, method=request.method, status_code=response.status_code
    ).inc()
    return response


def _require_model() -> None:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.is_loaded else "unavailable",
        model_loaded=model_service.is_loaded,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    _require_model()
    meta = model_service.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        trained_at=meta.get("trained_at", ""),
        mlflow_run_id=meta.get("mlflow_run_id"),
        metrics=meta.get("metrics", {}),
        feature_schema=meta.get("feature_schema", []),
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: PatientFeatures) -> PredictionResponse:
    _require_model()
    result = model_service.predict_one(features.model_dump())
    PREDICTION_COUNT.labels(high_risk_flag=str(result["high_risk_flag"])).inc()
    return PredictionResponse(
        **result,
        model_version=model_service.metadata.get("model_name", "unknown"),
        mlflow_run_id=model_service.metadata.get("mlflow_run_id"),
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    _require_model()
    results = model_service.predict_batch([p.model_dump() for p in payload.patients])
    now = datetime.now(timezone.utc).isoformat()

    predictions = []
    for result in results:
        PREDICTION_COUNT.labels(high_risk_flag=str(result["high_risk_flag"])).inc()
        predictions.append(
            PredictionResponse(
                **result,
                model_version=model_service.metadata.get("model_name", "unknown"),
                mlflow_run_id=model_service.metadata.get("mlflow_run_id"),
                predicted_at=now,
            )
        )
    return BatchPredictionResponse(predictions=predictions)


@app.post("/api/v1/predict/explain", response_model=ExplainResponse, tags=["explainability"])
def predict_explain(features: PatientFeatures) -> ExplainResponse:
    _require_model()
    try:
        result = model_service.explain_one(features.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplainResponse(**result)


# Mounted last so it never shadows the API/ops routes above; serves web/index.html at "/".
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
