"""Loads the trained pipeline once and serves predictions/explanations to the API.

A module-level singleton (`model_service`) is loaded at FastAPI startup
(see src/api/main.py) and reused across requests, so the pickle/SHAP
explainer setup cost is paid once, not per request.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import joblib
import pandas as pd
import shap

from src.explain import transform_up_to_classifier
from src.train import BACKGROUND_PATH, BEST_MODEL_PATH, METADATA_PATH

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}
        self.explainer: Optional[shap.Explainer] = None

    def load(self) -> None:
        logger.info("Loading pipeline from %s", BEST_MODEL_PATH)
        self.pipeline = joblib.load(BEST_MODEL_PATH)

        if METADATA_PATH.exists():
            self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        else:
            logger.warning("No metadata file found at %s", METADATA_PATH)
            self.metadata = {}

        if BACKGROUND_PATH.exists():
            background = joblib.load(BACKGROUND_PATH)
            classifier = self.pipeline.named_steps["classifier"]
            self.explainer = shap.Explainer(classifier, background)
            logger.info("SHAP explainer ready (background size=%d)", len(background))
        else:
            logger.warning("No SHAP background sample found at %s; /predict/explain will be unavailable", BACKGROUND_PATH)
            self.explainer = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict_one(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        proba = float(self.pipeline.predict_proba(df)[0, 1])
        return {"risk_probability": proba, "high_risk_flag": int(proba >= 0.5)}

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        df = pd.DataFrame(features_list)
        probas = self.pipeline.predict_proba(df)[:, 1]
        return [{"risk_probability": float(p), "high_risk_flag": int(p >= 0.5)} for p in probas]

    def explain_one(self, features: dict) -> dict:
        if self.explainer is None:
            raise RuntimeError("Explainer not available (missing SHAP background sample)")

        df = pd.DataFrame([features])
        processed = transform_up_to_classifier(self.pipeline, df)
        shap_values = self.explainer(processed)
        proba = float(self.pipeline.predict_proba(df)[0, 1])

        contributions = [
            {
                "feature": col,
                "value": float(processed.iloc[0][col]),
                "shap_value": float(shap_values.values[0][i]),
            }
            for i, col in enumerate(processed.columns)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        base_value = shap_values.base_values[0]
        base_value = float(base_value[0] if hasattr(base_value, "__len__") else base_value)

        return {
            "risk_probability": proba,
            "high_risk_flag": int(proba >= 0.5),
            "base_value": base_value,
            "contributions": contributions,
        }


model_service = ModelService()
