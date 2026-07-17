"""SHAP explainability for the best trained pipeline: a global summary plot
and a local waterfall plot for a correctly-flagged high-risk (true positive)
patient.

Ported from Module 6 of CDC_CCI_Model.ipynb. Works for whichever model
src/train.py picked as best (tree or linear), since it explains the raw
classifier on top of the already-fitted preprocessing steps rather than
assuming XGBoost specifically.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_STATE, RAW_DATA_PATH, ROOT_DIR
from src.train import BEST_MODEL_PATH, load_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT_DIR / "reports" / "figures"


def transform_up_to_classifier(pipeline, X):
    """Runs X through every pipeline step except SMOTE and the classifier,
    returning the exact numeric feature matrix the classifier was trained on.

    Shared by the offline report generator below and by the API's
    /predict/explain endpoint (src/api/model_service.py), so both compute
    SHAP values against identically-preprocessed features."""
    X_transformed = X
    for _, step in pipeline.steps[:-2]:  # skip "smote" and "classifier"
        X_transformed = step.transform(X_transformed)
    return X_transformed


def run_explainability(
    model_path: Path = BEST_MODEL_PATH,
    data_path: Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pipeline from %s", model_path)
    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps["classifier"]

    _, X_test, _, y_test = load_and_split(data_path, test_size, random_state)
    X_test_processed = transform_up_to_classifier(pipeline, X_test)

    logger.info("Computing SHAP values for classifier=%s", type(classifier).__name__)
    explainer = shap.Explainer(classifier, X_test_processed)
    shap_values = explainer(X_test_processed)

    summary_path = REPORTS_DIR / "shap_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_processed, show=False)
    plt.title("SHAP Summary: feature impact on surgical complication risk")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    logger.info("Saved %s", summary_path)

    y_test_reset = y_test.reset_index(drop=True)
    y_pred_series = pd.Series(pipeline.predict(X_test)).reset_index(drop=True)
    true_positive_idx = y_test_reset[(y_test_reset == 1) & (y_pred_series == 1)].index

    waterfall_path = None
    if len(true_positive_idx) > 0:
        idx = true_positive_idx[0]
        waterfall_path = REPORTS_DIR / "shap_waterfall_true_positive.png"
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.title(f"SHAP Waterfall: true-positive high-risk patient (test idx {idx})")
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        logger.info("Saved %s", waterfall_path)
    else:
        logger.warning("No true-positive predictions found in test set; skipping waterfall plot")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="explainability"):
        mlflow.log_artifact(str(summary_path))
        if waterfall_path is not None:
            mlflow.log_artifact(str(waterfall_path))

    return REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    args = parser.parse_args()
    run_explainability(model_path=Path(args.model_path), data_path=Path(args.data_path))


if __name__ == "__main__":
    main()
