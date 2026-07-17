"""Train and evaluate 3 candidate models (Logistic Regression, Random Forest,
XGBoost) for surgical complication risk prediction, with MLflow experiment
tracking and a single saved end-to-end inference pipeline for the best model.

Ported from Module 4 (SMOTE) and Module 5 (training/evaluation) of
CDC_CCI_Model.ipynb. Two fixes versus the notebook:
  1. Preprocessing (KNN impute + scale) is fit on the train split only
     (see src/preprocessing.SurgeryPreprocessor), not on the full dataset.
  2. SMOTE and preprocessing are wrapped into a single imblearn Pipeline per
     model, so the saved artifact takes raw feature rows in and returns a
     risk prediction/probability out -- no separate preprocessing step has
     to be re-implemented by whatever serves the model (see src/api).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    RAW_DATA_PATH,
    ROOT_DIR,
)
from src.preprocessing import (
    SurgeryPreprocessor,
    encode_categoricals,
    load_raw_data,
    split_features_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"
SHAP_BACKGROUND_SIZE = 100


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=random_state),
    }


def build_pipeline(classifier, random_state: int = RANDOM_STATE) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("encode", FunctionTransformer(encode_categoricals)),
            ("preprocess", SurgeryPreprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", classifier),
        ]
    )


def evaluate(pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def load_and_split(
    data_path: Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Loads raw data and returns a stratified train/test split of the RAW
    (unencoded) features, so callers (training, explainability, tests) all
    see identical splits without duplicating the split logic."""
    df = load_raw_data(data_path)
    X, y, _ = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def run_training(
    data_path: Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Loading raw data from %s", data_path)
    X_train, X_test, y_train, y_test = load_and_split(data_path, test_size, random_state)
    logger.info("Train size=%d, Test size=%d, positive rate train=%.1f%%",
                len(X_train), len(X_test), y_train.mean() * 100)

    results = []
    fitted_pipelines = {}
    run_ids = {}

    with mlflow.start_run(run_name="training_run") as parent_run:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        for name, classifier in build_candidate_models(random_state).items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                logger.info("Training %s ...", name)
                pipeline = build_pipeline(classifier, random_state)
                pipeline.fit(X_train, y_train)

                metrics = evaluate(pipeline, X_test, y_test)
                mlflow.log_params({f"classifier": name})
                mlflow.log_metrics(metrics)
                # pickle (not skops) is required: the pipeline includes our
                # own SurgeryPreprocessor/encode_categoricals plus imblearn
                # SMOTE, which skops' trusted-type allowlist rejects.
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

                fitted_pipelines[name] = pipeline
                run_ids[name] = child_run.info.run_id
                results.append({"model": name, **metrics})
                logger.info("%s -> %s", name, metrics)

        results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
        best_name = results_df.iloc[0]["model"]
        best_pipeline = fitted_pipelines[best_name]
        best_metrics = results_df.iloc[0].drop("model").astype(float).to_dict()

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_roc_auc", results_df.iloc[0]["roc_auc"])

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, BEST_MODEL_PATH)
        mlflow.log_artifact(str(BEST_MODEL_PATH))

        # Small background sample (post encode+impute+scale, pre-SMOTE/classifier)
        # for the API's /predict/explain endpoint, so it doesn't need to
        # reload and re-split the full dataset just to build a SHAP explainer.
        X_train_transformed = X_train
        for _, step in best_pipeline.steps[:-2]:  # skip "smote" and "classifier"
            X_train_transformed = step.transform(X_train_transformed)
        background_sample = X_train_transformed.sample(
            n=min(SHAP_BACKGROUND_SIZE, len(X_train_transformed)), random_state=random_state
        )
        joblib.dump(background_sample, BACKGROUND_PATH)
        mlflow.log_artifact(str(BACKGROUND_PATH))

        metadata = {
            "model_name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run_ids[best_name],
            "metrics": best_metrics,
            "feature_schema": list(X_train.columns),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(METADATA_PATH))

        logger.info("Best model: %s (ROC-AUC=%.3f) saved to %s",
                    best_name, results_df.iloc[0]["roc_auc"], BEST_MODEL_PATH)

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    results_df = run_training(
        data_path=Path(args.data_path), test_size=args.test_size, random_state=args.random_state
    )
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
