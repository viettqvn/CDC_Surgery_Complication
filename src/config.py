"""Shared paths and constants for the CDC/CCI surgical risk pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "mock_surgery_data.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "surgery_data_cleaned.csv"

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT_NAME = "cdc_cci_surgical_risk"

TARGET_COL = "High_Risk_Flag"
ID_COL = "Visit_ID"

ASA_MAPPING = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
SURGERY_TYPE_MAPPING = {"Chương trình": 0, "Cấp cứu": 1, "CT": 0, "CC": 1}

CONTINUOUS_COLS = ["Age", "BMI", "PreOp_WBC", "PreOp_Albumin"]
RANDOM_STATE = 42

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
