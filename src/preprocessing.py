"""Feature encoding, missing-value imputation and scaling for the
CDC/CCI surgical risk dataset.

Ported from Module 2 (EDA) and Module 3 (preprocessing) of
CDC_CCI_Model.ipynb, with one deliberate fix versus the notebook:
the KNNImputer / StandardScaler are now fit ONLY on the training split
and reused (via `SurgeryPreprocessor`) to transform validation/test/
inference data, instead of being fit on the full dataset before the
train/test split (which leaked test-set statistics into training).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.config import (
    ASA_MAPPING,
    CONTINUOUS_COLS,
    ID_COL,
    RAW_DATA_PATH,
    SURGERY_TYPE_MAPPING,
    TARGET_COL,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, non-data-dependent encoding -> safe to apply before split."""
    df = df.copy()
    if df["ASA_Score"].dtype == object:
        df["ASA_Score"] = df["ASA_Score"].map(ASA_MAPPING)
    if df["Surgery_Type"].dtype == object:
        df["Surgery_Type"] = df["Surgery_Type"].map(SURGERY_TYPE_MAPPING)
    return df


def split_features_target(df: pd.DataFrame):
    """Returns (X, y, visit_ids). Drops the ID column so it never leaks into training."""
    visit_ids = df[ID_COL] if ID_COL in df.columns else None
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns])
    return X, y, visit_ids


class SurgeryPreprocessor(BaseEstimator, TransformerMixin):
    """KNN-imputes missing labs then standard-scales continuous columns.

    Must be fit on TRAIN data only; call .transform() (never .fit()) on
    validation/test/inference data to avoid leakage.
    """

    def __init__(self, continuous_cols=None, n_neighbors: int = 5):
        self.continuous_cols = continuous_cols or CONTINUOUS_COLS
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_columns_ = list(X.columns)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        self.imputer_.fit(X)

        X_imputed = pd.DataFrame(
            self.imputer_.transform(X), columns=self.feature_columns_, index=X.index
        )
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imputed[self.continuous_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_columns_]
        X_imputed = pd.DataFrame(
            self.imputer_.transform(X), columns=self.feature_columns_, index=X.index
        )
        X_imputed[self.continuous_cols] = self.scaler_.transform(X_imputed[self.continuous_cols])
        return X_imputed
