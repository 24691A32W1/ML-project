"""
Data loading, cleaning and the leakage-safe preprocessing pipeline.

Leakage control
---------------
* Row-level cleaning (duplicates, invalid target) is done before the split and
  uses no statistics learned from the data.
* Everything that LEARNS from data (imputation values, outlier bounds, scaling
  parameters, one-hot categories, variance filter) lives inside a scikit-learn
  Pipeline and is fitted on the TRAINING set only.
* The target (Final_Exam_Marks) and Student_ID are never model inputs.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, ID_COL,
                     NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise outliers: clip each column to [Q1 - k*IQR, Q3 + k*IQR].

    Bounds are learned in fit() (training data only). Clipping keeps every row,
    which matters for a small dataset, while limiting the pull of extreme values.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor() -> Pipeline:
    """Impute -> clip outliers -> scale (numeric); impute -> one-hot (categorical);
    then drop zero-variance columns (simple feature selection)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("outliers", IQRClipper(factor=1.5)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("columns", columns),
        ("select", VarianceThreshold(threshold=1e-3)),
    ])


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns=[ID_COL]).duplicated().sum()),
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that learns nothing from the data."""
    df = df.copy()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COL])
    df = df.dropna(subset=[TARGET])                    # cannot learn from rows with no label
    df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)]   # invalid marks
    return df.reset_index(drop=True)


def get_split(df: pd.DataFrame):
    """80:20 split with a fixed seed. Returns X_train, X_test, y_train, y_test."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
