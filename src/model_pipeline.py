"""Scikit-learn modeling pipeline and threshold-based inference for appointment no-show prediction."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """Clips numeric features using IQR fences computed on the training partition."""

    def __init__(self, factor: float = 1.5, clip_min_zero: bool = True):
        self.factor = factor
        self.clip_min_zero = clip_min_zero
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - (self.factor * iqr)
        if self.clip_min_zero:
            lower = np.maximum(0.0, lower)
        self.lower_bounds_ = lower
        self.upper_bounds_ = q3 + (self.factor * iqr)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class BinaryMapper(BaseEstimator, TransformerMixin):
    """Maps binary categorical string indicators (e.g. Yes/No) to 1/0."""

    def __init__(self, positive_val: str = "Yes", negative_val: str = "No"):
        self.positive_val = positive_val
        self.negative_val = negative_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return (X_df == self.positive_val).astype(int).to_numpy()


def compute_zhang_yu_relative_risk(odds_ratio: float, baseline_prevalence: float) -> float:
    """Converts an odds ratio to relative risk for common outcomes: RR = OR / ((1 - P0) + (P0 * OR))."""
    return odds_ratio / ((1.0 - baseline_prevalence) + (baseline_prevalence * odds_ratio))


def build_production_pipeline() -> Pipeline:
    """Builds a scikit-learn preprocessing and Logistic Regression pipeline."""
    numeric_features = ["age", "fee_inr", "lead_days", "prior_no_shows"]
    distance_feature = ["distance_km"]
    categorical_features = ["booking_channel", "visit_type"]
    binary_features = ["reminder_sent"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    dist_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("capper", IQROutlierCapper(factor=1.5, clip_min_zero=True)),
        ("scaler", StandardScaler()),
    ])

    bin_pipeline = Pipeline([
        ("mapper", BinaryMapper(positive_val="Yes", negative_val="No")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("dist", dist_pipeline, distance_feature),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
            ("bin", bin_pipeline, binary_features),
        ],
        remainder="drop"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])


@dataclass
class ThresholdPredictor:
    """Wraps a trained model to predict binary risk flags at a custom probability threshold."""
    pipeline: Pipeline
    threshold: float = 0.35

    def predict_risk(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        flags = (probabilities >= self.threshold).astype(int)
        return probabilities, flags
