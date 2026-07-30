"""
umashankar.core
================
Thin, friendly wrapper around Google's TabFM (zero-shot tabular foundation
model) so people can `pip install umashankar` and get going in 3 lines,
without touching TabFM's lower-level API directly.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

try:
    from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "umashankar requires the 'tabfm' package. Install it with: pip install tabfm"
    ) from e


_MODEL_CACHE = {}


def _load_backend():
    """Load and cache the underlying TabFM pretrained model (downloads once)."""
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = tabfm_v1_0_0.load()
    return _MODEL_CACHE["model"]


class Umashankar:
    """
    High-level zero-shot predictor for tabular data.

    Example
    -------
    >>> from umashankar import Umashankar
    >>> model = Umashankar(task="classification")
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.score(X_test, y_test)
    """

    def __init__(self, task: str = "classification"):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        backbone = _load_backend()
        if task == "classification":
            self._estimator = TabFMClassifier(model=backbone)
        else:
            self._estimator = TabFMRegressor(model=backbone)
        self._is_fit = False

    def fit(self, X, y):
        self._estimator.fit(X, y)
        self._is_fit = True
        return self

    def predict(self, X):
        self._check_fit()
        return self._estimator.predict(X)

    def predict_proba(self, X):
        self._check_fit()
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        return self._estimator.predict_proba(X)

    def score(self, X, y) -> dict:
        """Return a small dict of relevant metrics for the task type."""
        self._check_fit()
        preds = self.predict(X)
        if self.task == "classification":
            return {
                "accuracy": accuracy_score(y, preds),
                "f1_weighted": f1_score(y, preds, average="weighted"),
            }
        return {
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
        }

    def _check_fit(self):
        if not self._is_fit:
            raise RuntimeError("Call .fit(X, y) before predicting or scoring.")


def quick_run(df: pd.DataFrame, target: str, task: str = "classification", test_size: float = 0.2):
    """
    Convenience one-liner: split a dataframe, fit, predict, and score.
    Returns (model, metrics_dict, X_test, y_test, preds).
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = Umashankar(task=task)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = model.score(X_test, y_test)
    return model, metrics, X_test, y_test, preds
