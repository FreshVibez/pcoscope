"""Evaluation helpers for PCOS risk-screening models."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall_sensitivity": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return screening-focused model metrics.

    Recall is treated as especially important because a missed high-risk patient
    is more concerning than a false positive in a screening workflow.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def metrics_to_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Convert nested metric dictionaries into a display-ready table."""

    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({key: value for key, value in metrics.items() if key != "confusion_matrix"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["recall_sensitivity", "f1", "roc_auc"], ascending=False)


def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    """Evaluate a pipeline with stratified cross-validation.

    The full sklearn pipeline is passed to cross-validation so imputation,
    scaling, encoding, and model fitting happen inside each fold. This prevents
    data leakage from validation folds into preprocessing.
    """

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    return summarize_cv_results(scores)


def summarize_cv_results(cv_scores: dict) -> dict:
    """Return mean and standard deviation for each cross-validation metric."""

    summary = {}
    for metric in SCORING:
        values = cv_scores[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = float(values.mean())
        summary[f"cv_{metric}_std"] = float(values.std())
    return summary
