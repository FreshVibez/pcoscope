"""Train and save PCOScope screening models."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .evaluate import evaluate_classifier, evaluate_model_cv, metrics_to_frame
    from .preprocess import build_preprocessor, load_clean_bundle
    from .explain import generate_shap_artifacts
except ImportError:
    from evaluate import evaluate_classifier, evaluate_model_cv, metrics_to_frame
    from preprocess import build_preprocessor, load_clean_bundle
    from explain import generate_shap_artifacts


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pcos_dataset.csv"
MODEL_PATH = ROOT / "models" / "saved_model.pkl"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SHAP_OUTPUT_DIR = OUTPUTS_DIR / "shap"
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
RANDOM_STATE = 42

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / "matplotlib_cache"))


def _boosting_model():
    """Return XGBoost when available; otherwise use sklearn gradient boosting.

    Some macOS setups need the OpenMP runtime before XGBoost can load. The
    fallback keeps the hackathon demo reproducible while preserving the same
    boosted-tree modelling intent.
    """

    try:
        from xgboost import XGBClassifier

        return (
            "XGBoost",
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            True,
        )
    except Exception:
        return (
            "Gradient Boosting (XGBoost fallback)",
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=RANDOM_STATE),
            False,
        )


def build_candidate_models(numeric_features: list[str], categorical_features: list[str]) -> dict:
    """Build all candidate sklearn pipelines."""

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }

    boosting_name, boosting_estimator, using_xgboost = _boosting_model()
    boosted = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", boosting_estimator)])
    stacking = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                StackingClassifier(
                    estimators=[
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=250,
                                min_samples_leaf=4,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                        ("boosted", _boosting_model()[1]),
                    ],
                    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                    cv=5,
                    stack_method="predict_proba",
                ),
            ),
        ]
    )
    models[boosting_name] = boosted
    ensemble_name = "Stacking Ensemble (RF + XGBoost)" if using_xgboost else "Stacking Ensemble (RF + boosted-tree fallback)"
    models[ensemble_name] = stacking

    return models


def choose_best_model_medical(results: dict[str, dict]) -> str:
    """Choose the safest screening model by recall, then F1, then ROC-AUC."""

    table = metrics_to_frame(results)
    return str(table.iloc[0]["model"])


def choose_best_model(results: dict[str, dict]) -> str:
    """Backward-compatible wrapper for the medical ranking rule."""

    return choose_best_model_medical(results)


def _calibrated_classifier(model, cv: int = 5) -> CalibratedClassifierCV:
    """Create a sigmoid-calibrated copy of a sklearn pipeline."""

    try:
        return CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=clone(model), method="sigmoid", cv=cv)


def _is_calibration_candidate(model_name: str) -> bool:
    """Return whether a model should get a calibrated probability variant."""

    lower_name = model_name.lower()
    return "random forest" in lower_name or "xgboost" in lower_name or "gradient boosting" in lower_name


def _ensure_output_dirs() -> None:
    """Create all output folders used by the training pipeline."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _model_for_metadata(model_name: str, all_models: dict, trained_models: dict):
    """Return a fitted pipeline that exposes preprocessing metadata."""

    model = all_models[model_name]
    if hasattr(model, "named_steps"):
        return model
    if model_name.endswith(" (calibrated)"):
        base_name = model_name.replace(" (calibrated)", "")
        return trained_models.get(base_name, model)
    return model


def train_best_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    """Train candidate models, evaluate them, and persist the best model bundle."""

    _ensure_output_dirs()
    bundle = load_clean_bundle(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        stratify=bundle.y,
        random_state=RANDOM_STATE,
    )

    candidates = build_candidate_models(bundle.numeric_features, bundle.categorical_features)
    using_xgboost = any(name == "XGBoost" for name in candidates)
    trained_models = {}
    calibrated_models = {}
    results = {}
    cv_results = {}
    calibration_comparison = {}

    for name, model in candidates.items():
        cv_results[name] = evaluate_model_cv(model, bundle.X, bundle.y)
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = {
            **evaluate_classifier(model, X_test, y_test),
            **cv_results[name],
        }

        if _is_calibration_candidate(name):
            try:
                calibrated = _calibrated_classifier(model)
                calibrated.fit(X_train, y_train)
                calibrated_name = f"{name} (calibrated)"
                calibrated_models[calibrated_name] = calibrated
                calibrated_metrics = {
                    **evaluate_classifier(calibrated, X_test, y_test),
                    **evaluate_model_cv(_calibrated_classifier(model), bundle.X, bundle.y),
                }
                results[calibrated_name] = calibrated_metrics
                calibration_comparison[name] = {
                    "uncalibrated_roc_auc": results[name]["roc_auc"],
                    "calibrated_roc_auc": calibrated_metrics["roc_auc"],
                    "roc_auc_delta": calibrated_metrics["roc_auc"] - results[name]["roc_auc"],
                }
            except Exception as exc:
                calibration_comparison[name] = {
                    "uncalibrated_roc_auc": results[name]["roc_auc"],
                    "calibrated_roc_auc": None,
                    "roc_auc_delta": None,
                    "error": str(exc),
                }

    best_name = choose_best_model_medical(results)
    all_models = {**trained_models, **calibrated_models}
    best_model = all_models[best_name]
    metadata_model = _model_for_metadata(best_name, all_models, trained_models)
    metrics_table = metrics_to_frame(results)
    metrics_table.to_csv(MODEL_COMPARISON_PATH, index=False)

    for name, model in all_models.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")

    best_preprocessor = metadata_model.named_steps["preprocessor"] if hasattr(metadata_model, "named_steps") else None
    permutation = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        scoring="f1",
        random_state=RANDOM_STATE,
    )

    model_bundle = {
        "model": best_model,
        "preprocessor": best_preprocessor,
        "best_model_name": best_name,
        "trained_models": trained_models,
        "calibrated_models": calibrated_models,
        "metrics": results,
        "metrics_table": metrics_table,
        "calibration_comparison": calibration_comparison,
        "model_comparison_path": str(MODEL_COMPARISON_PATH),
        "feature_columns": bundle.feature_columns,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "training_columns": bundle.X.columns.tolist(),
        "input_summary": _input_summary(bundle.X),
        "shap_background": X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE),
        "using_xgboost": using_xgboost,
        "boosting_status": (
            "XGBoost loaded successfully."
            if using_xgboost
            else "XGBoost was unavailable locally, so sklearn histogram gradient boosting was used as a boosted-tree fallback."
        ),
        "explainability_method": "Global permutation importance plus SHAP patient-level explanations when compatible, with a permutation-based fallback.",
        "permutation_importance": pd.DataFrame(
            {
                "feature": bundle.X.columns,
                "importance": permutation.importances_mean,
            }
        ).sort_values("importance", ascending=False),
        "shap_artifacts": {},
    }

    model_bundle["shap_artifacts"] = {
        "available": False,
        "reason": "Training keeps SHAP plot generation optional for deployment stability. Patient-level SHAP explanations are generated in-app when compatible.",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    return model_bundle


def _input_summary(X: pd.DataFrame) -> dict:
    """Capture values needed to build safe Streamlit defaults."""

    summary = {}
    for col in X.columns:
        series = X[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            summary[col] = {
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "unique": sorted(series.unique().tolist()) if series.nunique() <= 8 else None,
            }
        else:
            summary[col] = {
                "type": "categorical",
                "values": sorted([str(value) for value in series.unique().tolist()]),
                "mode": str(series.mode().iloc[0]) if not series.mode().empty else "",
            }
    return summary


if __name__ == "__main__":
    bundle = train_best_model()
    print(f"Saved best model: {bundle['best_model_name']} -> {MODEL_PATH}")
    print(bundle["metrics_table"].to_string(index=False))
