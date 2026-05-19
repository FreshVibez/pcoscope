"""Explainability utilities for PCOScope predictions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "matplotlib_cache"))


def get_feature_names(preprocessor, original_features: list[str]) -> list[str]:
    """Return transformed feature names when sklearn exposes them."""

    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return original_features


def global_feature_importance(model_bundle: dict, top_n: int = 15) -> pd.DataFrame:
    """Extract global feature importance from the best available model signal."""

    if "permutation_importance" in model_bundle:
        stored = model_bundle["permutation_importance"].copy()
        return stored.sort_values("importance", ascending=False).head(top_n)

    model = model_bundle["model"]
    preprocessor = model_bundle["preprocessor"]
    feature_names = get_feature_names(preprocessor, model_bundle["feature_columns"])

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("classifier", model)

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()
    elif hasattr(estimator, "final_estimator_") and hasattr(estimator.final_estimator_, "coef_"):
        importances = np.abs(estimator.final_estimator_.coef_).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).head(top_n)


def _pipeline_for_shap(model_bundle: dict):
    """Return a fitted pipeline suitable for SHAP when one is available."""

    model = model_bundle["model"]
    if hasattr(model, "named_steps"):
        return model

    best_name = model_bundle.get("best_model_name", "")
    if best_name.endswith(" (calibrated)"):
        base_name = best_name.replace(" (calibrated)", "")
        return model_bundle.get("trained_models", {}).get(base_name)

    return None


def _shap_estimator_from_pipeline(pipeline):
    """Extract a SHAP-compatible estimator from a fitted preprocessing pipeline."""

    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return None, None
    estimator = pipeline.named_steps.get("classifier")
    if estimator is None:
        return None, None
    compatible_names = ("Forest", "XGB", "GradientBoosting", "HistGradientBoosting")
    if any(name in estimator.__class__.__name__ for name in compatible_names):
        return estimator, "tree"
    if estimator.__class__.__name__ == "LogisticRegression":
        return estimator, "linear"
    return None, None


def _compute_shap_values(
    estimator,
    estimator_kind: str,
    transformed_array: np.ndarray,
    background_array: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Compute SHAP values for supported estimator families."""

    import shap

    if estimator_kind == "tree":
        explainer = shap.TreeExplainer(estimator)
        return _normalise_shap_values(explainer.shap_values(transformed_array)), "SHAP TreeExplainer"

    if estimator_kind == "linear":
        explainer = shap.LinearExplainer(estimator, background_array if background_array is not None else transformed_array)
        return _normalise_shap_values(explainer.shap_values(transformed_array)), "SHAP LinearExplainer"

    raise ValueError("Estimator is not supported by the configured SHAP explainers.")


def _normalise_shap_values(shap_values) -> np.ndarray:
    """Return positive-class SHAP values as a 2D array."""

    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def explain_prediction_with_shap(model_bundle: dict, patient_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Explain one patient-level prediction with SHAP for compatible tree models."""

    pipeline = _pipeline_for_shap(model_bundle)
    estimator, estimator_kind = _shap_estimator_from_pipeline(pipeline)
    if estimator is None:
        raise ValueError("SHAP is only enabled for compatible tree-based models and logistic regression.")

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(patient_df)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    background_df = model_bundle.get("shap_background")
    background_array = None
    if background_df is not None:
        background = preprocessor.transform(background_df)
        background_array = background.toarray() if hasattr(background, "toarray") else np.asarray(background)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = model_bundle["training_columns"]

    shap_input = background_array if estimator_kind == "linear" and background_array is not None else transformed_array
    shap_values, method = _compute_shap_values(estimator, estimator_kind, transformed_array, shap_input)
    contribution = np.abs(shap_values[0])

    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contribution,
            "method": method,
        }
    )
    return ranking.sort_values("contribution", ascending=False).head(top_n)


def generate_shap_artifacts(model, X_reference: pd.DataFrame, output_dir: str | Path, original_features: list[str]) -> dict:
    """Generate SHAP plots for a compatible tree-based fitted pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "named_steps"):
        return {"available": False, "reason": "Best model does not expose a direct sklearn pipeline."}

    estimator, estimator_kind = _shap_estimator_from_pipeline(model)
    if estimator is None:
        return {"available": False, "reason": "Best model is not compatible with the configured SHAP explainers."}

    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X_reference)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = original_features

    try:
        import matplotlib.pyplot as plt
        shap_values, method = _compute_shap_values(estimator, estimator_kind, transformed_array)

        summary_plot = output_path / "summary_plot.png"
        importance_plot = output_path / "feature_importance_plot.png"
        individual_plot = output_path / "individual_prediction_plot.png"

        mean_abs = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).head(15)

        plt.figure(figsize=(7, 5))
        plt.barh(mean_abs["feature"][::-1], mean_abs["mean_abs_shap"][::-1], color="#5a1f49")
        plt.xlabel("Mean absolute SHAP contribution")
        plt.title("SHAP summary")
        plt.tight_layout()
        plt.savefig(summary_plot, dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.barh(mean_abs["feature"][::-1], mean_abs["mean_abs_shap"][::-1], color="#8f4b76")
        plt.xlabel("Mean absolute SHAP contribution")
        plt.title("SHAP feature importance")
        plt.tight_layout()
        plt.savefig(importance_plot, dpi=160, bbox_inches="tight")
        plt.close()

        individual = pd.DataFrame(
            {
                "feature": feature_names,
                "abs_shap": np.abs(shap_values[0]),
            }
        ).sort_values("abs_shap", ascending=True).tail(10)
        plt.figure(figsize=(7, 4.8))
        plt.barh(individual["feature"], individual["abs_shap"], color="#5a1f49")
        plt.xlabel("Absolute SHAP contribution")
        plt.tight_layout()
        plt.savefig(individual_plot, dpi=160, bbox_inches="tight")
        plt.close()

        return {
            "available": True,
            "method": method,
            "summary_plot": str(summary_plot),
            "feature_importance_plot": str(importance_plot),
            "individual_prediction_plot": str(individual_plot),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def top_prediction_factors(model_bundle: dict, patient_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Estimate the top patient-level contributors.

    SHAP is used when installed and compatible. Otherwise, this falls back to a
    transparent approximation: global importance multiplied by the patient's
    transformed feature values.
    """

    try:
        return explain_prediction_with_shap(model_bundle, patient_df, top_n=top_n)
    except Exception:
        if "permutation_importance" in model_bundle:
            importance = global_feature_importance(model_bundle, top_n=len(model_bundle["training_columns"]))
            scores = []
            for _, row in importance.iterrows():
                feature = row["feature"]
                meta = model_bundle["input_summary"].get(feature, {})
                value = patient_df.iloc[0].get(feature, 0)
                if meta.get("type") == "numeric":
                    span = max(float(meta["max"]) - float(meta["min"]), 1e-9)
                    distance = abs(float(value) - float(meta["median"])) / span
                else:
                    distance = 1.0 if str(value) != str(meta.get("mode", value)) else 0.5
                scores.append(
                    {
                        "feature": feature,
                        "contribution": max(float(row["importance"]), 0.0) * (1.0 + distance),
                        "method": "permutation importance fallback",
                    }
                )
            return pd.DataFrame(scores).sort_values("contribution", ascending=False).head(top_n)

    model = model_bundle["model"]
    preprocessor = model_bundle["preprocessor"]
    feature_names = get_feature_names(preprocessor, model_bundle["feature_columns"])
    transformed = preprocessor.transform(patient_df)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    try:
        import shap

        estimator = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
        explainer = shap.Explainer(estimator, transformed_array)
        shap_values = explainer(transformed_array)
        values = np.abs(np.asarray(shap_values.values)[0])
        method = "SHAP"
    except Exception:
        global_importance = global_feature_importance(model_bundle, top_n=len(feature_names))
        importance_map = dict(zip(global_importance["feature"], global_importance["importance"]))
        values = np.array([importance_map.get(name, 0.0) for name in feature_names]) * np.abs(transformed_array[0])
        method = "importance-weighted contribution"

    ranking = pd.DataFrame({"feature": feature_names, "contribution": values})
    ranking = ranking.sort_values("contribution", ascending=False).head(top_n)
    ranking["method"] = method
    return ranking


def risk_category(probability: float) -> str:
    """Map a model probability into a clinician-friendly screening category."""

    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Moderate"
    return "High"


def recommended_follow_up(category: str) -> str:
    """Return non-diagnostic next steps appropriate to the risk band."""

    if category == "High":
        return (
            "Consider clinician review, targeted history, endocrine/metabolic labs, "
            "and ultrasound or specialist referral if clinically appropriate."
        )
    if category == "Moderate":
        return (
            "Consider follow-up assessment, review of menstrual and metabolic symptoms, "
            "and repeat or additional testing based on clinical judgement."
        )
    return "Continue routine care and reassess if symptoms change or clinical concern increases."
