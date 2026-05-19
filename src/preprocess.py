"""Data loading and preprocessing utilities for PCOScope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "PCOS (Y/N)"
DROP_COLUMNS = ["Sl. No", "Patient File No."]


@dataclass
class DatasetBundle:
    """Container for cleaned model inputs and metadata."""

    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the clinical PCOS dataset from CSV."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, normalize column names, and coerce obvious numbers."""

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        if converted.notna().sum() > 0:
            cleaned[col] = converted

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> DatasetBundle:
    """Separate model features from the PCOS risk label."""

    feature_df = df.drop(columns=[TARGET_COLUMN])
    feature_df = feature_df.drop(columns=[col for col in DROP_COLUMNS if col in feature_df], errors="ignore")
    y = df[TARGET_COLUMN]

    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in feature_df.columns if col not in numeric_features]

    return DatasetBundle(
        X=feature_df,
        y=y,
        feature_columns=feature_df.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a sklearn preprocessor for missing values, scaling, and encoding."""

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def load_clean_bundle(csv_path: str | Path) -> DatasetBundle:
    """Load, clean, and split the dataset in one beginner-friendly call."""

    df = clean_dataset(load_dataset(csv_path))
    return split_features_target(df)
