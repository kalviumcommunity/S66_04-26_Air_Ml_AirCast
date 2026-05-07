from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_derived_features(
    df: pd.DataFrame,
    derived_feature_builders: Mapping[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Create reusable derived features from an input DataFrame.

    Args:
        df: Input data.
        derived_feature_builders: Mapping of output column names to callables.

    Returns:
        A new DataFrame containing the derived columns.
    """

    engineered = df.copy()
    if not derived_feature_builders:
        return engineered

    for feature_name, builder in derived_feature_builders.items():
        values = builder(engineered)
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=engineered.index, name=feature_name)
        engineered[feature_name] = values

    return engineered


def build_preprocessing_pipeline(
    categorical_columns: Sequence[str],
    numerical_columns: Sequence[str],
) -> ColumnTransformer:
    """Build a preprocessing pipeline for categorical and numerical data.

    Args:
        categorical_columns: Columns that require imputation and one-hot encoding.
        numerical_columns: Columns that require imputation and scaling.

    Returns:
        A fitted-ready ColumnTransformer.

    Raises:
        ValueError: If no columns are supplied.
    """

    if not categorical_columns and not numerical_columns:
        raise ValueError("At least one categorical or numerical column must be provided.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numerical_pipeline, list(numerical_columns)))

    return ColumnTransformer(transformers=transformers, remainder="drop")
