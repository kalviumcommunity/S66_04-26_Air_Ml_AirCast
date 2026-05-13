from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    metric: str = "minkowski",
) -> Pipeline:
    """Train a scaled KNN classification pipeline."""

    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric),
            ),
        ]
    )
    knn_pipeline.fit(X_train, y_train)
    return knn_pipeline


def train_decision_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    max_depth: int | None = 6,
    min_samples_leaf: int = 2,
    min_samples_split: int = 4,
) -> DecisionTreeClassifier:
    """Train a constrained Decision Tree classifier."""

    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importance(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Return impurity and permutation importances by feature name."""

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Provided model does not expose feature_importances_.")

    impurity_scores = {
        feature: float(score)
        for feature, score in zip(X_valid.columns, model.feature_importances_)
    }

    perm = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=10,
        random_state=random_state,
    )
    permutation_scores = {
        feature: float(score)
        for feature, score in zip(X_valid.columns, perm.importances_mean)
    }

    return {
        "impurity": impurity_scores,
        "permutation": permutation_scores,
    }


def tune_model_with_grid_search(
    estimator,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1",
    cv: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Tune any estimator with GridSearchCV and return fitted search object."""

    safe_estimator = clone(estimator)
    search = GridSearchCV(
        estimator=safe_estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    return search
