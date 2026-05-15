from __future__ import annotations

from collections.abc import Sequence, Mapping

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compare_models(
    models: Mapping[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train and evaluate multiple models, returning a comparison table.
    
    Args:
        models: Dictionary mapping model names to estimator objects.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        
    Returns:
        A DataFrame containing evaluation metrics for each model.
    """

    results = []
    
    for name, model in models.items():
        # Clone to avoid modifying original object
        estimator = clone(model)
        estimator.fit(X_train, y_train)
        
        y_pred = estimator.predict(X_test)
        
        # Try to get probabilities for ROC-AUC
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None
            
        results.append({
            "Model": name,
            "F1": f1_score(y_test, y_pred, average="weighted"),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": auc,
        })
        
    return pd.DataFrame(results).sort_values("F1", ascending=False)


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


def tune_model_with_random_search(
    estimator,
    param_distributions: dict[str, Sequence],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    scoring: str = "f1",
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Tune any estimator with RandomizedSearchCV and return fitted search object."""

    safe_estimator = clone(estimator)
    search = RandomizedSearchCV(
        estimator=safe_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    return search


def get_search_results(search: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Return a DataFrame of search results sorted by rank.
    
    Args:
        search: A fitted GridSearchCV or RandomizedSearchCV object.
        
    Returns:
        A pandas DataFrame containing hyperparameter combinations and scores.
    """

    results_df = pd.DataFrame(search.cv_results_)
    # Extract only relevant columns: parameters, mean/std scores, and rank
    param_cols = [col for col in results_df.columns if col.startswith("param_")]
    score_cols = [
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    
    if "mean_train_score" in results_df.columns:
        score_cols.append("mean_train_score")
        
    return results_df[param_cols + score_cols].sort_values("rank_test_score")
