from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
    class_weight: str | dict[str, float] | None = None,
) -> RandomForestClassifier:
    """Fit a classification model on processed training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed used by the estimator.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth, or None for unrestricted depth.
        class_weight: Weights associated with classes in the form {class_label: weight}.

    Returns:
        A fitted RandomForestClassifier instance.
    """

    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model
