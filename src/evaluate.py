from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute evaluation metrics for a trained model.

    Args:
        model: A fitted estimator that exposes predict and optionally predict_proba.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        A dictionary of metric names mapped to scores.
    """

    predictions = model.predict(X_test)
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])

    return metrics


def compute_confusion_details(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, int]:
    """Return confusion matrix components for binary classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing true negatives, false positives,
        false negatives, and true positives.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
