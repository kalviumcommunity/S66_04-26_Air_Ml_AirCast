from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_artifacts(model_path: str | Path, pipeline_path: str | Path) -> tuple[Any, Any]:
    """Load a persisted model and preprocessing pipeline.

    Args:
        model_path: Path to the saved model artifact.
        pipeline_path: Path to the saved preprocessing pipeline artifact.

    Returns:
        A tuple of (model, pipeline).
    """

    model = joblib.load(Path(model_path))
    pipeline = joblib.load(Path(pipeline_path))
    return model, pipeline


def predict(new_data: pd.DataFrame, model, pipeline) -> pd.DataFrame:
    """Generate predictions for new, unseen data.

    Args:
        new_data: Raw input records.
        model: A fitted estimator.
        pipeline: A fitted preprocessing pipeline.

    Returns:
        A DataFrame containing the original data plus prediction columns.
    """

    processed_data = pipeline.transform(new_data)
    predictions = model.predict(processed_data)

    result = new_data.copy()
    result["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_data)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            result["prediction_probability"] = probabilities[:, 1]

    return result
