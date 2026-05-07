from __future__ import annotations

from pathlib import Path

import joblib


def save_artifacts(model, pipeline, model_path: str | Path, pipeline_path: str | Path) -> None:
    """Save the fitted model and preprocessing pipeline to disk.

    Args:
        model: Fitted estimator to persist.
        pipeline: Fitted preprocessing pipeline to persist.
        model_path: Destination path for the model artifact.
        pipeline_path: Destination path for the pipeline artifact.
    """

    model_file = Path(model_path)
    pipeline_file = Path(pipeline_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_file)
    joblib.dump(pipeline, pipeline_file)
