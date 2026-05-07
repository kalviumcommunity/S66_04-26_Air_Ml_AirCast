from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Centralized configuration for the ML workflow."""

    data_path: Path = Path("data/raw/telco_churn.csv")
    processed_data_path: Path = Path("data/processed/processed.csv")
    model_path: Path = Path("models/model.joblib")
    pipeline_path: Path = Path("models/preprocessing.joblib")
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    categorical_columns: tuple[str, ...] = ()
    numerical_columns: tuple[str, ...] = ()
