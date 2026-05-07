from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Centralized configuration for the ML workflow.
    
    Attributes:
        data_path: Path to raw data CSV file.
        processed_data_path: Path for saving processed data.
        model_path: Path for saving trained model artifact.
        pipeline_path: Path for saving fitted preprocessing pipeline.
        target_column: Name of the column to predict.
        test_size: Fraction of data reserved for testing (0.2 = 20%).
        random_state: Random seed (42 ensures reproducibility).
        n_estimators: Number of decision trees in the random forest.
        max_depth: Maximum tree depth (None = unlimited).
        categorical_columns: Tuple of categorical feature names.
        numerical_columns: Tuple of numerical feature names.
    
    Data Splitting Strategy:
        - Train/test split uses 80/20 by default.
        - Stratified splitting preserves class distribution.
        - Split occurs BEFORE any preprocessing fitting.
        - Test set remains untouched until final evaluation.
    """

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
