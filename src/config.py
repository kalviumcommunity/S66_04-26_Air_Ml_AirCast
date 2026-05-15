from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Centralized configuration for the ML workflow."""

    data_path: Path = Path("data/raw/air_pollution_data.csv")
    processed_data_path: Path = Path("data/processed/processed.csv")
    model_path: Path = Path("models/model.joblib")
    pipeline_path: Path = Path("models/preprocessing.joblib")
    target_column: str = "risk_level"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = 10
    class_weight: str | dict[str, float] | None = "balanced"
    categorical_columns: tuple[str, ...] = ("city", "season", "day_of_week")
    numerical_columns: tuple[str, ...] = ("temperature", "humidity", "wind_speed", "pm25", "pm10")
