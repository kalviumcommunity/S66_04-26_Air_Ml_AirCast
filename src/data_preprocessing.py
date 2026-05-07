from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load tabular data from a CSV file.

    Args:
        filepath: Path to the source CSV file.

    Returns:
        A DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    source_path = Path(filepath)
    if not source_path.exists():
        raise FileNotFoundError(f"Data file not found: {source_path}")

    return pd.read_csv(source_path)


def clean_data(df: pd.DataFrame, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Perform lightweight, reusable cleaning on a DataFrame.

    Args:
        df: Raw input data.
        required_columns: Optional list of columns that must exist.

    Returns:
        A cleaned copy of the input DataFrame.

    Raises:
        KeyError: If any required columns are missing.
    """

    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if required_columns is not None:
        missing_columns = [column for column in required_columns if column not in cleaned.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

    return cleaned


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train and test sets.

    Args:
        df: Input DataFrame containing the target column.
        target_column: Name of the target column.
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test in that order.

    Raises:
        KeyError: If the target column is missing.
    """

    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    features = df.drop(columns=[target_column])
    target = df[target_column]
    stratify = target if target.nunique(dropna=False) > 1 else None

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
