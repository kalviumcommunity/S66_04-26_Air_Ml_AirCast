from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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
    """Split a DataFrame into train and test sets with stratification.

    Implements the critical principle: SPLIT BEFORE ANY FITTING.
    
    Stratified splitting preserves class distribution across train and test sets,
    essential for classification tasks with class imbalance.
    
    Args:
        df: Input DataFrame containing the target column (must be pre-cleaned).
        target_column: Name of the column to predict.
        test_size: Fraction of data reserved for testing (default 0.2 = 20%).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
        - X_train: Features for training (80% by default)
        - X_test: Features for evaluation (20% by default)
        - y_train: Target labels for training
        - y_test: Target labels for evaluation (untouched until final evaluation)

    Raises:
        KeyError: If the target column is missing from the DataFrame.
        
    Example:
        >>> X_train, X_test, y_train, y_test = split_data(
        ...     df, 
        ...     target_column='AQI_Category',
        ...     test_size=0.2,
        ...     random_state=42
        ... )
        >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    Critical Rules:
        1. Call this function BEFORE preprocessing.fit_transform()
        2. Fit transformers only on X_train, apply to X_test
        3. Never use X_test statistics during X_train processing
        4. Keep test set completely untouched until final evaluation
    """

    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # Stratified splitting: preserves class distribution
    stratify = target if target.nunique(dropna=False) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    return X_train, X_test, y_train, y_test


def verify_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Verify that train and test splits are correctly separated.
    
    Checks for:
    - Correct shapes
    - No overlap between train and test indices
    - Class distribution balance (for classification)
    
    Args:
        X_train: Training feature set.
        X_test: Testing feature set.
        y_train: Training target labels.
        y_test: Testing target labels.
        
    Raises:
        AssertionError: If verification fails.
        
    Example:
        >>> verify_split(X_train, X_test, y_train, y_test)
        ✓ Split verification passed
          - Training set: (5604, 19) samples
          - Testing set: (1401, 19) samples
          - Total: 7005 samples
          - No overlap: 0 shared indices
          - Training target distribution:
            No    0.733
            Yes   0.267
            Name: target, dtype: float64
          - Testing target distribution:
            No    0.735
            Yes   0.265
            Name: target, dtype: float64
    """
    
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train shape mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test shape mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
    
    # Check no overlap
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} samples appear in both train and test"
    
    # Print verification results
    print("✓ Split verification passed")
    print(f"  - Training set: {X_train.shape} samples")
    print(f"  - Testing set: {X_test.shape} samples")
    print(f"  - Total: {X_train.shape[0] + X_test.shape[0]} samples")
    print(f"  - No overlap: {len(overlap)} shared indices")
    print(f"  - Training target distribution:")
    print(f"    {y_train.value_counts(normalize=True)}")
    print(f"  - Testing target distribution:")
    print(f"    {y_test.value_counts(normalize=True)}")


def check_class_imbalance(y: pd.Series) -> pd.DataFrame:
    """Return counts and percentages for each class in a target Series.
    
    Args:
        y: Target labels.
        
    Returns:
        A DataFrame showing absolute count and relative percentage for each class.
    """

    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    imbalance_df = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })
    
    return imbalance_df


def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to the training data.
    
    Critical Rule: Apply ONLY to training data, never to testing data.
    
    Args:
        X_train: Training features.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_resampled, y_resampled).
    """

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
