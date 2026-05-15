from __future__ import annotations

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data, verify_split
from src.evaluate import evaluate_model
from src.feature_engineering import add_derived_features, build_full_pipeline
from src.persistence import save_artifacts
from sklearn.ensemble import RandomForestClassifier


def run_workflow(config: Config | None = None) -> dict[str, float]:
    """Run the end-to-end training workflow.
    
    Workflow sequence (prevents data leakage):
        1. Load raw data
        2. Clean data
        3. Add derived features
        4. SPLIT into train/test (BEFORE preprocessing)
        5. Verify split (no leakage)
        6. Build FULL pipeline (preprocessing + model)
        7. Fit full pipeline on training data only
        8. Evaluate on test data (never touched before)
        9. Save artifacts for production
    
    Returns:
        Dictionary of evaluation metrics on the test set.
    """

    settings = config or Config()

    if not settings.categorical_columns and not settings.numerical_columns:
        raise ValueError(
            "Set categorical_columns and/or numerical_columns in Config before running the workflow."
        )

    # Step 1-3: Load, clean, and derive features
    data = load_data(settings.data_path)
    data = clean_data(data, required_columns=[settings.target_column])
    data = add_derived_features(data)

    # Step 4: SPLIT BEFORE PREPROCESSING (critical for preventing leakage)
    X_train, X_test, y_train, y_test = split_data(
        data,
        target_column=settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    
    # Step 5: Verify split integrity
    verify_split(X_train, X_test, y_train, y_test)

    # Step 6: Define model
    model = RandomForestClassifier(
        random_state=settings.random_state,
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
    )

    # Step 7: Build and fit FULL pipeline
    full_pipeline = build_full_pipeline(
        categorical_columns=settings.categorical_columns,
        numerical_columns=settings.numerical_columns,
        model=model,
    )
    full_pipeline.fit(X_train, y_train)

    # Step 8: Evaluate on test data
    metrics = evaluate_model(full_pipeline, X_test, y_test)
    
    # Step 9: Save artifacts for production use
    save_artifacts(full_pipeline.named_steps["model"], full_pipeline.named_steps["preprocessing"], settings.model_path, settings.pipeline_path)

    return metrics


if __name__ == "__main__":
    workflow_metrics = run_workflow()
    print(workflow_metrics)
