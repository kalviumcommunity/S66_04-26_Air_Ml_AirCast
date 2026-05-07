from __future__ import annotations

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import add_derived_features, build_preprocessing_pipeline
from src.persistence import save_artifacts
from src.train import train_model


def run_workflow(config: Config | None = None) -> dict[str, float]:
    """Run the end-to-end training workflow."""

    settings = config or Config()

    if not settings.categorical_columns and not settings.numerical_columns:
        raise ValueError(
            "Set categorical_columns and/or numerical_columns in Config before running the workflow."
        )

    data = load_data(settings.data_path)
    data = clean_data(data, required_columns=[settings.target_column])
    data = add_derived_features(data)

    X_train, X_test, y_train, y_test = split_data(
        data,
        target_column=settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_columns=settings.categorical_columns,
        numerical_columns=settings.numerical_columns,
    )
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    model = train_model(
        X_train_processed,
        y_train,
        random_state=settings.random_state,
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
    )
    metrics = evaluate_model(model, X_test_processed, y_test)
    save_artifacts(model, preprocessing_pipeline, settings.model_path, settings.pipeline_path)

    return metrics


if __name__ == "__main__":
    workflow_metrics = run_workflow()
    print(workflow_metrics)
