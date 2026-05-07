# S66_04-26_Air_Ml_AirCast

Modular starter scaffold for an ML workflow focused on reusable functions and clean imports.

## Project Structure

```text
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── persistence.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## What Each Module Does

- `src/config.py` keeps file paths, target column, and model hyperparameters in one place.
- `src/data_preprocessing.py` handles loading, cleaning, and train/test splitting.
- `src/feature_engineering.py` builds reusable derived features and a preprocessing pipeline.
- `src/train.py` fits the model and returns the trained artifact.
- `src/evaluate.py` returns metrics as a dictionary instead of printing them.
- `src/persistence.py` saves fitted artifacts to disk.
- `src/predict.py` loads artifacts and generates predictions on new data.
- `main.py` orchestrates the full workflow.

## Module Structure Plan

The project follows a clean `Data -> Preprocessing -> Features -> Model -> Evaluation -> Prediction` flow so each stage stays isolated and reusable.

- `main.py` is the entry point. It imports the functions it needs and runs them in sequence.
- `src/config.py` holds shared constants such as file paths, random seeds, and hyperparameters.
- `src/data_preprocessing.py` defines reusable functions for loading, cleaning, and splitting data.
- `src/feature_engineering.py` defines reusable feature transformations and the preprocessing pipeline.
- `src/train.py` trains the model only. It should not evaluate, print, or predict.
- `src/evaluate.py` computes metrics on a fitted model and returns them as structured data.
- `src/predict.py` loads saved artifacts and generates predictions on new data.
- `src/persistence.py` saves and loads artifacts so prediction does not depend on retraining.

Import rules for the project:

- Use absolute imports from `src`, such as `from src.train import train_model`.
- Keep imports explicit and avoid wildcard imports.
- Do not let `predict.py` import training logic.
- Store shared values in `config.py` instead of duplicating them across files.
- Keep preprocessing logic in one place so training and prediction use the same transformations.

This structure makes the repository easier to test, easier to review, and safer to extend when the model or dataset changes.

The full sprint assignment plan is documented in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Lesson Plan Summary

This repository is organized around the lesson idea that structure matters more than algorithm choice once an ML project grows beyond a notebook.

### Why Structure Matters

- Notebooks are fine for exploration, but they become fragile when execution order matters.
- Functions make preprocessing, training, evaluation, and prediction reusable and testable.
- Modules make it possible to split responsibilities so one file does not control the whole workflow.
- Clean imports prevent hidden dependencies and make the codebase easier to hand off to another developer.

### Script-to-Module Shift

The project is designed so each Python file can be imported safely without triggering training automatically.

- `main.py` contains the entry point guarded by `if __name__ == "__main__":`.
- `train.py` defines training behavior but does not execute it on import.
- `predict.py` only loads artifacts and predicts.
- `evaluate.py` only computes metrics.
- `feature_engineering.py` only transforms features.

### Configuration Plan

Shared values are centralized in `src/config.py` so paths, seeds, and hyperparameters are not duplicated.

- File paths stay in one place.
- Random state stays in one place.
- Column names stay in one place.
- Model hyperparameters stay in one place.

### Training and Prediction Boundary

Training and prediction are separated on purpose.

- Training uses `fit()` or `fit_transform()` on the training set only.
- Prediction uses `transform()` with already-fitted preprocessing artifacts.
- `predict.py` never retrains the model.
- Saved artifacts in `models/` make inference reproducible.

### Testing and Maintainability Plan

- Each function should return data instead of printing it.
- Each function should have a docstring and type hints.
- Each module should have one responsibility.
- Each reusable step should be isolated enough to unit test later.

### Recommended Development Flow

1. Load raw data in `data_preprocessing.py`.
2. Clean and validate data in the same module.
3. Build features in `feature_engineering.py`.
4. Train a model in `train.py`.
5. Evaluate the fitted model in `evaluate.py`.
6. Save and load artifacts in `persistence.py`.
7. Generate predictions in `predict.py`.
8. Orchestrate the full flow from `main.py`.

## How To Use

1. Create a virtual environment with `python -m venv venv`.
2. Activate it with `venv\\Scripts\\activate` on Windows or `source venv/bin/activate` on macOS/Linux.
3. Install dependencies with `pip install -r requirements.txt`.
4. Put your CSV file in `data/raw/` and update `src/config.py` with the correct file path and column names.
5. Define the categorical and numerical columns in `Config`.
6. Run `python main.py` to train the model, evaluate it, and save artifacts.
7. When the environment is ready, freeze exact versions with `pip freeze > requirements.txt`.

## Environment Notes

- Keep `venv/` out of version control.
- Avoid mixing global Python packages with project-specific packages.
- Recreate the same environment later with `pip install -r requirements.txt`.

## Notes

- The scaffold is intentionally generic so it can be adapted to any tabular ML classification task.
- `Config.categorical_columns` and `Config.numerical_columns` must be set before the workflow can run.

## Project Plan

### Problem Statement

Indian cities produce vast amounts of pollution data, but citizens rarely understand trends or forecasted risk levels. Clear visualisations can turn raw air-quality readings into actionable guidance so people can make health-conscious decisions about outdoor activity, travel timing, and protective measures.

### Solution Overview

This project will build an ML-supported pollution insights workflow that cleans city air-quality data, engineers time-based and location-based features, trains a predictive model for near-term risk classification or AQI forecasting, and presents the output through clear visual summaries. The goal is not only to predict values, but to make the prediction easy to interpret and useful for everyday decision-making.

### Dataset Definition

- Dataset source: City air-quality or pollution dataset from a public source such as Kaggle or an open government portal.
- Target variable: Near-term AQI category or forecasted pollution risk level.
- Task type: Classification or regression, depending on the final dataset.
- Key inputs: Pollutant readings, timestamp, city/location, weather signals, and related environmental factors.
- Known risks: Missing sensor readings, uneven city coverage, and potential seasonality effects.

### Scope

In scope:

- Data cleaning and validation
- Feature engineering for time, city, and pollutant trends
- Baseline model and one stronger model
- Evaluation with appropriate metrics
- Reusable prediction pipeline
- Clear charts and dashboard-ready outputs for interpretation

Out of scope:

- Real-time deployment
- Mobile app development
- Deep learning experiments
- Cloud infrastructure setup

### MVP

The MVP will include a reproducible pipeline that loads pollution data, preprocesses it, trains a model, evaluates it on a held-out test set, saves the model artifacts, and generates readable visual outputs that explain risk trends.

### Timeline

- Week 1: Acquire dataset, inspect missing values, define target variable, and finalize scope.
- Week 2: Build preprocessing and feature engineering functions, then create baseline visuals.
- Week 3: Train models, tune parameters, and compare evaluation results.
- Week 4: Finalize the workflow, save artifacts, polish the README, and prepare the demo.

### Success Metrics

- The pipeline runs end to end without manual notebook steps.
- The model produces usable pollution-risk predictions or forecasts.
- Visualisations make trends and risk levels easy to interpret.
- The final code is modular, reusable, and reproducible from a fresh environment.
