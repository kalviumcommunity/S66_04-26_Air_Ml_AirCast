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

## How To Use

1. Install dependencies with `pip install -r requirements.txt`.
2. Put your CSV file in `data/raw/` and update `src/config.py` with the correct file path and column names.
3. Define the categorical and numerical columns in `Config`.
4. Run `python main.py` to train the model, evaluate it, and save artifacts.

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
