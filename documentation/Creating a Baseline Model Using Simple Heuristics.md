# Creating a Baseline Model Using Simple Heuristics

## Purpose

This documentation captures the implementation of the baseline modeling lesson. The goal is to show how to establish a simple reference point for both classification and regression tasks before building more complex models.

## Implementation Location

- Lesson notebook: [notebooks/creating_a_baseline_model_using_simple_heuristics.ipynb](../notebooks/creating_a_baseline_model_using_simple_heuristics.ipynb)

## What the Lesson Covers

The notebook explains:

- Why baseline models are necessary in machine learning workflows.
- How majority-class, stratified, uniform, mean, and median baselines work.
- How to compare a baseline against a real model using the same metric.
- Why baselines help detect leakage, imbalance, and weak feature sets.
- How to interpret a baseline that performs surprisingly well.
- How to keep baseline evaluation inside the train-test boundary.

## Implementation Notes

- The notebook uses synthetic data so the lesson can run independently.
- A classification example compares a DummyClassifier with logistic regression.
- A regression example compares a DummyRegressor with linear regression.
- The baseline is fit only on the training split to avoid leakage.
- The lesson emphasizes side-by-side metric comparison for honest evaluation.

## Validation

- The notebook is stored as valid JSON.
- The examples respect the train-test split before fitting any model.
- The documentation is separate from source code and can be linked from the docs index.

## Related Guidance

This lesson reinforces the project discipline of always measuring a model against a simple reference point before claiming success.
