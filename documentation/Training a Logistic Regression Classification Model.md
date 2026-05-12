# Training a Logistic Regression Classification Model

## Purpose

This documentation captures the implementation of the Logistic Regression classification lesson. The goal is to show how to train a probabilistic classifier, evaluate it correctly, and interpret coefficients as odds ratios.

## Implementation Location

- Lesson notebook: [notebooks/training_a_logistic_regression_classification_model.ipynb](../notebooks/training_a_logistic_regression_classification_model.ipynb)

## What the Lesson Covers

The notebook explains:

- What Logistic Regression is and why it is used for classification.
- Why a linear regression model is not appropriate for binary classification.
- How the sigmoid function maps scores to probabilities.
- How to compare Logistic Regression against a majority-class baseline.
- How to evaluate using accuracy, classification report, and ROC-AUC.
- How to interpret coefficients and odds ratios.
- How to use stratified cross-validation for stability checks.

## Implementation Notes

- The notebook uses synthetic binary classification data so it can run independently.
- A DummyClassifier baseline is fit on the training split before evaluation.
- LogisticRegression is wrapped in a StandardScaler pipeline to keep preprocessing leakage-free.
- The lesson includes both probability outputs and class-label outputs.
- Cross-validation is included to show consistency across folds.

## Validation

- The notebook is stored as valid JSON.
- The examples split data before fitting baseline or model.
- The preprocessing and modeling steps are encapsulated in a pipeline.

## Related Guidance

This lesson reinforces the project discipline of comparing every classification model against a meaningful baseline before claiming success.
