# Training a Linear Regression Model

## Purpose

This documentation captures the implementation of the Linear Regression lesson. The goal is to show how to move from a baseline regressor to a real supervised model, evaluate it fairly, and interpret the learned coefficients.

## Implementation Location

- Lesson notebook: [notebooks/training_a_linear_regression_model.ipynb](../notebooks/training_a_linear_regression_model.ipynb)

## What the Lesson Covers

The notebook explains:

- What Linear Regression is and when it is useful.
- Why the model should be compared against a regression baseline.
- How the training objective relates to Mean Squared Error.
- How to evaluate with RMSE, MAE, and R².
- How to interpret coefficients after scaling.
- How to keep preprocessing inside a pipeline to avoid leakage.

## Implementation Notes

- The notebook uses synthetic regression data so it can run independently.
- A DummyRegressor baseline is fit on the training split before the model.
- LinearRegression is wrapped in a pipeline with StandardScaler.
- Coefficients are displayed after training so the lesson can explain interpretation.
- Cross-validation is included to show how to check stability beyond a single split.

## Validation

- The notebook is stored as valid JSON.
- The examples split data before fitting any model.
- The preprocessing and modeling steps are encapsulated in a pipeline.

## Related Guidance

This lesson supports the project discipline of comparing every regression model against a meaningful baseline before claiming improvement.
