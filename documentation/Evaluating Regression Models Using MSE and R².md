# Evaluating Regression Models Using MSE and R²

## Purpose

This documentation captures the implementation of the regression evaluation lesson focused on Mean Squared Error and R². The goal is to show how to measure absolute error and explained variance together, and how to compare both metrics against a baseline.

## Implementation Location

- Lesson notebook: [notebooks/evaluating_regression_models_using_mse_and_r2.ipynb](../notebooks/evaluating_regression_models_using_mse_and_r2.ipynb)

## What the Lesson Covers

The notebook explains:

- What MSE measures and why squaring errors changes behavior.
- What R² means and how it relates to the mean baseline.
- How to compute both metrics in scikit-learn.
- Why MSE and R² must be interpreted together.
- How to compare a regression model against a baseline.
- How to use cross-validation to assess stability.

## Implementation Notes

- The notebook uses synthetic regression data so it can run independently.
- A DummyRegressor baseline is fit on the training split before evaluation.
- LinearRegression is used as the comparison model.
- Cross-validation is included for both R² and RMSE stability checks.
- The lesson emphasizes metric context and baseline comparison rather than raw scores alone.

## Validation

- The notebook is stored as valid JSON.
- The examples split data before fitting baseline or model.
- The documentation is separate from source code and indexed with the other lesson notes.

## Related Guidance

This lesson reinforces the project discipline of evaluating regression models with both an absolute error metric and an explanatory-power metric.
