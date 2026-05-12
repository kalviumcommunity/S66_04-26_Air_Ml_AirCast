# Evaluating Regression Models Using MAE

## Purpose

This documentation captures the implementation of the MAE evaluation lesson. The goal is to show how to measure regression error in a way that is interpretable, honest, and easy to compare against a baseline.

## Implementation Location

- Lesson notebook: [notebooks/evaluating_regression_models_using_mae.ipynb](../notebooks/evaluating_regression_models_using_mae.ipynb)

## What the Lesson Covers

The notebook explains:

- What Mean Absolute Error measures.
- Why MAE is intuitive for business communication.
- How MAE differs from MSE and RMSE.
- How to compare MAE against a baseline regressor.
- How to use cross-validation with negative MAE scoring in scikit-learn.
- Why residual inspection and metric context are still necessary.

## Implementation Notes

- The notebook uses synthetic regression data so it can run independently.
- A DummyRegressor baseline is fit on the training split before evaluating the model.
- The lesson uses LinearRegression as the comparison model.
- Cross-validation is included to show how to check stability across folds.
- MAE is reported alongside RMSE and R² for a more complete evaluation.

## Validation

- The notebook is stored as valid JSON.
- The examples split before fitting baseline or model.
- The documentation is separate from source code and indexed with the other lesson notes.

## Related Guidance

This lesson reinforces the project discipline of evaluating regression models with an interpretable metric and a meaningful baseline.
