# Normalizing Features Using MinMaxScaler

## Purpose

This documentation captures the implementation of the normalization lesson built around MinMaxScaler. The goal is to explain how bounded feature scaling works, when to use it, and how to apply it safely in a production-style pipeline.

## Implementation Location

- Lesson notebook: [notebooks/normalizing_features_using_minmaxscaler.ipynb](../notebooks/normalizing_features_using_minmaxscaler.ipynb)

## What the Lesson Covers

The notebook explains:

- What normalization is and how it differs from standardization.
- The MinMaxScaler formula and the meaning of the transformed range.
- Which model families benefit from normalization.
- Why tree-based models usually do not need it.
- How to avoid leakage by fitting only on training data.
- How to combine MinMaxScaler with categorical encoding using ColumnTransformer.
- How outliers can compress the bulk of the data into a narrow range.

## Implementation Notes

- The notebook uses synthetic data so the lesson can be run independently.
- Numerical and categorical features are handled separately in the pipeline.
- MinMaxScaler is applied only to numerical columns.
- A deliberate outlier is included to demonstrate MinMaxScaler sensitivity.
- The preprocessing is wrapped in a pipeline so the fitted scaler can be reused at inference time.

## Validation

- The notebook is stored as valid JSON.
- The preprocessing sequence respects the train-test boundary.
- The lesson is documented separately from source code to keep implementation notes discoverable.

## Related Guidance

This lesson aligns with the project requirement to normalize numerical features intentionally, avoid fitting on test data, and document preprocessing choices clearly.
