# Scaling Numerical Features Using StandardScaler

## Purpose

This documentation captures the implementation of the lesson on scaling numerical features with StandardScaler. The goal is to show why scaling matters, when it is needed, and how to apply it without leaking information from the test set.

## Implementation Location

- Lesson notebook: [notebooks/scaling_numerical_features_standardscaler.ipynb](../notebooks/scaling_numerical_features_standardscaler.ipynb)

## What the Lesson Covers

The notebook explains:

- Why feature scaling matters for optimization-based and distance-based models.
- Which model families usually require scaling and which are generally scale-invariant.
- How StandardScaler transforms values using mean and standard deviation.
- How to split before fitting to avoid data leakage.
- How to combine scaling with categorical encoding using ColumnTransformer.
- How to save the fitted preprocessing pipeline for inference.

## Implementation Notes

- The notebook uses synthetic data so the lesson is self-contained.
- Numerical and categorical features are separated explicitly before preprocessing.
- StandardScaler is applied only to the numerical feature group.
- A ColumnTransformer is used so scaling and encoding live inside one pipeline.
- The pipeline is fit only on training data to preserve evaluation integrity.

## Validation

- The notebook file is stored as valid JSON.
- The preprocessing steps follow the train-test boundary correctly.
- The documentation is separated from source code so the lesson is easy to review and reuse.

## Related Guidance

This lesson supports the project plan requirement to keep preprocessing disciplined and to avoid fitting transformations on test or inference data.
