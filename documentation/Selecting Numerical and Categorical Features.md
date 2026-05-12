# Feature Type Selection Lesson Documentation

## Purpose

This documentation captures the implementation of the feature type selection lesson added to the project as a separate notebook. The goal is to teach how to distinguish numerical and categorical features before modeling and how that decision drives preprocessing.

## Implementation Location

- Lesson notebook: [notebooks/feature_type_selection_lesson.ipynb](../notebooks/feature_type_selection_lesson.ipynb)

## What the Lesson Covers

The notebook walks through a practical tabular workflow and includes:

- A definition of numerical and categorical features.
- The effect of feature type choice on encoding and scaling.
- An explicit target-first feature split.
- A synthetic telco-style example with numerical, categorical, and excluded columns.
- A `ColumnTransformer` pipeline that applies the right preprocessing to each group.
- A short checklist for validating feature assignments before modeling.

## Implementation Notes

- The notebook uses synthetic data so the lesson can run independently of project datasets.
- `CustomerID` is excluded to show that identifier columns should not be treated as modeling features.
- `SeniorCitizen` is kept as a binary example to reinforce that storage type and semantic type are not always the same.
- The preprocessing pipeline separates numeric scaling from categorical encoding using `ColumnTransformer`.
- The model is wrapped in a single pipeline so preprocessing is fitted only on training data.

## Validation

- The notebook was created as valid JSON.
- Feature groups are explicitly defined and checked for overlap.
- The lesson is stored separately from source code and can be referenced from project documentation.

## Related Guidance

This lesson supports the project plan requirement to validate feature groups and avoid columns that would not be available at prediction time.
