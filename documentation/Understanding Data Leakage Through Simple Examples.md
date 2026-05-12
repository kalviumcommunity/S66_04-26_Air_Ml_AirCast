# Data Leakage Lesson Documentation

## Purpose

This documentation captures the implementation of the data leakage lesson added to the project as a separate Jupyter notebook. The goal is to teach the core leakage patterns in machine learning and show how to prevent them in a practical tabular workflow.

## Implementation Location

- Lesson notebook: [notebooks/data_leakage_lesson.ipynb](../notebooks/data_leakage_lesson.ipynb)

## What the Lesson Covers

The notebook is structured as a short teaching module with a mix of explanation and runnable examples. It includes:

- A definition of data leakage and why it breaks evaluation.
- A target leakage example using a synthetic churn dataset.
- A safe preprocessing example that splits first and fits only on training data.
- A prediction-moment checklist for reviewing features.
- A short practical summary and reference resources.

## Implementation Notes

- The notebook uses synthetic data so the lesson is self-contained and easy to run.
- The leaking feature is intentionally included in one example to show how performance can look unrealistically good.
- The safe example demonstrates the correct pattern: split first, then fit preprocessing only on the training data.
- The notebook avoids project-specific assumptions so it can serve as a reusable teaching asset.

## Validation

- The notebook file was validated as well-formed JSON after creation.
- The lesson is stored separately from the source code to keep documentation and implementation details organized.

## Related Project Guidance

This lesson aligns with the project plan guidance to check for leakage risks, especially columns that would not be available at prediction time.
