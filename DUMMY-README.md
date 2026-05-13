# DUMMY-README.md

## Learning Unit 5.26: Evaluating Classification Models Using Accuracy
- Date: 2026-05-13
- Timestamp: Initial draft

### What it covers
How to evaluate classification predictions using accuracy, confusion matrix context, and balanced accuracy.

### Features implemented
- Baseline `accuracy_score` reporting.
- Confusion matrix based error split review.
- Balanced accuracy comparison for imbalanced labels.

### Key learnings
- Accuracy alone is not enough on imbalanced datasets.
- Balanced metrics are required to validate class-wise performance.

### Output/result
Established an evaluation-first workflow that reduces metric misinterpretation in AirCast classification modules.

## Learning Unit 5.27: Evaluating Classification Models Using Precision and Recall
- Date: 2026-05-13
- Timestamp: Follow-up draft

### What it covers
How to evaluate classification models with precision and recall and decide metric priority by product risk.

### Features implemented
- Precision metric reporting in evaluation summaries.
- Recall metric reporting for positive-class detection quality.
- Trade-off documentation for threshold and metric balancing.

### Key learnings
- Precision and recall are essential on imbalanced datasets.
- Metric choice must align with false-positive vs false-negative cost.

### Output/result
Evaluation artifacts now include actionable precision-recall insights for model selection.

## Learning Unit 5.28: Evaluating Classification Models Using F1-Score
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How F1-score combines precision and recall into a balanced performance metric for imbalanced datasets.

### Features implemented
- F1-score reporting for classification evaluation.
- Macro/weighted F1 comparison guidance.
- Metric-priority framework for model selection.

### Key learnings
- F1 is often more reliable than raw accuracy on skewed labels.
- Balanced metric choices improve production model decisions.

### Output/result
AirCast model reports now include F1-driven ranking for robust classifier selection.

## Learning Unit 5.29: Creating and Interpreting a Confusion Matrix
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How to construct confusion matrices and interpret TP, FP, FN, and TN to diagnose model behavior.

### Features implemented
- Confusion matrix generation for classification outputs.
- Matrix interpretation template for error analysis.
- Error-type driven improvement notes for next iterations.

### Key learnings
- Matrix analysis reveals what aggregate metrics hide.
- Error categories guide targeted model corrections.

### Output/result
AirCast evaluation now includes transparent error diagnostics for each classifier run.

## Learning Unit 5.30: Training a K-Nearest Neighbours (KNN) Model
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How KNN predicts using nearest data points and why scaling and distance metrics are critical.

### Features implemented
- KNN training pipeline with configurable neighbors.
- Feature scaling integration before KNN modeling.
- Validation-based K selection notes.

### Key learnings
- KNN quality depends heavily on feature scaling.
- Hyperparameter `k` controls bias-variance behavior in local predictions.

### Output/result
Added a tuned KNN modeling stage to strengthen AirCast baseline model comparisons.

## Learning Unit 5.31: Understanding Bias and Variance Through Model Behavior
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How to identify underfitting and overfitting using train/test behavior and learning curves.

### Features implemented
- Bias-variance diagnosis framework.
- Learning-curve based model behavior checks.
- Generalization-focused correction strategy notes.

### Key learnings
- Strong training scores do not guarantee real-world performance.
- Controlled complexity improves stability across unseen data.

### Output/result
AirCast model review now includes systematic generalization diagnostics before promotion.

## Learning Unit 5.32: Training a Decision Tree Model
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How Decision Trees split data recursively and how to constrain complexity for robust performance.

### Features implemented
- Decision Tree training workflow.
- Overfitting controls via depth and leaf parameters.
- Interpretable rule-review process.

### Key learnings
- Decision Trees are highly interpretable but can overfit if unconstrained.
- Controlled depth improves test performance stability.

### Output/result
AirCast includes a constrained Decision Tree benchmark for transparent and reliable predictions.

## Learning Unit 5.33: Interpreting Feature Importance from Tree-Based Models
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How to extract and compare feature importance from tree-based models using both impurity and permutation methods.

### Features implemented
- Feature importance extraction workflow.
- Cross-method comparison guidance.
- Interpretation guardrails for trustworthy insights.

### Key learnings
- Importance scores are method-dependent and require context.
- Permutation importance helps validate true predictive contribution.

### Output/result
AirCast reporting now highlights reliable feature drivers for explainable model behavior.

## Learning Unit 5.34: Improving Model Performance with Hyperparameter Tuning using GridSearchCV
- Date: 2026-05-13
- Timestamp: Continuation draft

### What it covers
How to tune model hyperparameters using `GridSearchCV`, cross-validation, and metric-guided model selection.

### Features implemented
- Reusable GridSearchCV tuning helper in `src/model_selection.py`.
- KNN and Decision Tree training helpers for consistent experimentation.
- Extended evaluation with balanced accuracy and confusion details.

### Key learnings
- Cross-validated tuning improves reliability over manual guessing.
- Unified utilities reduce leakage and inconsistency in experiments.

### Output/result
AirCast now supports repeatable, evidence-based model optimization and selection.
