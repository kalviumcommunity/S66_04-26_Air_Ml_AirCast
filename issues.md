# issues.md

## Learning Unit 5.26: Evaluating Classification Models Using Accuracy
- Date: 2026-05-13
- Timestamp: Initial draft

### Issue description
Accuracy looked acceptable while minority-class predictions were weak.

### When it occurred
During baseline model review after first classification run.

### Root cause
Class imbalance allowed majority-class correctness to dominate the overall score.

### Solution
Added balanced accuracy and confusion matrix inspection to the required evaluation checklist.

## Learning Unit 5.27: Evaluating Classification Models Using Precision and Recall
- Date: 2026-05-13
- Timestamp: Follow-up draft

### Issue description
Model ranking changed when precision and recall were introduced.

### When it occurred
During model comparison on imbalanced classification labels.

### Root cause
Accuracy masked false-negative and false-positive behavior.

### Solution
Added precision/recall reporting and business-context threshold tuning notes.

## Learning Unit 5.28: Evaluating Classification Models Using F1-Score
- Date: 2026-05-13
- Timestamp: Continuation draft

### Issue description
Precision improved in one model while recall dropped sharply.

### When it occurred
During metric comparison between baseline and tuned classifiers.

### Root cause
Single-metric optimization biased one side of the precision-recall trade-off.

### Solution
Adopted F1-score as the primary comparison metric for balanced classification behavior.

## Learning Unit 5.29: Creating and Interpreting a Confusion Matrix
- Date: 2026-05-13
- Timestamp: Continuation draft

### Issue description
False negatives were overlooked during early model review.

### When it occurred
While validating classifier performance from top-level metrics only.

### Root cause
Lack of explicit TP/FP/FN/TN breakdown masked critical error categories.

### Solution
Added confusion matrix analysis as a mandatory evaluation artifact in model reports.

## Learning Unit 5.30: Training a K-Nearest Neighbours (KNN) Model
- Date: 2026-05-13
- Timestamp: Continuation draft

### Issue description
KNN produced unstable results across runs with different feature scales.

### When it occurred
During first KNN benchmark experiments in the model training phase.

### Root cause
Distance-based learning is sensitive to feature magnitude differences.

### Solution
Introduced mandatory scaling and documented metric-aware K selection strategy.

## Learning Unit 5.31: Understanding Bias and Variance Through Model Behavior
- Date: 2026-05-13
- Timestamp: Continuation draft

### Issue description
Model performance dropped sharply from training to test datasets.

### When it occurred
During candidate model comparison after adding more complex learners.

### Root cause
High variance due to model complexity relative to available training signal.

### Solution
Introduced learning-curve diagnostics and regularization/depth constraints in model selection logic.
