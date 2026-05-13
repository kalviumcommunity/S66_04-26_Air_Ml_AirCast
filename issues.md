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
