# video_script.md

## Learning Unit 5.26: Evaluating Classification Models Using Accuracy
- Date: 2026-05-13
- Timestamp: Initial draft

### Introduction
In this learning unit, we build the first evaluation layer for the AirCast classifier by measuring how often predictions are correct overall.

### What was built
- Added an evaluation plan to compute `accuracy_score`.
- Added confusion-matrix-driven interpretation to inspect error distribution.
- Added balanced accuracy checks for class imbalance.

### Step-by-step explanation
1. Load true labels and model predictions from the baseline classification pipeline.
2. Compute standard accuracy to get the top-level correctness rate.
3. Build a confusion matrix to understand where errors occur.
4. Compute balanced accuracy so each class contributes equally.
5. Compare accuracy vs balanced accuracy and document mismatch risk.

### Challenges faced
- Accuracy can look high even when minority classes are poorly predicted.

### Solutions applied
- Added balanced accuracy and confusion matrix review as mandatory checks before accepting a model.

### Final outcome
The project now has a reliable baseline evaluation stage that prevents false confidence from raw accuracy alone.

## Learning Unit 5.27: Evaluating Classification Models Using Precision and Recall
- Date: 2026-05-13
- Timestamp: Follow-up draft

### Introduction
This unit extends AirCast evaluation to prioritize error types, especially when false alarms and missed alerts have different business impact.

### What was built
- Added precision and recall metrics to the evaluation flow.
- Added threshold-aware interpretation for false positives and false negatives.
- Added imbalance-safe metric reporting beside accuracy.

### Step-by-step explanation
1. Use model predictions and labels from the classification pipeline.
2. Compute precision to measure prediction correctness among positive predictions.
3. Compute recall to measure positive-class capture rate.
4. Compare both metrics under business risk assumptions.
5. Record trade-offs and choose a preferred operating point.

### Challenges faced
- Improving recall can reduce precision, and vice versa.

### Solutions applied
- Added metric trade-off discussion and scenario-based threshold selection.

### Final outcome
AirCast can now evaluate classification quality using error-aware metrics rather than relying on accuracy alone.

## Learning Unit 5.28: Evaluating Classification Models Using F1-Score
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit adds F1-score to balance precision and recall when AirCast decisions must avoid both missed events and noisy alerts.

### What was built
- Added F1-score to the evaluation dashboard.
- Added macro and weighted F1 interpretation for imbalance.
- Added metric selection guidance for threshold tuning.

### Step-by-step explanation
1. Compute precision and recall for the target class.
2. Calculate F1 as the harmonic mean of precision and recall.
3. Compare F1 with accuracy and balanced accuracy.
4. Use macro F1 for class-balanced comparison.
5. Record model ranking based on chosen F1 objective.

### Challenges faced
- Models with strong accuracy did not always retain strong F1.

### Solutions applied
- Prioritized F1 when minority class detection quality matters more than majority accuracy.

### Final outcome
The evaluation process now includes a robust single-score metric for imbalanced classification performance.

## Learning Unit 5.29: Creating and Interpreting a Confusion Matrix
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit adds error-visibility tooling so AirCast teams can inspect exactly where the classifier succeeds and fails.

### What was built
- Added confusion matrix construction for classification outputs.
- Added TN/FP/FN/TP interpretation notes.
- Added class-level error diagnosis workflow.

### Step-by-step explanation
1. Generate predicted labels for the test split.
2. Build the confusion matrix from true and predicted labels.
3. Map each cell to TN, FP, FN, and TP.
4. Calculate class-specific error indicators from matrix values.
5. Document targeted improvements based on dominant error type.

### Challenges faced
- Team initially interpreted row/column axes inconsistently.

### Solutions applied
- Standardized label ordering and added explicit matrix interpretation guidelines.

### Final outcome
The project now has a clear diagnostic layer for classification mistakes, enabling focused model improvements.
