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

## Learning Unit 5.30: Training a K-Nearest Neighbours (KNN) Model
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit introduces a distance-based model in AirCast to benchmark non-linear neighborhood behavior for both classification and regression tracks.

### What was built
- Added KNN training workflow with configurable `k` and distance metric.
- Added feature scaling requirement before KNN fitting.
- Added validation routine for selecting best neighbor count.

### Step-by-step explanation
1. Prepare train/test data and scale numeric features.
2. Initialize KNN model with candidate hyperparameters.
3. Train KNN on scaled training data.
4. Evaluate performance on validation/test split.
5. Compare KNN against previous baseline models.

### Challenges faced
- Unscaled features biased distance calculations.

### Solutions applied
- Enforced standardization before KNN and documented this as non-optional.

### Final outcome
AirCast now includes a properly trained KNN baseline with reliable evaluation and tuning structure.

## Learning Unit 5.31: Understanding Bias and Variance Through Model Behavior
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit strengthens model diagnostics by identifying underfitting and overfitting patterns before deployment decisions.

### What was built
- Added bias-variance diagnosis checklist using train vs test metrics.
- Added learning-curve interpretation notes.
- Added practical mitigation strategy mapping.

### Step-by-step explanation
1. Compare training and test performance across candidate models.
2. Detect high-bias patterns from uniformly low scores.
3. Detect high-variance patterns from wide train-test gaps.
4. Use learning curves to verify whether more data or regularization helps.
5. Apply targeted interventions and re-evaluate.

### Challenges faced
- Some models looked strong on train data but failed to generalize.

### Solutions applied
- Added mandatory generalization checks and learning-curve evidence before model approval.

### Final outcome
AirCast now has a structured approach to controlling overfitting and improving model generalization.

## Learning Unit 5.32: Training a Decision Tree Model
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit adds interpretable tree-based modeling so AirCast predictions can be explained through explicit split rules.

### What was built
- Added Decision Tree training plan for classification/regression use.
- Added depth and leaf constraints to reduce overfitting.
- Added evaluation and interpretability review steps.

### Step-by-step explanation
1. Prepare features and labels from processed data.
2. Train a baseline Decision Tree.
3. Apply constraints (`max_depth`, `min_samples_leaf`, `min_samples_split`).
4. Evaluate train/test metrics for generalization.
5. Inspect tree behavior and finalize a constrained model.

### Challenges faced
- Unconstrained trees overfit quickly on training data.

### Solutions applied
- Added regularized tree settings and comparison against baseline complexity.

### Final outcome
AirCast now has an interpretable Decision Tree stage with overfitting control built into the training workflow.

## Learning Unit 5.33: Interpreting Feature Importance from Tree-Based Models
- Date: 2026-05-13
- Timestamp: Continuation draft

### Introduction
This unit adds explainability for AirCast by identifying which input features most influence tree-based model predictions.

### What was built
- Added feature importance extraction for Decision Tree, Random Forest, and Gradient Boosting models.
- Added impurity-based vs permutation importance comparison guidance.
- Added interpretation rules to avoid misleading conclusions.

### Step-by-step explanation
1. Train tree-based models with stable validation setup.
2. Extract built-in feature importance values.
3. Compute permutation importance on held-out data.
4. Compare rankings across both methods.
5. Document robust feature drivers and caution notes.

### Challenges faced
- Impurity-based importance overvalued high-cardinality features.

### Solutions applied
- Added permutation importance checks and cross-model consistency review.

### Final outcome
AirCast now includes explainable feature-impact reporting to support model transparency and better decisions.
