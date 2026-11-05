# TrustLens — Model Experiments

## Objective

Compare initial multi-label toxicity classification models using the same training and validation split.

## Dataset and split

* Dataset: Jigsaw Toxic Comment Classification
* Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
* Training examples: 127,656
* Validation examples: 31,915
* Split: MultilabelStratifiedShuffleSplit
* Random state: 42
* Features: Word-level TF-IDF, unigrams and bigrams
* Maximum vocabulary size: 300,000

## Model comparison

| Model                              | Threshold strategy | Micro F1 | Macro F1 |
| ---------------------------------- | ------------------ | -------: | -------: |
| Unweighted Logistic Regression     | Default 0.5        |    0.649 |    0.452 |
| Unweighted Logistic Regression     | Tuned per-label    |    0.736 |    0.608 |
| Class-weighted Logistic Regression | Tuned per-label    |    0.742 |    0.621 |
| Multinomial Naive Bayes            | Default 0.5        |    0.648 |    0.415 |
| Multinomial Naive Bayes            | Tuned per-label    |    0.674 |    0.529 |

## Naive Bayes tuned results

| Label         | Threshold | Precision | Recall |    F1 |
| ------------- | --------: | --------: | -----: | ----: |
| Toxic         |     0.332 |     0.774 |  0.684 | 0.726 |
| Severe toxic  |     0.090 |     0.477 |  0.511 | 0.493 |
| Obscene       |     0.311 |     0.755 |  0.667 | 0.708 |
| Threat        |     0.015 |     0.400 |  0.208 | 0.274 |
| Insult        |     0.215 |     0.641 |  0.679 | 0.659 |
| Identity hate |     0.067 |     0.354 |  0.285 | 0.316 |

## Interpretation

Threshold tuning improves Naive Bayes, particularly Macro F1. However, it remains behind both Logistic Regression experiments.

Class-weighted Logistic Regression is the current leading candidate, with tuned Micro F1 of 0.742 and Macro F1 of 0.621.

Threat and identity-hate detection remain difficult because of label imbalance and limited positive examples. These results do not justify fully automated punitive moderation.

## Evaluation caveat

Thresholds were selected and evaluated on the same validation set. The tuned results may be optimistic. An independent test evaluation is required before reporting final performance.

## Current decision

Retain:

* Class-weighted Logistic Regression as the leading candidate
* Unweighted Logistic Regression as a baseline
* Multinomial Naive Bayes as a comparison benchmark

Next:

1. Compare rule-based moderation with ML predictions.
2. Inspect false positives and false negatives.
3. Develop a hybrid moderation decision layer.
4. Evaluate the final selected pipeline on independent test data.


## Linear SVM Experiment

### Configuration

A One-vs-Rest Linear SVM was trained using TF-IDF features and class-balanced weights.

```python
OneVsRestClassifier(
    LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
        tol=1e-3
    ),
    n_jobs=1
)
```

The model was evaluated on the existing validation split using its default decision boundary.

### Results

| Metric   |  Score |
| -------- | -----: |
| Micro F1 | 0.7402 |
| Macro F1 | 0.6105 |

| Label         | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| Toxic         |      0.79 |   0.78 | 0.78 |
| Severe Toxic  |      0.40 |   0.61 | 0.49 |
| Obscene       |      0.82 |   0.81 | 0.82 |
| Threat        |      0.45 |   0.39 | 0.42 |
| Insult        |      0.69 |   0.74 | 0.71 |
| Identity Hate |      0.45 |   0.44 | 0.45 |

### Interpretation

The Linear SVM performed similarly to class-weighted Logistic Regression, with no clear improvement in the current comparison. Logistic Regression remains the leading candidate, although the comparison is not fully equivalent because its per-label thresholds were tuned while SVM used its default decision boundary.

The SVM initially emitted a convergence warning at 2,000 iterations. Increasing `max_iter` to 5,000 removed the warning, and the resulting metrics remained unchanged.

Threat and Identity Hate remain challenging labels. The model should be treated as a moderation-support signal rather than an automatic enforcement decision.


## Threshold-Tuned Linear SVM Comparison

### Method

The Linear SVM was evaluated using per-label decision-score thresholds selected from precision-recall curves on the validation set. These thresholds were applied to the same validation split used for the Logistic Regression experiments.

Unlike Logistic Regression, Linear SVM produces decision scores rather than probabilities. Its thresholds are therefore model-specific and should not be interpreted as probabilities.

### Results

| Model                              | Threshold Method | Micro F1 | Macro F1 |
| ---------------------------------- | ---------------- | -------: | -------: |
| Class-weighted Logistic Regression | Per-label tuned  |   0.7420 |   0.6210 |
| Linear SVM                         | Per-label tuned  |   0.7412 |   0.6223 |
| Multinomial Naive Bayes            | Per-label tuned  |   0.6742 |   0.5294 |

The tuned Linear SVM achieved a Micro F1 of 0.7412 and Macro F1 of 0.6223. Its performance was very close to class-weighted Logistic Regression.

Logistic Regression had a slightly higher Micro F1, while Linear SVM had a slightly higher Macro F1. These differences are small and do not establish a decisive winner.

### Interpretation

Both Logistic Regression and Linear SVM are viable TF-IDF baselines. The choice should consider per-label behavior, inference requirements, calibration, and the final untouched test-set results—not only the small difference in validation F1.

The tuned SVM showed a higher threat recall than its default decision-boundary result, with a corresponding precision trade-off. For TrustLens, this may be useful when prioritizing content for human review, but should not be treated as sufficient justification for automatic enforcement.

### Evaluation caveat

Per-label thresholds were selected on the validation set, and the reported tuned metrics were calculated on that same set. Final performance must be assessed on an untouched test set after model and threshold selection.
