# TrustLens — Model Evaluation

## 1. Objective

Evaluate initial multi-label toxicity classification models for TrustLens. The models predict six toxicity categories independently, allowing a comment to receive multiple labels.

## 2. Dataset

**Dataset:** Jigsaw Toxic Comment Classification
**Training examples:** 159,571
**Labels:**

* Toxic
* Severe toxic
* Obscene
* Threat
* Insult
* Identity hate

The dataset is imbalanced. Threat and identity-hate examples are particularly rare, so accuracy alone is not an adequate evaluation metric.

## 3. Data Preparation

* Train/validation split: 80% / 20%
* Training examples: 127,656
* Validation examples: 31,915
* Split method: `MultilabelStratifiedShuffleSplit`
* Random state: `42`
* Text normalization: HTML entity decoding, whitespace normalization, and trimming
* Feature extraction: word-level TF-IDF
* N-gram range: `(1, 2)`
* Minimum document frequency: `2`
* Maximum document frequency: `0.98`
* Maximum features: `300,000`
* Sublinear TF: enabled
* Feature dtype: `float32`

The TF-IDF vectorizer was fitted only on the training split and then applied to validation data to avoid vocabulary leakage.

## 4. Models

Both experiments used One-vs-Rest Logistic Regression with the `liblinear` solver.

### Experiment A — Unweighted Logistic Regression

Configuration:

* `max_iter=1000`
* `solver="liblinear"`
* `class_weight=None`
* `random_state=42`

### Experiment B — Class-Weighted Logistic Regression

Configuration:

* `max_iter=1000`
* `solver="liblinear"`
* `class_weight="balanced"`
* `random_state=42`

Class weighting was tested to address label imbalance.

## 5. Evaluation Metrics

The experiments report:

* Precision
* Recall
* F1-score
* Micro F1
* Macro F1

Per-label thresholds were selected using precision-recall curves on the validation set.

## 6. Results

### Overall Results

| Model                              | Threshold strategy   | Micro F1 | Macro F1 |
| ---------------------------------- | -------------------- | -------: | -------: |
| Unweighted Logistic Regression     | Per-label thresholds |    0.736 |    0.608 |
| Class-Weighted Logistic Regression | Per-label thresholds |    0.742 |    0.621 |

The class-weighted model improved overall Micro F1 by approximately 0.006 and Macro F1 by approximately 0.013. This is a modest improvement, not a decisive performance leap.

### Class-Weighted Logistic Regression — Per-Label Results

| Label         | Threshold | Precision | Recall |    F1 |
| ------------- | --------: | --------: | -----: | ----: |
| Toxic         |     0.609 |     0.763 |  0.791 | 0.777 |
| Severe toxic  |     0.907 |     0.447 |  0.611 | 0.517 |
| Obscene       |     0.680 |     0.830 |  0.796 | 0.813 |
| Threat        |     0.880 |     0.476 |  0.406 | 0.438 |
| Insult        |     0.743 |     0.717 |  0.724 | 0.721 |
| Identity hate |     0.838 |     0.440 |  0.480 | 0.459 |

## 7. Interpretation

Class weighting improved five of the six label-level F1 scores. Threat classification was marginally lower than the unweighted experiment.

The strongest results were for toxic and obscene comments. Threat and identity-hate classification remain substantially weaker and require further investigation.

The results support using class-weighted Logistic Regression as the current baseline candidate, while continuing to test alternatives.

## 8. Limitations

* Thresholds were selected and evaluated on the same validation set. Reported tuned scores may therefore be optimistic.
* No independent final test evaluation has been completed.
* Threat and identity-hate labels have relatively few positive examples.
* The dataset reflects its collection and annotation process; performance may not generalize equally to every community or platform.
* Model predictions should not be treated as definitive judgments about user intent.

## 9. Current Decision

**Current candidate:** Class-Weighted One-vs-Rest Logistic Regression with per-label thresholds.

This is a baseline for comparison—not the final production model.

Next experiments:

1. Multinomial Naive Bayes baseline
2. Compare models using the same validation split and metrics
3. Review threshold behavior and false positives/false negatives
4. Evaluate the selected model on an independent test set before making final performance claims
