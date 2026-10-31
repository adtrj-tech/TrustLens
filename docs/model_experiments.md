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
