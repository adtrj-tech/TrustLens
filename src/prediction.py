import html
import json
import re
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def normalize_comment(text: str) -> str:
    """Apply the same conservative normalization used during training."""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_artifacts():
    """Load the trained classifier, fitted vectorizer, and thresholds."""
    model = joblib.load(MODEL_DIR / "trustlens_logistic_weighted.joblib")
    vectorizer = joblib.load(MODEL_DIR / "trustlens_tfidf_vectorizer.joblib")

    with open(
        MODEL_DIR / "trustlens_logistic_weighted_thresholds.json",
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    thresholds = {
        label: float(threshold_data[label])
        for label in LABEL_COLUMNS
    }

    return model, vectorizer, thresholds


def predict_comment(text: str) -> dict:
    """Predict moderation labels for a single raw comment."""
    if not isinstance(text, str):
        raise TypeError("Comment must be a string.")

    model, vectorizer, thresholds = load_artifacts()

    normalized_text = normalize_comment(text)
    features = vectorizer.transform([normalized_text])
    probabilities = model.predict_proba(features)[0]

    results = {}

    for index, label in enumerate(LABEL_COLUMNS):
        probability = float(probabilities[index])
        threshold = thresholds[label]

        results[label] = {
            "probability": probability,
            "threshold": threshold,
            "predicted": probability >= threshold,
        }

    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "predictions": results,
    }