"""Hybrid moderation decision layer for TrustLens."""

from src.prediction import predict_comment

HIGH_PRIORITY_LABELS = {"threat", "severe_toxic"}

REVIEW_LABELS = {
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
}

def get_confidence_band(
    probability: float,
    threshold: float,
) -> str:
    """
    Categorize a model score relative to its configured threshold.

    These bands are operational score bands, not calibrated
    probabilities of correctness.
    """
    margin = probability - threshold

    if margin >= 0.20:
        return "HIGH"

    if margin >= 0:
        return "BORDERLINE"

    return "NOT_PREDICTED"

def moderate_comment(text: str) -> dict:
    """
    Generate a moderation recommendation for one comment.

    This recommends human review only. It does not automatically
    delete, hide, or enforce action against the comment.
    """
    prediction = predict_comment(text)
    label_predictions = prediction["predictions"]
    confidence_bands = {
    label: get_confidence_band(
        result["probability"],
        result["threshold"],
    )
    for label, result in label_predictions.items()
    }

    predicted_labels = [
        label
        for label, result in label_predictions.items()
        if result["predicted"]
    ]

    high_priority_labels = [
        label
        for label in predicted_labels
        if label in HIGH_PRIORITY_LABELS
    ]

    review_labels = [
        label
        for label in predicted_labels
        if label in REVIEW_LABELS
    ]

    if high_priority_labels:
        recommendation = "PRIORITY_REVIEW"
        reason = "A high-priority moderation label was predicted."
    elif review_labels:
        recommendation = "REVIEW"
        reason = "One or more moderation labels were predicted."
    else:
        recommendation = "NO_ACTION"
        reason = "No moderation labels crossed their configured thresholds."

    return {
        "original_text": prediction["original_text"],
        "normalized_text": prediction["normalized_text"],
        "recommendation": recommendation,
        "reason": reason,
        "predicted_labels": predicted_labels,
        "high_priority_labels": high_priority_labels,
        "review_labels": review_labels,
        "model_output": label_predictions,
        "confidence_bands": confidence_bands,
    }



def test_confidence_band_high():
    assert get_confidence_band(0.95, 0.60) == "HIGH"


def test_confidence_band_borderline():
    assert get_confidence_band(0.65, 0.60) == "BORDERLINE"


def test_confidence_band_not_predicted():
    assert get_confidence_band(0.40, 0.60) == "NOT_PREDICTED"


def test_moderation_result_contains_confidence_bands():
    result = moderate_comment("You are an idiot and nobody likes you.")

    assert "confidence_bands" in result
    assert len(result["confidence_bands"]) == 6
    assert all(
        band in {"HIGH", "BORDERLINE", "NOT_PREDICTED"}
        for band in result["confidence_bands"].values()
    )