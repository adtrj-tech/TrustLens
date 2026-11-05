from src.moderation import moderate_comment


def test_insulting_comment_is_recommended_for_review():
    result = moderate_comment(
        "You are an idiot and nobody likes you."
    )

    assert result["recommendation"] in {
        "REVIEW",
        "PRIORITY_REVIEW",
    }

    assert "toxic" in result["predicted_labels"]
    assert "insult" in result["predicted_labels"]


def test_non_empty_comment_returns_expected_structure():
    result = moderate_comment("I disagree with your opinion.")

    expected_keys = {
        "original_text",
        "normalized_text",
        "recommendation",
        "reason",
        "predicted_labels",
        "high_priority_labels",
        "review_labels",
        "model_output",
    }

    assert expected_keys.issubset(result.keys())


def test_moderation_does_not_perform_automatic_enforcement():
    result = moderate_comment("You are a wonderful person.")

    assert result["recommendation"] in {
        "NO_ACTION",
        "REVIEW",
        "PRIORITY_REVIEW",
    }

    assert "delete" not in result
    assert "ban" not in result