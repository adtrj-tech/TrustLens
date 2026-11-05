from src.prediction import predict_comment


def test_predict_comment_returns_all_labels():
    result = predict_comment("You are a wonderful person.")

    assert "predictions" in result
    assert len(result["predictions"]) == 6


def test_prediction_values_have_expected_fields():
    result = predict_comment("You are a wonderful person.")

    for label_result in result["predictions"].values():
        assert "probability" in label_result
        assert "threshold" in label_result
        assert "predicted" in label_result
        assert 0 <= label_result["probability"] <= 1
        assert isinstance(label_result["predicted"], bool)


def test_prediction_rejects_non_string_input():
    try:
        predict_comment(None)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-string input")