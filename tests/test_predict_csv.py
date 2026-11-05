import pandas as pd
import pytest

from src.predict_csv import predict_csv


def test_predict_csv_creates_output_file(tmp_path):
    input_file = tmp_path / "comments.csv"
    output_file = tmp_path / "predictions.csv"

    pd.DataFrame({
        "comment_text": [
            "You are a wonderful person.",
            "I disagree with your opinion.",
        ]
    }).to_csv(input_file, index=False)

    predict_csv(str(input_file), str(output_file))

    assert output_file.exists()

    result = pd.read_csv(output_file)

    assert len(result) == 2
    assert "toxic_probability" in result.columns
    assert "toxic_predicted" in result.columns
    assert "insult_probability" in result.columns
    assert "insult_predicted" in result.columns


def test_predict_csv_requires_comment_text_column(tmp_path):
    input_file = tmp_path / "invalid.csv"
    output_file = tmp_path / "predictions.csv"

    pd.DataFrame({"message": ["hello"]}).to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="comment_text"):
        predict_csv(str(input_file), str(output_file))


def test_predict_csv_rejects_missing_input_file(tmp_path):
    input_file = tmp_path / "missing.csv"
    output_file = tmp_path / "predictions.csv"

    with pytest.raises(FileNotFoundError):
        predict_csv(str(input_file), str(output_file))