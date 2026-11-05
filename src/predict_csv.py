import argparse
from pathlib import Path

import pandas as pd

from src.prediction import (
    LABEL_COLUMNS,
    load_artifacts,
    normalize_comment,
)


def predict_csv(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if "comment_text" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'comment_text' column."
        )

    model, vectorizer, thresholds = load_artifacts()

    normalized_comments = (
        df["comment_text"]
        .fillna("")
        .astype(str)
        .map(normalize_comment)
    )

    features = vectorizer.transform(normalized_comments)
    probabilities = model.predict_proba(features)

    for index, label in enumerate(LABEL_COLUMNS):
        df[f"{label}_probability"] = probabilities[:, index]
        df[f"{label}_predicted"] = (
            probabilities[:, index] >= thresholds[label]
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Analyzed {len(df)} comments.")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run TrustLens predictions on a CSV file."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input CSV containing a comment_text column",
    )

    parser.add_argument(
        "output",
        type=str,
        help="Path where the prediction CSV should be saved",
    )

    args = parser.parse_args()
    predict_csv(args.input, args.output)


if __name__ == "__main__":
    main()