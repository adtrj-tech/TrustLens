import argparse
import json

from src.prediction import predict_comment


def main():
    parser = argparse.ArgumentParser(
        description="Run TrustLens moderation predictions on a comment."
    )
    parser.add_argument(
        "comment",
        type=str,
        help="The comment text to analyze",
    )

    args = parser.parse_args()
    result = predict_comment(args.comment)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()