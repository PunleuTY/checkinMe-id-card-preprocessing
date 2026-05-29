"""
Experiment: send a raw NID card image straight to Gemini for text extraction.

No local preprocessing — this tests Gemini's end-to-end capability on the
original image (Gemini handles deskew/denoise/reading internally).

Usage:
    python -m experiments.gemini.pipeline path/to/id_card.jpg
    python -m experiments.gemini.pipeline path/to/id_card.jpg --model gemini-2.5-pro
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NID text extraction with Gemini")
    parser.add_argument("image", help="Path to input ID card image")
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model override (default: from settings.gemini_model)",
    )
    args = parser.parse_args()

    from gemini.extractor import extract_from_file

    result = extract_from_file(args.image, model=args.model)

    print(f"\n--- Model: {result['model']} ---")
    print("\n--- Extracted Text ---")
    print(result["text"])
    print("\n--- Structured Fields (JSON) ---")
    print(json.dumps(result["fields"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
