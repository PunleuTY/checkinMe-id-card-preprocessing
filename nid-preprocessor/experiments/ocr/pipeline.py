"""
Full experiment pipeline: preprocess NID card image → extract OCR text.

Usage:
    python -m experiments.ocr.pipeline path/to/id_card.jpg
    python -m experiments.ocr.pipeline path/to/id_card.jpg --method fast
    python -m experiments.ocr.pipeline path/to/id_card.jpg --save-preprocessed /tmp/clean.jpg
"""

import argparse
import json
import logging
import sys

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run(
    image_path: str, decode_method: str = "accurate", save_preprocessed: str = None
) -> dict:
    from app.pipeline.perspective_transformation import perspective_correct
    from app.pipeline.img_segment import segment_card
    from app.pipeline.normalize import normalize_resolution
    from app.core.config import settings
    from experiments.ocr.extractor import extract_from_array

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    logger.info("pipeline: loaded %s (%dx%d)", image_path, img.shape[1], img.shape[0])

    img = perspective_correct(img)
    img = segment_card(img)
    img = normalize_resolution(img, settings.output_width, settings.output_height)

    if save_preprocessed:
        cv2.imwrite(save_preprocessed, img)
        logger.info("pipeline: saved preprocessed card to %s", save_preprocessed)

    result = extract_from_array(img, decode_method=decode_method)
    return result


def main():
    parser = argparse.ArgumentParser(description="NID preprocess + OCR experiment")
    parser.add_argument("image", help="Path to input ID card image")
    parser.add_argument(
        "--method",
        default="accurate",
        choices=["fast", "accurate", "beam"],
        help="kiri-ocr decode method (default: accurate)",
    )
    parser.add_argument(
        "--save-preprocessed",
        default=None,
        metavar="PATH",
        help="Optional path to save the preprocessed card before OCR",
    )
    args = parser.parse_args()

    result = run(
        args.image, decode_method=args.method, save_preprocessed=args.save_preprocessed
    )

    print("\n--- Extracted Text ---")
    print(result["text"])
    print("\n--- Raw Lines (JSON) ---")
    print(json.dumps(result["lines"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
