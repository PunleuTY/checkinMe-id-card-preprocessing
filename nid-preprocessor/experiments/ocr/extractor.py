"""
Thin wrapper around kiri-ocr for NID card text extraction.

Accepts either a file path or a preprocessed BGR numpy array.
Returns raw text and per-line results with confidence scores.
"""

import logging
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_from_array(img: np.ndarray, decode_method: str = "accurate") -> dict:
    """
    Run OCR on a preprocessed BGR numpy array.

    decode_method: "fast" | "accurate" | "beam"
    Returns {"text": str, "lines": list[dict]}
    """
    try:
        from kiri_ocr import OCR
    except ImportError:
        raise ImportError(
            "kiri-ocr is not installed. "
            "Run: pip install kiri-ocr  (or pip install -r experiments/ocr/requirements.txt)"
        )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name

    return extract_from_file(tmp_path, decode_method=decode_method)


def extract_from_file(image_path: str, decode_method: str = "accurate") -> dict:
    """
    Run OCR directly on an image file path.

    Returns {"text": str, "lines": list[dict]}
    """
    try:
        from kiri_ocr import OCR
    except ImportError:
        raise ImportError(
            "kiri-ocr is not installed. "
            "Run: pip install kiri-ocr  (or pip install -r experiments/ocr/requirements.txt)"
        )

    logger.info("extractor: running OCR on %s (method=%s)", image_path, decode_method)

    ocr = OCR(decode_method=decode_method)
    text, results = ocr.extract_text(image_path)

    logger.info("extractor: extracted %d chars, %d lines", len(text), len(results))

    return {
        "text": text,
        "lines": results,
    }
