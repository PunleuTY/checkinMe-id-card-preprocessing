import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_resolution(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize the card image to the target resolution using the best interpolation
    for the direction of scaling:
      - Shrinking → INTER_AREA   (avoids moiré, preserves detail)
      - Enlarging → INTER_CUBIC  (smooth upscale)

    The image is stretched to exactly (target_w × target_h) without preserving
    aspect ratio — the card has already been perspective-corrected and segmented,
    so both dimensions are meaningful and should match the expected output size.
    """
    src_h, src_w = img.shape[:2]

    if src_w == target_w and src_h == target_h:
        logger.info("normalize_resolution: already %dx%d, no resize needed", target_w, target_h)
        return img

    shrinking = (target_w * target_h) < (src_w * src_h)
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC

    resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    logger.info(
        "normalize_resolution: %dx%d → %dx%d",
        src_w, src_h, target_w, target_h,
    )
    return resized
