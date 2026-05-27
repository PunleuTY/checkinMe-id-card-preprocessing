import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BLACK_THRESHOLD = 5  # pixels with all channels ≤ this are treated as warp-fill black
_ERODE_PX = 3  # shrink mask inward to strip 1-2px edge noise

"""
 What it does:
  1. Builds a mask of all pixels where any channel value is above 5 — these are card content
  2. Erodes the mask by 3px to strip single-pixel edge noise at the card border 
  3. Finds the bounding box of the remaining content
  4. Crops and returns that bounding box as the clean card

  Result: a tight rectangle of just the card content, no black padding, ready for the next pipeline step (normalize resolution).

  Fallbacks:
  - If the entire image is black → returns original unchanged
  - If the bounding box is smaller than 10×10px → returns original unchanged
  - If there are no black corners (tight-crop input) → bounding box equals full image, returns as-is (no-op)
  - Image must only contain one card — if multiple cards are present, they will be merged into one bounding box and cropped together.(somehow you'll get only one card in the output)
  """


def segment_card(img: np.ndarray) -> np.ndarray:
    """
    Remove black corner artifacts left by warpPerspective and return a
    tight crop of the card content.

    warpPerspective fills out-of-bounds pixels with exact (0, 0, 0).
    We mask those out, find the bounding box of the remaining content,
    and crop to it.  If there are no black corners (tight-crop input)
    the bounding box equals the full image and the function is a no-op.
    """
    mask = _content_mask(img)

    coords = cv2.findNonZero(mask)
    if coords is None:
        logger.warning("segment_card: mask is empty — returning original image")
        return img

    x, y, w, h = cv2.boundingRect(coords)

    if w < 10 or h < 10:
        logger.warning(
            "segment_card: bounding box too small (%dx%d) — returning original image",
            w,
            h,
        )
        return img

    cropped = img[y : y + h, x : x + w]
    logger.info(
        "segment_card: cropped to %dx%d from %dx%d", w, h, img.shape[1], img.shape[0]
    )
    return cropped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_mask(img: np.ndarray) -> np.ndarray:
    """
    Return a single-channel mask: 255 where the pixel is card content,
    0 where it is a black warp artifact.
    """
    # Mark pixels where ANY channel exceeds the black threshold.
    above = np.any(img > _BLACK_THRESHOLD, axis=2).astype(np.uint8) * 255

    # Erode slightly to remove isolated edge-noise pixels at the card border.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_ERODE_PX, _ERODE_PX))
    eroded = cv2.erode(above, kernel, iterations=1)

    return eroded
