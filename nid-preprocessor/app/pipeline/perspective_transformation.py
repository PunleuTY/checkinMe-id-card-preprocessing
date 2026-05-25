import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1 card: 85.60 × 53.98 mm → ratio ≈ 1.586
_CARD_ASPECT_RATIO = 85.60 / 53.98
_ASPECT_TOLERANCE = 0.30   # allow 30% deviation for angled shots
_MIN_CARD_AREA_RATIO = 0.05  # card must occupy at least 5% of the image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Return 4 points ordered [top-left, top-right, bottom-right, bottom-left].

    Uses sum and difference of (x, y) as discriminators:
      top-left     → smallest x+y
      bottom-right → largest  x+y
      top-right    → smallest y-x
      bottom-left  → largest  y-x
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]     # top-left
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _is_valid_card_quad(corners: np.ndarray, img_area: int) -> bool:
    """
    Reject quads that are too small or whose aspect ratio does not match
    a standard ID card (landscape or portrait orientation both accepted).
    """
    tl, tr, br, bl = corners

    avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0

    if avg_h < 1:
        return False

    if avg_w * avg_h < img_area * _MIN_CARD_AREA_RATIO:
        return False

    ratio = avg_w / avg_h
    expected = _CARD_ASPECT_RATIO

    landscape_ok = abs(ratio - expected) / expected <= _ASPECT_TOLERANCE
    portrait_ok = abs((1.0 / ratio) - expected) / expected <= _ASPECT_TOLERANCE

    return landscape_ok or portrait_ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_card_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate the four corners of an ID card in *img*.

    Strategy:
      1. Grayscale + Gaussian blur to suppress noise.
      2. Canny edge detection.
      3. Dilate edges to close small gaps along the card border.
      4. Find external contours, sort by area (largest first).
      5. Approximate each contour to a polygon; keep the first
         quadrilateral whose size and aspect ratio match an ID card.

    Returns ordered corners [TL, TR, BR, BL] as float32 (4×2),
    or None when no suitable card is found.
    """
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    # Close small gaps so the card outline forms a single closed contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            continue

        corners = _order_corners(approx.reshape(4, 2).astype(np.float32))

        if _is_valid_card_quad(corners, img_area):
            return corners

    return None


def perspective_correct(img: np.ndarray) -> np.ndarray:
    """
    Detect the ID card in *img* and return a flat, de-skewed crop.

    Steps:
      1. Find the four card corners.
      2. Compute output dimensions from the real edge lengths
         (preserves the card's true proportions rather than forcing
         a fixed size — normalisation to 1024×640 happens downstream).
      3. Build the homography with getPerspectiveTransform.
      4. Warp with warpPerspective.
      5. Rotate to landscape if the result comes out portrait.

    Falls back to returning *img* unchanged if no card is detected,
    so the pipeline continues with a degraded but non-crashing result.
    """
    corners = find_card_corners(img)

    if corners is None:
        logger.warning("perspective_correct: card not detected — returning original image")
        return img

    tl, tr, br, bl = corners

    # Average opposite edge lengths for sub-pixel accuracy
    out_w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    out_h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    # Ensure landscape orientation
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped
