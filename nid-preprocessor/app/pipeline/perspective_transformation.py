import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1 card: 85.60 × 53.98 mm → ratio ≈ 1.586
_CARD_ASPECT_RATIO = 85.60 / 53.98
_ASPECT_TOLERANCE = 0.40          # allow 40% deviation for angled shots
_MIN_CARD_AREA_RATIO = 0.05       # card must occupy at least 5% of the image
_MAX_CARD_AREA_RATIO = 0.98       # reject quads that are just the image border
_WORK_LONG_EDGE = 1000            # downscale long edge for stable edge detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Return 4 points ordered [top-left, top-right, bottom-right, bottom-left].

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


def _auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image median — adapts to lighting."""
    v = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def _is_valid_card_quad(corners: np.ndarray, img_area: float) -> bool:
    """Accept quads whose size and aspect ratio match a standard ID card."""
    tl, tr, br, bl = corners

    avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if avg_h < 1 or avg_w < 1:
        return False

    quad_area = avg_w * avg_h
    if quad_area < img_area * _MIN_CARD_AREA_RATIO:
        return False
    if quad_area > img_area * _MAX_CARD_AREA_RATIO:
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

    Works on a downscaled copy for stable edge detection, then scales the
    detected corners back to the original resolution.

    Returns ordered corners [TL, TR, BR, BL] as float32 (4×2) in the
    coordinate space of the original image, or None when no card is found.
    """
    h, w = img.shape[:2]

    # Downscale so detection behaves the same regardless of input resolution.
    long_edge = max(h, w)
    scale = _WORK_LONG_EDGE / float(long_edge) if long_edge > _WORK_LONG_EDGE else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    sh, sw = small.shape[:2]
    small_area = float(sh * sw)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = _auto_canny(blur)

    # Close gaps so the card outline forms a single closed contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        # Try a range of approximation tolerances to land on a clean quad.
        for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
                if _is_valid_card_quad(corners, small_area):
                    return corners / scale  # back to original-image coordinates

    return None


def perspective_correct(img: np.ndarray) -> np.ndarray:
    """
    Detect the ID card in *img* and return a flat, de-skewed crop.

    Falls back to returning *img* unchanged if no card is detected, so the
    pipeline continues with a degraded but non-crashing result.
    """
    corners = find_card_corners(img)

    if corners is None:
        logger.warning("perspective_correct: card not detected — returning original image")
        return img

    tl, tr, br, bl = corners

    # Output size from the real edge lengths (preserves true proportions).
    out_w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    out_h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    if out_w < 1 or out_h < 1:
        logger.warning("perspective_correct: degenerate quad — returning original image")
        return img

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    # Ensure landscape orientation.
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    logger.info("perspective_correct: card detected — output %dx%d", warped.shape[1], warped.shape[0])
    return warped
