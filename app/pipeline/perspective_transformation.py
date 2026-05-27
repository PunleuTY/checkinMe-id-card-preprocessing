import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1 card: 85.60 × 53.98 mm → ratio ≈ 1.586
_CARD_ASPECT_RATIO = 85.60 / 53.98
_ASPECT_TOLERANCE = 0.42  # allow 42% deviation for angled shots
_MIN_CARD_AREA_RATIO = 0.04  # card must occupy at least 4% of the image
_MAX_CARD_AREA_RATIO = 0.99  # reject only exact-boundary quads
_WORK_LONG_EDGE = 1000  # downscale long edge for stable edge detection
_PAD_PX = 30  # border padding added when tight-crop fallback runs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[2] = pts[np.argmax(s)]  # bottom-right
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


def _search_contours(
    small: np.ndarray,
    small_area: float,
    retrieval: int = cv2.RETR_LIST,
) -> Optional[np.ndarray]:
    """
    Run the full edge→contour→quad pipeline on a pre-scaled image.
    Returns ordered corners in *small*-coordinate space, or None.

    retrieval: cv2.RETR_LIST (Pass 1, finds all contours) or
               cv2.RETR_EXTERNAL (Pass 2/padded, finds only outer contours).

    Three preprocessing strategies are tried in order:
      1. Gaussian blur + adaptive Canny  — works for high-contrast scenes
      2. CLAHE + adaptive Canny          — works for low-contrast/uneven lighting
      3. Bilateral filter + fixed Canny  — works for textured backgrounds (card on surface)
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    strategies = [
        # (processed_image, canny_lower, canny_upper, close_kernel_size)
        (cv2.GaussianBlur(gray, (5, 5), 0), None, None, 5),
        (
            cv2.GaussianBlur(
                cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
                (5, 5),
                0,
            ),
            None,
            None,
            5,
        ),
        (cv2.bilateralFilter(gray, 9, 75, 75), 30, 100, 11),
    ]

    for processed, lo, hi, kern_size in strategies:
        edges = _auto_canny(processed) if lo is None else cv2.Canny(processed, lo, hi)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_size, kern_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:15]:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
                approx = cv2.approxPolyDP(contour, eps * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
                    if _is_valid_card_quad(corners, small_area):
                        return corners

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_card_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate the four corners of an ID card in *img*.

    Strategy:
      1. Downscale to _WORK_LONG_EDGE for stable detection, search contours.
      2. If not found (tight-crop / no background), pad with a black border and retry —
         the border creates a clear card-background edge for the detector.

    Returns ordered corners [TL, TR, BR, BL] as float32 (4×2) in the coordinate
    space of the original image, or None when no card is found.
    """
    h, w = img.shape[:2]

    long_edge = max(h, w)
    scale = _WORK_LONG_EDGE / float(long_edge) if long_edge > _WORK_LONG_EDGE else 1.0
    small = (
        cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale != 1.0
        else img
    )
    sh, sw = small.shape[:2]
    small_area = float(sh * sw)

    # --- Pass 1: normal detection (all contours, card against a visible background) ---
    corners = _search_contours(small, small_area, cv2.RETR_LIST)
    if corners is not None:
        return corners / scale

    # --- Pass 2: pad with a black border so tight-crop cards get a detectable edge.
    #     Use RETR_EXTERNAL — the card boundary is now the dominant outer contour. ---
    p = _PAD_PX
    padded = cv2.copyMakeBorder(small, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ph, pw = padded.shape[:2]
    padded_area = float(ph * pw)

    corners = _search_contours(padded, padded_area, cv2.RETR_EXTERNAL)
    if corners is not None:
        # Subtract padding offset, then map back to original-image coordinates.
        corners -= _PAD_PX
        return corners / scale

    return None


def perspective_correct(img: np.ndarray) -> np.ndarray:
    """
    Detect the ID card in *img* and return a flat, de-skewed crop.

    Falls back to returning *img* unchanged if no card is detected.
    """
    corners = find_card_corners(img)

    if corners is None:
        logger.warning(
            "perspective_correct: card not detected — returning original image"
        )
        return img

    tl, tr, br, bl = corners

    # Output size from the real edge lengths (preserves true proportions).
    out_w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    out_h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    if out_w < 1 or out_h < 1:
        logger.warning(
            "perspective_correct: degenerate quad — returning original image"
        )
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

    logger.info(
        "perspective_correct: card detected — output %dx%d",
        warped.shape[1],
        warped.shape[0],
    )
    return warped


"""
command to test the perspective transformation in isolation (requires curl, base64, python3):

BASE64=$(base64 -i /path/to/your/image.jpg) && \
  curl -s -X POST http://127.0.0.1:8000/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$BASE64\"}" \
    | python3 -c "
  import sys, json, base64
  r = json.load(sys.stdin)
  open('/tmp/result.jpg','wb').write(base64.b64decode(r['segmented_image']))
  " && open /tmp/result.jpg && open /path/to/your/image.jpg
  
  """
