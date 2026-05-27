import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.9
_THICKNESS = 2
_COLOR = (255, 255, 255)  # white text
_ANGLE = -30              # diagonal tilt in degrees


def apply_watermark(img: np.ndarray, text: str, opacity: float = 0.35) -> np.ndarray:
    """
    Stamp a diagonal repeating text watermark across the card image.

    Strategy:
      1. Build a blank text layer (same size as img) with tiled text.
      2. Where text exists, blend original pixel with white text at given opacity.

    opacity: 0.0 = invisible, 1.0 = fully opaque text.
    """
    if not text:
        return img

    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity == 0.0:
        return img

    h, w = img.shape[:2]
    text_layer = _make_text_layer(w, h, text)

    # Blend only on pixels where text was drawn.
    mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    blended = cv2.addWeighted(img, 1.0 - opacity, text_layer, opacity, 0)
    result = img.copy()
    result[mask == 255] = blended[mask == 255]

    logger.info(
        "apply_watermark: stamped '%s' at opacity=%.2f on %dx%d image",
        text, opacity, w, h,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_text_layer(w: int, h: int, text: str) -> np.ndarray:
    """
    Return a (h×w) black image with white tiled text rotated at _ANGLE.

    Works on an oversized canvas so rotation does not clip edge tiles,
    then crops the centre back to (w×h).
    """
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)

    # Pad canvas so rotated tiles still cover the full image.
    pad = int(max(w, h) * 0.6)
    cw, ch = w + 2 * pad, h + 2 * pad

    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

    step_x = tw + 60
    step_y = th + 50

    for y in range(0, ch, step_y):
        for x in range(0, cw, step_x):
            cv2.putText(canvas, text, (x, y + th),
                        _FONT, _FONT_SCALE, _COLOR, _THICKNESS, cv2.LINE_AA)

    # Rotate the tiled layer around its centre.
    cx, cy = cw // 2, ch // 2
    M = cv2.getRotationMatrix2D((cx, cy), _ANGLE, 1.0)
    rotated = cv2.warpAffine(canvas, M, (cw, ch))

    # Crop centre to match original image size.
    x0 = (cw - w) // 2
    y0 = (ch - h) // 2
    return rotated[y0: y0 + h, x0: x0 + w]
