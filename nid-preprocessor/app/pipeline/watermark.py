import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.9
_THICKNESS = 2
_COLOR = (255, 255, 255)  # white text


def apply_watermark(img: np.ndarray, text: str, opacity: float = 0.35) -> np.ndarray:
    """
    Stamp a diagonal repeating text watermark across the card image.

    The watermark is drawn on a transparent overlay and blended with the
    original using the given opacity (0.0 = invisible, 1.0 = fully opaque).
    """
    if not text:
        return img

    opacity = float(np.clip(opacity, 0.0, 1.0))
    h, w = img.shape[:2]

    overlay = img.copy()
    _draw_tiled_text(overlay, text, w, h)

    result = cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0)
    logger.info("apply_watermark: stamped '%s' at opacity %.2f on %dx%d image", text, opacity, w, h)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_tiled_text(canvas: np.ndarray, text: str, w: int, h: int) -> None:
    """Draw the watermark text in a diagonal repeating grid across canvas."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)

    step_x = tw + 60
    step_y = th + 50
    angle = -30

    # Rotate canvas centre so tiles appear diagonal.
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (w, h))

    for y in range(-step_y, h + step_y, step_y):
        for x in range(-step_x, w + step_x, step_x):
            cv2.putText(rotated, text, (x, y), _FONT, _FONT_SCALE, _COLOR, _THICKNESS, cv2.LINE_AA)

    # Rotate back and blend only the text pixels into the original canvas.
    M_inv = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    text_layer = cv2.warpAffine(rotated, M_inv, (w, h))

    mask = cv2.cvtColor(
        cv2.absdiff(text_layer, canvas), cv2.COLOR_BGR2GRAY
    )
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    canvas[mask == 255] = text_layer[mask == 255]
