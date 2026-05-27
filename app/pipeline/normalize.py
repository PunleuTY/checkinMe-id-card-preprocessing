import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_resolution(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """
    Resize the card to fit within (target_w x target_h) while preserving aspect
    ratio, then pad to exactly that size.

    Padding avoids the distortion caused by blind stretching when the detected
    quad is slightly off — text stays readable and the card shape stays correct.
    """
    src_h, src_w = img.shape[:2]

    if src_w == target_w and src_h == target_h:
        logger.info("normalize_resolution: already %dx%d, no resize needed", target_w, target_h)
        return img

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    shrinking = scale < 1.0
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, img.shape[2]), pad_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    logger.info(
        "normalize_resolution: %dx%d → %dx%d (scale=%.3f, pad x=%d y=%d)",
        src_w, src_h, target_w, target_h, scale, x_off, y_off,
    )
    return canvas
