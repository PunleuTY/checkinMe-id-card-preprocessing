import base64
import numpy as np
import cv2


def b64_to_ndarray(b64: str) -> np.ndarray:
    """Decode a base64 string to a BGR numpy array."""
    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 data.")
    return img


def ndarray_to_b64(img: np.ndarray, fmt: str = ".jpg", **encode_params) -> str:
    """Encode a BGR numpy array to a base64 string."""
    ok, buf = cv2.imencode(fmt, img, encode_params.get("params", []))
    if not ok:
        raise ValueError(f"Could not encode image to {fmt}.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def ndarray_to_webp_b64(img: np.ndarray, quality: int = 85) -> str:
    params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    return ndarray_to_b64(img, fmt=".webp", params=params)
