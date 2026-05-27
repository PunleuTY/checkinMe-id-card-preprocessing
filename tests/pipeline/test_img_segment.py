import numpy as np
import pytest
from app.pipeline.img_segment import segment_card, _content_mask


def _solid(h, w, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _with_black_border(h, w, border):
    """Card content in the center, solid black rows/columns around it (like an over-sized warp canvas)."""
    img = _solid(h, w, (200, 180, 160))
    img[:border, :] = 0        # top rows
    img[h - border:, :] = 0   # bottom rows
    img[:, :border] = 0        # left columns
    img[:, w - border:] = 0   # right columns
    return img


# ---------------------------------------------------------------------------
# _content_mask
# ---------------------------------------------------------------------------

class TestContentMask:
    def test_all_black_returns_empty_mask(self):
        img = _solid(50, 80, (0, 0, 0))
        mask = _content_mask(img)
        assert mask.max() == 0

    def test_all_white_returns_full_mask(self):
        img = _solid(50, 80, (255, 255, 255))
        mask = _content_mask(img)
        # After erosion the mask may shrink slightly at borders — just ensure mostly filled.
        assert mask.mean() > 200

    def test_near_black_pixels_treated_as_black(self):
        img = _solid(50, 80, (5, 5, 5))  # all channels == threshold, not above
        mask = _content_mask(img)
        assert mask.max() == 0

    def test_one_channel_above_threshold_counts_as_content(self):
        img = _solid(50, 80, (0, 0, 6))  # blue channel just above threshold
        mask = _content_mask(img)
        assert mask.max() == 255


# ---------------------------------------------------------------------------
# segment_card
# ---------------------------------------------------------------------------

class TestSegmentCard:
    def test_no_black_corners_returns_same_size(self):
        img = _solid(100, 160, (200, 200, 200))
        result = segment_card(img)
        # Slight shrinkage from erosion is acceptable; should not grow.
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]

    def test_black_border_rows_and_columns_are_cropped(self):
        # Solid black rows/columns around the image (over-sized warp canvas case).
        img = _with_black_border(100, 160, border=15)
        result = segment_card(img)
        # Output must be smaller than input on both axes.
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]

    def test_cropped_output_starts_with_content_not_black(self):
        img = _with_black_border(100, 160, border=15)
        result = segment_card(img)
        # Top-left pixel of the cropped result should not be black.
        assert result[0, 0].max() > 5

    def test_fully_black_image_returns_original(self):
        img = _solid(50, 80, (0, 0, 0))
        result = segment_card(img)
        assert result.shape == img.shape

    def test_small_bounding_box_returns_original(self):
        # Only a 5x5 non-black region — below minimum size.
        img = _solid(50, 80, (0, 0, 0))
        img[24:29, 38:43] = (200, 200, 200)
        result = segment_card(img)
        assert result.shape == img.shape

    def test_real_sample_id2(self):
        import cv2
        from app.pipeline.perspective_transformation import perspective_correct
        img = cv2.imread("sample_imgs/id2.jpg")
        if img is None:
            pytest.skip("sample_imgs/id2.jpg not found")
        warped = perspective_correct(img)
        result = segment_card(warped)
        # Result must not be larger than the warped image.
        assert result.shape[0] <= warped.shape[0]
        assert result.shape[1] <= warped.shape[1]
        # Result must have actual content.
        assert result.size > 0
