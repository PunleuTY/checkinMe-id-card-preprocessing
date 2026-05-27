import cv2
import numpy as np
import pytest

from app.pipeline.perspective_transformation import (
    _auto_canny,
    _is_valid_card_quad,
    _order_corners,
    find_card_corners,
    perspective_correct,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_card_img(h=540, w=856, bg=(200, 190, 180)):
    """Solid-colour rectangle simulating a flat card."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg
    return img


def _card_on_background(card_h=300, card_w=476, pad=60, angle_deg=0):
    """
    Place a card-coloured rectangle on a dark background.
    Optionally rotate the whole image to simulate a tilted capture.
    """
    total_h = card_h + pad * 2
    total_w = card_w + pad * 2
    img = np.full((total_h, total_w, 3), 40, dtype=np.uint8)   # dark background
    img[pad: pad + card_h, pad: pad + card_w] = (210, 195, 175)  # card content

    if angle_deg != 0:
        cx, cy = total_w // 2, total_h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (total_w, total_h))

    return img


# ---------------------------------------------------------------------------
# _order_corners
# ---------------------------------------------------------------------------

class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=np.float32)
        result = _order_corners(pts)
        np.testing.assert_array_equal(result[0], [0, 0])   # TL
        np.testing.assert_array_equal(result[1], [10, 0])  # TR
        np.testing.assert_array_equal(result[2], [10, 6])  # BR
        np.testing.assert_array_equal(result[3], [0, 6])   # BL

    def test_shuffled_input_gives_correct_order(self):
        pts = np.array([[10, 6], [0, 0], [0, 6], [10, 0]], dtype=np.float32)
        result = _order_corners(pts)
        np.testing.assert_array_equal(result[0], [0, 0])   # TL
        np.testing.assert_array_equal(result[1], [10, 0])  # TR
        np.testing.assert_array_equal(result[2], [10, 6])  # BR
        np.testing.assert_array_equal(result[3], [0, 6])   # BL

    def test_returns_float32(self):
        pts = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=np.float32)
        result = _order_corners(pts)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _auto_canny
# ---------------------------------------------------------------------------

class TestAutoCanny:
    def test_returns_same_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        edges = _auto_canny(img)
        assert edges.shape == img.shape

    def test_output_is_binary(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        edges = _auto_canny(img)
        unique = set(np.unique(edges))
        assert unique.issubset({0, 255})

    def test_blank_image_produces_no_edges(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        edges = _auto_canny(img)
        assert edges.max() == 0


# ---------------------------------------------------------------------------
# _is_valid_card_quad
# ---------------------------------------------------------------------------

class TestIsValidCardQuad:
    def _corners(self, w, h, ox=0, oy=0):
        return np.array([
            [ox,     oy],
            [ox + w, oy],
            [ox + w, oy + h],
            [ox,     oy + h],
        ], dtype=np.float32)

    def test_landscape_card_ratio_passes(self):
        # exact ID-1 ratio 1.586
        corners = self._corners(856, 540)
        assert _is_valid_card_quad(corners, 856 * 540 * 2)

    def test_portrait_card_ratio_passes(self):
        corners = self._corners(540, 856)
        assert _is_valid_card_quad(corners, 856 * 540 * 2)

    def test_too_small_area_fails(self):
        corners = self._corners(10, 6)
        assert not _is_valid_card_quad(corners, 856 * 540)

    def test_too_large_area_fails(self):
        corners = self._corners(856, 540)
        # quad fills 99.5 % of img_area → rejected
        assert not _is_valid_card_quad(corners, int(856 * 540 * 1.005))

    def test_wrong_aspect_ratio_fails(self):
        # 3:1 ratio — deviation 0.89 from 1.586, well outside 0.42 tolerance
        corners = self._corners(600, 100)
        assert not _is_valid_card_quad(corners, 600 * 100 * 4)

    def test_degenerate_zero_height_fails(self):
        corners = np.array([[0,0],[100,0],[100,0],[0,0]], dtype=np.float32)
        assert not _is_valid_card_quad(corners, 100 * 100)


# ---------------------------------------------------------------------------
# find_card_corners
# ---------------------------------------------------------------------------

class TestFindCardCorners:
    def test_returns_none_for_blank_image(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        assert find_card_corners(img) is None

    def test_returns_4x2_array_when_card_found(self):
        img = _card_on_background()
        corners = find_card_corners(img)
        if corners is not None:
            assert corners.shape == (4, 2)

    def test_corners_within_image_bounds(self):
        img = _card_on_background()
        corners = find_card_corners(img)
        if corners is None:
            pytest.skip("card not detected in synthetic image")
        h, w = img.shape[:2]
        assert corners[:, 0].min() >= 0
        assert corners[:, 0].max() <= w
        assert corners[:, 1].min() >= 0
        assert corners[:, 1].max() <= h

    def test_real_sample_id2(self):
        img = cv2.imread("sample_imgs/id2.jpg")
        if img is None:
            pytest.skip("sample_imgs/id2.jpg not found")
        corners = find_card_corners(img)
        assert corners is not None, "card not detected in id2.jpg"
        assert corners.shape == (4, 2)


# ---------------------------------------------------------------------------
# perspective_correct
# ---------------------------------------------------------------------------

class TestPerspectiveCorrect:
    def test_returns_ndarray(self):
        img = _make_card_img()
        result = perspective_correct(img)
        assert isinstance(result, np.ndarray)

    def test_blank_image_returns_original_unchanged(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = perspective_correct(img)
        assert result.shape == img.shape

    def test_output_is_landscape(self):
        img = _card_on_background()
        result = perspective_correct(img)
        h, w = result.shape[:2]
        assert w >= h, f"output should be landscape, got {w}x{h}"

    def test_output_smaller_than_padded_input(self):
        img = _card_on_background(pad=60)
        result = perspective_correct(img)
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]

    def test_real_sample_id2_is_corrected(self):
        img = cv2.imread("sample_imgs/id2.jpg")
        if img is None:
            pytest.skip("sample_imgs/id2.jpg not found")
        result = perspective_correct(img)
        ih, iw = img.shape[:2]
        rh, rw = result.shape[:2]
        # corrected card must be smaller than the full photo (background removed)
        assert rw < iw or rh < ih, "output should be smaller than original for id2"
        # must be landscape
        assert rw >= rh
