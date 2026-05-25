import numpy as np
import pytest
from app.pipeline.watermark import apply_watermark


def _img(h=640, w=1024):
    return np.full((h, w, 3), 100, dtype=np.uint8)


class TestApplyWatermark:
    def test_output_same_shape_as_input(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=0.35)
        assert result.shape == img.shape

    def test_output_is_uint8(self):
        result = apply_watermark(_img(), "CheckinMe", opacity=0.35)
        assert result.dtype == np.uint8

    def test_watermark_modifies_image(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=0.35)
        assert not np.array_equal(result, img)

    def test_empty_text_returns_original(self):
        img = _img()
        result = apply_watermark(img, "")
        assert np.array_equal(result, img)

    def test_zero_opacity_returns_original(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=0.0)
        assert np.array_equal(result, img)

    def test_full_opacity_applies_watermark(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=1.0)
        assert not np.array_equal(result, img)

    def test_opacity_clipped_above_one(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=5.0)
        assert result.shape == img.shape

    def test_opacity_clipped_below_zero(self):
        img = _img()
        result = apply_watermark(img, "CheckinMe", opacity=-1.0)
        assert np.array_equal(result, img)

    def test_small_image(self):
        img = _img(h=50, w=80)
        result = apply_watermark(img, "CheckinMe", opacity=0.35)
        assert result.shape == img.shape

    def test_custom_text(self):
        img = _img()
        result = apply_watermark(img, "CONFIDENTIAL", opacity=0.5)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)
