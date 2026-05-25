import numpy as np
import pytest
from app.pipeline.normalize import normalize_resolution


def _img(h, w):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestNormalizeResolution:
    def test_output_exact_target_size(self):
        result = normalize_resolution(_img(300, 476), 1024, 640)
        assert result.shape == (640, 1024, 3)

    def test_already_correct_size_returns_same_array(self):
        img = _img(640, 1024)
        result = normalize_resolution(img, 1024, 640)
        assert result is img

    def test_upscale_produces_correct_size(self):
        result = normalize_resolution(_img(100, 160), 1024, 640)
        assert result.shape == (640, 1024, 3)

    def test_downscale_produces_correct_size(self):
        result = normalize_resolution(_img(2000, 3000), 1024, 640)
        assert result.shape == (640, 1024, 3)

    def test_output_is_uint8(self):
        result = normalize_resolution(_img(300, 476), 1024, 640)
        assert result.dtype == np.uint8

    def test_non_standard_target_size(self):
        result = normalize_resolution(_img(200, 300), 800, 500)
        assert result.shape == (500, 800, 3)

    def test_single_pixel_image(self):
        result = normalize_resolution(_img(1, 1), 1024, 640)
        assert result.shape == (640, 1024, 3)
