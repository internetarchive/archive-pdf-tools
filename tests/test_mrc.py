"""Unit tests for MRC decomposition components using real test data.

Requires test data files in test-files/ (provided separately).
"""

import pytest
import numpy as np
from PIL import Image
from internetarchivepdf.mrc import (
    threshold_image,
    create_threshold_mask,
    create_hocr_mask,
    partial_blur,
    partial_boxblur,
    encode_mrc_mask,
    estimate_noise,
)


class TestThresholdImage:
    def test_dpi_affects_window_size(self, test_img_sim_english):
        """Higher DPI increases Sauvola window size, affecting the result."""
        img = np.array(test_img_sim_english.convert('L'))
        result_low = threshold_image(img, dpi=72)
        result_high = threshold_image(img, dpi=600)
        assert result_low.shape == result_high.shape


class TestCreateThresholdMask:
    def test_text_area_detected(self, test_img_sim_english):
        """Text regions should appear in the threshold mask."""
        img = np.array(test_img_sim_english.convert('L'), dtype=np.float32)
        mask = np.zeros(img.shape, dtype=bool)
        create_threshold_mask(mask, img, dpi=300)
        assert mask.any(), "Expected some pixels in the mask for an image with text"


class TestCreateHocrMask:
    def test_empty_hocr_no_mask(self, test_img_sim_english):
        """Without hOCR data, no mask should be generated from hOCR."""
        img = test_img_sim_english.convert('L')
        mask = np.zeros(img.size[::-1], dtype=bool)
        create_hocr_mask(img, mask, [])
        assert not mask.any()

    def test_hocr_creates_mask_in_text_regions(self, test_img_sim_english, test_hocr_data_sim_english):
        """hOCR word bounding boxes should produce mask entries in those regions."""
        from hocr.parse import hocr_page_to_word_data, hocr_page_iterator
        import io

        page = next(hocr_page_iterator(io.StringIO(test_hocr_data_sim_english)))
        word_data = hocr_page_to_word_data(page)

        img = test_img_sim_english.convert('L')
        mask = np.zeros(img.size[::-1], dtype=bool)
        create_hocr_mask(img, mask, word_data, dpi=300)
        assert mask.any(), "Expected mask pixels from hOCR data"


class TestEstimateNoise:
    def test_constant_less_noisy_than_random(self):
        """A constant image should have lower estimated noise than a random image."""
        constant = np.full((100, 100), 128, dtype=np.float32)
        rng = np.random.default_rng(42)
        random_img = rng.integers(0, 256, (100, 100)).astype(np.float32)
        assert estimate_noise(constant) < estimate_noise(random_img)


class TestPartialBlur:
    def test_preserves_masked_pixels(self, test_img_sim_english):
        """Blur should not modify pixels in masked regions."""
        img = np.array(test_img_sim_english.convert('L'))
        mask = np.zeros(img.shape, dtype=bool)
        mask[10:20, 10:20] = True
        result = partial_blur(mask, img, sigma=5)
        assert np.array_equal(result[mask], img[mask])

    def test_blurs_unmasked_pixels(self, test_img_sim_english):
        """Unmasked pixels should be modified by the blur."""
        img = np.array(test_img_sim_english.convert('L'))
        mask = np.zeros(img.shape, dtype=bool)
        mask[10:20, 10:20] = True
        result = partial_blur(mask, img, sigma=5)
        assert not np.array_equal(result[~mask], img[~mask])


class TestPartialBoxblur:
    def test_preserves_masked_pixels(self, test_img_sim_english):
        """Box blur should not modify pixels in masked regions."""
        img = np.array(test_img_sim_english.convert('L'))
        mask = np.zeros(img.shape, dtype=bool)
        mask[10:20, 10:20] = True
        result = partial_boxblur(mask, img, size=5)
        assert np.array_equal(result[mask], img[mask])

    def test_blurs_unmasked_pixels(self, test_img_sim_english):
        """Unmasked pixels should be modified by the box blur."""
        img = np.array(test_img_sim_english.convert('L'))
        mask = np.zeros(img.shape, dtype=bool)
        mask[10:20, 10:20] = True
        result = partial_boxblur(mask, img, size=5)
        assert not np.array_equal(result[~mask], img[~mask])


class TestEncodeMrcMask:
    def test_encode_jbig2_skip_if_not_available(self, test_mask_array, tmp_path):
        """JBIG2 encoding should skip gracefully if binary is not found."""
        try:
            jbig2_path, png_path = encode_mrc_mask(test_mask_array, tmp_dir=str(tmp_path), jbig2=True)
        except FileNotFoundError:
            pytest.skip("jbig2 binary not found")
        assert jbig2_path is not None
        assert png_path is not None
