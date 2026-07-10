"""Integration tests for the full recode pipeline using real test data.

Requires test data files in test-files/ (provided separately).
"""

import os
import pytest
import fitz
import numpy as np

from internetarchivepdf.recode import recode
from internetarchivepdf.const import (
    IMAGE_MODE_MRC, JPEG2000_IMPL_PILLOW, COMPRESSOR_JPEG2000,
    DENOISE_FAST,
)


class TestRecodePipeline:
    def test_recode_grayscale_image(self, test_img_sim_english, test_hocr_data_sim_english, tmp_path):
        """Full recode pipeline with a grayscale image should produce a valid PDF."""
        img_path = str(tmp_path / 'input.png')
        test_img_sim_english.save(img_path)

        hocr_path = str(tmp_path / 'test.html')
        with open(hocr_path, 'w') as f:
            f.write(test_hocr_data_sim_english)

        out_path = str(tmp_path / 'output.pdf')

        result = recode(
            from_imagestack=img_path,
            hocr_file=hocr_path,
            out_pdf=out_path,
            dpi=400,
            image_mode=IMAGE_MODE_MRC,
            jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[500]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[750]'],
            mrc_image_format=COMPRESSOR_JPEG2000,
            bg_downsample=2,
            fg_downsample=1,
            denoise_mask=DENOISE_FAST,
            verbose=False,
            debug=False,
        )

        assert os.path.exists(out_path)
        assert result['compression_ratio'] > 1.0

        doc = fitz.open(out_path)
        assert doc.page_count == 1
        doc.close()

    def test_recode_rgb_image(self, test_img_alienate, test_hocr_data_alienate_page1, tmp_path):
        """Full recode pipeline with an RGB image should produce a valid PDF."""
        img_path = str(tmp_path / 'input_rgb.png')
        test_img_alienate.save(img_path)

        hocr_path = str(tmp_path / 'test_rgb.html')
        with open(hocr_path, 'w') as f:
            f.write(test_hocr_data_alienate_page1)

        out_path = str(tmp_path / 'output_rgb.pdf')

        result = recode(
            from_imagestack=img_path,
            hocr_file=hocr_path,
            out_pdf=out_path,
            dpi=300,
            image_mode=IMAGE_MODE_MRC,
            jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[500]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[750]'],
            mrc_image_format=COMPRESSOR_JPEG2000,
            bg_downsample=2,
            fg_downsample=1,
            denoise_mask=DENOISE_FAST,
            verbose=False,
            debug=False,
        )

        assert os.path.exists(out_path)
        assert result['compression_ratio'] > 1.0

        doc = fitz.open(out_path)
        assert doc.page_count == 1
        doc.close()

    def test_recode_deterministic_output(self, test_img_sim_english, test_hocr_data_sim_english, tmp_path):
        """Running the same input twice should produce identical PDFs (modulo metadata)."""
        img_path = str(tmp_path / 'input_det.png')
        test_img_sim_english.save(img_path)

        hocr_path = str(tmp_path / 'test_det.html')
        with open(hocr_path, 'w') as f:
            f.write(test_hocr_data_sim_english)

        out1 = str(tmp_path / 'output_det1.pdf')
        out2 = str(tmp_path / 'output_det2.pdf')

        kwargs = dict(
            from_imagestack=img_path,
            hocr_file=hocr_path,
            dpi=400,
            image_mode=IMAGE_MODE_MRC,
            jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[500]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[750]'],
            mrc_image_format=COMPRESSOR_JPEG2000,
            bg_downsample=2,
            fg_downsample=1,
            denoise_mask=DENOISE_FAST,
            verbose=False,
            debug=False,
        )

        recode(out_pdf=out1, **kwargs)
        recode(out_pdf=out2, **kwargs)

        doc1 = fitz.open(out1)
        doc2 = fitz.open(out2)
        assert doc1.page_count == doc2.page_count
        for page_idx in range(doc1.page_count):
            pix1 = doc1[page_idx].get_pixmap()
            pix2 = doc2[page_idx].get_pixmap()
            assert pix1.width == pix2.width
            assert pix1.height == pix2.height
            img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, -1)
            img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, -1)
            assert np.array_equal(img1, img2), f"Page {page_idx} pixel data differs"
        doc1.close()
        doc2.close()

    def test_recode_with_jbig2(self, test_img_sim_english, test_hocr_data_sim_english, tmp_path):
        """Full recode pipeline with JBIG2 mask compression."""
        img_path = str(tmp_path / 'input_jbig2.png')
        test_img_sim_english.save(img_path)

        hocr_path = str(tmp_path / 'test_jbig2.html')
        with open(hocr_path, 'w') as f:
            f.write(test_hocr_data_sim_english)

        out_path = str(tmp_path / 'output_jbig2.pdf')

        try:
            result = recode(
                from_imagestack=img_path,
                hocr_file=hocr_path,
                out_pdf=out_path,
                dpi=400,
                image_mode=IMAGE_MODE_MRC,
                jbig2=True,
                jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
                bg_compression_flags=['quality_mode:"rates";quality_layers:[500]'],
                fg_compression_flags=['quality_mode:"rates";quality_layers:[750]'],
                mrc_image_format=COMPRESSOR_JPEG2000,
                bg_downsample=2,
                fg_downsample=1,
                denoise_mask=DENOISE_FAST,
                verbose=False,
                debug=False,
            )
        except FileNotFoundError:
            pytest.skip("jbig2 binary not found")

        assert os.path.exists(out_path)
        assert result['compression_ratio'] > 1.0
        doc = fitz.open(out_path)
        assert doc.page_count == 1
        doc.close()
