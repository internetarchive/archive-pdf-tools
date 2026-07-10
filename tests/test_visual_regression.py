"""Visual regression tests comparing compressed output quality against originals.

Uses real test images from test-files/ and compares SSIM/PSNR between the
original and the MRC-compressed reconstruction. Also tracks compression ratios.

Requires test data files in test-files/ (provided separately).
"""

import os
import pytest
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from internetarchivepdf.mrc import create_mrc_hocr_components, encode_mrc_images
from internetarchivepdf.const import (
    JPEG2000_IMPL_PILLOW, COMPRESSOR_JPEG2000, DENOISE_FAST,
)


def compute_metrics(original, compressed):
    """Compute SSIM and PSNR between two images, handling size mismatches."""
    if original.shape != compressed.shape:
        compressed = np.array(Image.fromarray(compressed).resize(
            (original.shape[1], original.shape[0]), Image.LANCZOS))
    min_dim = min(original.shape[0], original.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        return 1.0, 100.0
    channel_axis = -1 if original.ndim == 3 else None
    ssim_val = ssim(original, compressed, data_range=255, win_size=win_size, channel_axis=channel_axis)
    psnr_val = psnr(original, compressed, data_range=255)
    return ssim_val, psnr_val


class TestVisualRegression:
    def _crop_center(self, img, size=500):
        """Crop a square from the center of an image."""
        w, h = img.size
        left = (w - size) // 2
        top = (h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def test_grayscale_high_vs_low_quality(self, test_img_sim_english, tmp_path):
        """Higher quality settings should yield better SSIM and PSNR."""
        # Use a smaller crop for faster testing
        img = self._crop_center(test_img_sim_english, 800)
        img_np = np.array(img.convert('L'))
        img_h, img_w = img_np.shape[:2]

        gen_hq = create_mrc_hocr_components(img, [], dpi=300,
                                            bg_downsample=1, fg_downsample=1,
                                            denoise_mask=DENOISE_FAST)
        gen_lq = create_mrc_hocr_components(img, [], dpi=300,
                                            bg_downsample=2, fg_downsample=2,
                                            denoise_mask=DENOISE_FAST)

        mask_hq, bg_hq, bg_s, fg_hq, fg_s = encode_mrc_images(
            gen_hq,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[100]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[100]'],
            tmp_dir=str(tmp_path), jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            mrc_image_format=COMPRESSOR_JPEG2000,
        )

        mask_lq, bg_lq, bg_s, fg_lq, fg_s = encode_mrc_images(
            gen_lq,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[1000]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[1000]'],
            tmp_dir=str(tmp_path), jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            mrc_image_format=COMPRESSOR_JPEG2000,
        )

        # Reconstruct images from components (resize to match original dimensions)
        bg_hq_img = np.array(Image.open(bg_hq).convert('L').resize((img_w, img_h), Image.LANCZOS))
        fg_hq_img = np.array(Image.open(fg_hq).convert('L').resize((img_w, img_h), Image.LANCZOS))
        mask_hq_img = np.array(Image.open(mask_hq).convert('L').resize((img_w, img_h), Image.LANCZOS)) > 128

        bg_lq_img = np.array(Image.open(bg_lq).convert('L').resize((img_w, img_h), Image.LANCZOS))
        fg_lq_img = np.array(Image.open(fg_lq).convert('L').resize((img_w, img_h), Image.LANCZOS))
        mask_lq_img = np.array(Image.open(mask_lq).convert('L').resize((img_w, img_h), Image.LANCZOS)) > 128

        recon_hq = np.where(mask_hq_img, fg_hq_img, bg_hq_img)
        recon_lq = np.where(mask_lq_img, fg_lq_img, bg_lq_img)

        ssim_hq, psnr_hq = compute_metrics(img_np, recon_hq)
        ssim_lq, psnr_lq = compute_metrics(img_np, recon_lq)

        assert ssim_hq >= ssim_lq, f"Expected SSIM_HQ ({ssim_hq}) >= SSIM_LQ ({ssim_lq})"
        assert psnr_hq >= psnr_lq, f"Expected PSNR_HQ ({psnr_hq}) >= PSNR_LQ ({psnr_lq})"

    def test_rgb_quality_tradeoff(self, test_img_alienate, tmp_path):
        """Higher quality settings should produce larger files and better metrics."""
        # Use a smaller crop for faster testing
        img = self._crop_center(test_img_alienate, 800)
        img_np = np.array(img.convert('RGB'))
        img_h, img_w = img_np.shape[:2]

        gen_hq = create_mrc_hocr_components(img, [], dpi=300,
                                            bg_downsample=1, fg_downsample=1,
                                            denoise_mask=DENOISE_FAST)
        gen_lq = create_mrc_hocr_components(img, [], dpi=300,
                                            bg_downsample=2, fg_downsample=2,
                                            denoise_mask=DENOISE_FAST)

        mask_hq, bg_hq, bg_s, fg_hq, fg_s = encode_mrc_images(
            gen_hq,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[100]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[100]'],
            tmp_dir=str(tmp_path), jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            mrc_image_format=COMPRESSOR_JPEG2000,
        )

        mask_lq, bg_lq, bg_s, fg_lq, fg_s = encode_mrc_images(
            gen_lq,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[1000]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[1000]'],
            tmp_dir=str(tmp_path), jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            mrc_image_format=COMPRESSOR_JPEG2000,
        )

        size_hq = os.path.getsize(bg_hq) + os.path.getsize(fg_hq) + os.path.getsize(mask_hq)
        size_lq = os.path.getsize(bg_lq) + os.path.getsize(fg_lq) + os.path.getsize(mask_lq)

        assert size_hq > size_lq, "Higher quality should produce larger files"

        bg_hq_img = np.array(Image.open(bg_hq).convert('RGB').resize((img_w, img_h), Image.LANCZOS))
        fg_hq_img = np.array(Image.open(fg_hq).convert('RGB').resize((img_w, img_h), Image.LANCZOS))
        mask_hq_img = np.array(Image.open(mask_hq).convert('L').resize((img_w, img_h), Image.LANCZOS)) > 128

        bg_lq_img = np.array(Image.open(bg_lq).convert('RGB').resize((img_w, img_h), Image.LANCZOS))
        fg_lq_img = np.array(Image.open(fg_lq).convert('RGB').resize((img_w, img_h), Image.LANCZOS))
        mask_lq_img = np.array(Image.open(mask_lq).convert('L').resize((img_w, img_h), Image.LANCZOS)) > 128

        recon_hq = np.where(mask_hq_img[:, :, None], fg_hq_img, bg_hq_img)
        recon_lq = np.where(mask_lq_img[:, :, None], fg_lq_img, bg_lq_img)

        ssim_hq, psnr_hq = compute_metrics(img_np, recon_hq)
        ssim_lq, psnr_lq = compute_metrics(img_np, recon_lq)

        assert ssim_hq >= ssim_lq, f"Expected SSIM_HQ ({ssim_hq}) >= SSIM_LQ ({ssim_lq})"
        assert psnr_hq >= psnr_lq, f"Expected PSNR_HQ ({psnr_hq}) >= PSNR_LQ ({psnr_lq})"

    def test_compression_ratio_tracking(self, test_img_alienate, tmp_path):
        """Compression ratio should be tracked and reported correctly."""
        # Use a smaller crop for faster testing
        img = self._crop_center(test_img_alienate, 800)
        img_np = np.array(img.convert('RGB'))

        gen = create_mrc_hocr_components(img, [], dpi=300,
                                         bg_downsample=2, fg_downsample=1,
                                         denoise_mask=DENOISE_FAST)
        mask, bg, bg_s, fg, fg_s = encode_mrc_images(
            gen,
            bg_compression_flags=['quality_mode:"rates";quality_layers:[500]'],
            fg_compression_flags=['quality_mode:"rates";quality_layers:[750]'],
            tmp_dir=str(tmp_path), jbig2=False,
            jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
            mrc_image_format=COMPRESSOR_JPEG2000,
        )

        raw_size = img_np.nbytes
        compressed_size = os.path.getsize(bg) + os.path.getsize(fg) + os.path.getsize(mask)
        assert compressed_size < raw_size, "Compressed size should be smaller than raw pixel data"
