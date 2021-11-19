# archive-pdf-tools
# Copyright (C) 2020-2021, Internet Archive
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Merlijn Boris Wolf Wajer <merlijn@archive.org>

import sys
from os import close, remove

from glob import glob
from tempfile import mkstemp
import subprocess
from time import time

import warnings

from PIL import Image, ImageOps
from skimage.filters import threshold_local, threshold_otsu
from skimage.restoration import denoise_tv_bregman, estimate_sigma

from scipy import ndimage
import numpy as np

from optimiser import optimise_gray, optimise_rgb, optimise_gray2, optimise_rgb2, fast_mask_denoise
from sauvola import binarise_sauvola

import fitz

fitz.TOOLS.set_icc(True) # For good measure, not required

from internetarchivepdf.const import (RECODE_RUNTIME_WARNING_TOO_SMALL_TO_DOWNSAMPLE, JPEG2000_IMPL_KAKADU,
        JPEG2000_IMPL_OPENJPEG, JPEG2000_IMPL_GROK, JPEG2000_IMPL_PILLOW,
        COMPRESSOR_JPEG, COMPRESSOR_JPEG2000, DENOISE_NONE, DENOISE_FAST,
        DENOISE_BREGMAN)


"""
"""

KDU_COMPRESS = 'kdu_compress'
KDU_EXPAND = 'kdu_expand'
OPJ_COMPRESS = 'opj_compress'
OPJ_DECOMPRESS = 'opj_decompress'
GRK_COMPRESS = 'grk_compress'
GRK_DECOMPRESS = 'grk_decompress'


# skimage throws useless UserWarnings in various functions
def mean_estimate_sigma(arr):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.mean(estimate_sigma(arr))


def threshold_image(img, dpi, k=0.34):
    window_size = 51

    if dpi is not None:
        window_size = int(dpi / 4)
        if window_size % 2 == 0:
            window_size += 1

    h, w = img.shape
    out_img = np.ndarray(img.shape, dtype=np.bool)
    out_img = np.reshape(out_img, w*h)
    in_img = np.reshape(img, w*h)

    binarise_sauvola(in_img, out_img, w, h, window_size, window_size, k, 128)
    out_img = np.reshape(out_img, (h, w))
    # TODO: optimise this, we can do it in binarise_sauvola
    out_img = np.invert(out_img)

    return out_img


def denoise_bregman(binary_img):
    thresf = np.array(binary_img, dtype=np.float32)
    #denoise = denoise_tv_bregman(thresf, weight=0.25)
    denoise = denoise_tv_bregman(thresf, weight=1.)

    #denoise = denoise > 0.6
    denoise = denoise > 0.4  # XXX: 0.4?
    denoise = np.array(denoise, dtype=np.bool)

    return denoise

# TODO: Rename, can be either foreground or background
def partial_blur(mask, img, sigma=5, mode=None):
    """
    Blur a part of the image 'img', where mask = 0.
    The actual values used by the blur are colours where mask = '1', effectively
    'erasing/blurring' parts of an image where mask = 0 with colours where mask = 1.

    At the end, restore all pixels from img where mask = 1.
    """
    maskf = np.array(mask, dtype=np.float32)

    if mode == 'RGB' or mode == 'RGBA':
        in_r = img[:, :, 0] * maskf
        in_g = img[:, :, 1] * maskf
        in_b = img[:, :, 2] * maskf
        filter_r = ndimage.filters.gaussian_filter(in_r, sigma = sigma)
        filter_g = ndimage.filters.gaussian_filter(in_g, sigma = sigma)
        filter_b = ndimage.filters.gaussian_filter(in_b, sigma = sigma)
    else:
        imgf = np.copy(img)
        imgf = np.array(imgf, dtype=np.float32)
        filter = ndimage.filters.gaussian_filter(imgf * maskf, sigma = sigma)

    weights = ndimage.filters.gaussian_filter(maskf, sigma = sigma)

    if mode == 'RGB' or mode == 'RGBA':
        filter_r /= weights + 0.00001
        filter_g /= weights + 0.00001
        filter_b /= weights + 0.00001

        newimg = np.copy(img)
        newimg[:, :, 0] = filter_r
        newimg[:, :, 1] = filter_g
        newimg[:, :, 2] = filter_b
    else:
        filter /= weights + 0.00001
        newimg = np.array(filter, dtype=np.uint8)

    newimg[mask] = img[mask]

    return newimg


def partial_boxblur(mask, fg, size=5, mode=None):
    maskf = np.array(mask, dtype=np.float32)

    if mode == 'RGB' or mode == 'RGBA':
        in_r = fg[:, :, 0] * maskf
        in_g = fg[:, :, 1] * maskf
        in_b = fg[:, :, 2] * maskf
        filter_r = ndimage.uniform_filter(in_r, size = size)
        filter_g = ndimage.uniform_filter(in_g, size = size)
        filter_b = ndimage.uniform_filter(in_b, size = size)
    else:
        fgf = np.copy(fg)
        fgf = np.array(fgf, dtype=np.float32)
        filter = ndimage.uniform_filter(fgf * maskf, size = size)

    weights = ndimage.uniform_filter(maskf, size = size)

    if mode == 'RGB' or mode == 'RGBA':
        filter_r /= weights + 0.00001
        filter_g /= weights + 0.00001
        filter_b /= weights + 0.00001

        newfg = np.copy(fg)
        newfg[:, :, 0] = filter_r
        newfg[:, :, 1] = filter_g
        newfg[:, :, 2] = filter_b
    else:
        filter /= weights + 0.00001
        newfg = np.array(filter, dtype=np.uint8)

    newfg[mask] = fg[mask]

    return newfg


def create_hocr_mask(img, mask_arr, hocr_word_data, downsample=None, dpi=None, timing_data=None):
    image_width, image_height = img.size
    np_img = np.array(img)

    t = time()

    for paragraph in hocr_word_data:
        for line in paragraph['lines']:
            coords = line['bbox']

            line_text = ' '.join([word['text'] for word in line['words']])
            line_confs = [word['confidence'] for word in line['words']]
            line_conf = sum(line_confs) / len(line_confs)

            if line_text.strip() == '' or line_conf < 20:
                continue

            if downsample is not None:
                coords = [int(x/downsample) for x in coords]
            else:
                coords = [int(x) for x in coords]

            left, top, right, bottom = coords
            # This can happen if we downsample and round to int
            if left == right or top == bottom:
                continue

            if (left >= right) or (top >= bottom):
                print('Invalid bounding box: (%d, %d, %d, %d)' % (left, top, right, bottom), file=sys.stderr)
                continue

            if (left < 0) or (right > image_width) or (top < 0) or (bottom > image_height):
                print('Invalid bounding box outside image: (%d, %d, %d, %d)' % (left, top, right, bottom), file=sys.stderr)
                continue

            np_lineimg = np_img[top:bottom,left:right]
            # Simple grayscale invert
            np_lineimg_invert = 255 - np.copy(np_lineimg)

            # XXX: If you tweak k, you must tweak the various ratio and sigma's
            # based on the test images
            k = 0.1
            thres = threshold_image(np_lineimg, dpi, k)
            ones = np.count_nonzero(thres)
            zero = (img.size[0] * img.size[1]) - ones
            ratio = (ones/(zero+ones))*100

            thres_invert = threshold_image(np_lineimg_invert, dpi, k)
            ones_i = np.count_nonzero(thres_invert)
            zero_i = (img.size[0] * img.size[1]) - ones
            inv_ratio = (ones_i/(zero_i+ones_i))*100

            if ratio < 0.3 or inv_ratio < 0.3:
                th = None

                perc_larger = 0.
                if inv_ratio != 0.0:
                    perc_larger = (ratio / inv_ratio) * 100

                if inv_ratio > 0.2 and ratio < 0.2:
                    th = thres
                else:
                    # mean_estimate_sigma is expensive, so let's only do it if
                    # we need to

                    ratio_sigma = mean_estimate_sigma(thres)
                    inv_ratio_sigma = mean_estimate_sigma(thres_invert)


                    # Prefer ratio over inv_ratio by a bit
                    if inv_ratio < 0.3 and inv_ratio < ratio and \
                    (inv_ratio_sigma < ratio_sigma or \
                    (ratio_sigma < 0.1 and inv_ratio_sigma < 0.1)):
                        th = thres_invert
                    elif ratio < 0.2:
                        th = thres

                if th is not None:
                    mask_arr[top:bottom, left:right] = th


    if timing_data is not None:
        timing_data.append(('hocr_mask_gen', time() - t))


def estimate_noise(imgf):
    #sigma_est = mean_estimate_sigma(imgf)
    #return sigma_est

    # We do this only on a part of the image, because it's accurate enough wrt
    # noise estimation (definitely for camera noise estimation since that's
    # everywhere in the image, and it's quite a bit faster this way).
    h, w = imgf.shape
    MUL = 4
    hs = int(h/2 - h/MUL)
    he = int(h/2 + h/MUL)
    ws = int(w/2 - w/MUL)
    we = int(w/2 + w/MUL)

    # Really small image?
    if he == 0 or we == 0:
        hs = 0
        he = h
        ws = 0
        we = w

    sigma_est = mean_estimate_sigma(imgf[hs:he, ws:we])

    return sigma_est



def create_threshold_mask(mask_arr, imgf, dpi=None, denoise_mask=None, timing_data=None):
    # We don't apply any of these blurs to the hOCR mask, we want that as
    # sharp as possible.

    t = time()
    sigma_est = estimate_noise(imgf)

    if timing_data is not None:
        timing_data.append(('est_1', time() - t))
    if sigma_est > 1.0:
        t = time()
        imgf = ndimage.filters.gaussian_filter(imgf, sigma=sigma_est*0.1)
        if timing_data is not None:
            timing_data.append(('blur_1', time() - t))

        #t = time()
        #n_sigma_est = mean_estimate_sigma(imgf)
        #time_data.append(('est_2', time() - t))
        #if sigma_est > 1.0 and n_sigma_est > 1.0:
            #    t = time()
        #    imgf = ndimage.filters.gaussian_filter(imgf, sigma=sigma_est*0.5)
        #    print('Going for second blur: n_sigma_est:',n_sigma_est)
        #    time_data.append(('blur_2', time() - t))

    t = time()
    thres_arr = threshold_image(imgf.astype(np.uint8), dpi)
    if timing_data is not None:
        timing_data.append(('threshold', time() - t))

    mask_arr |= thres_arr


# TODO: Reduce amount of memory active at one given point (keep less images in
# memory, write to disk sooner, etc), careful with numpy <-> PIL conversions
def create_mrc_hocr_components(image, hocr_word_data,
                               dpi=None,
                               downsample=None,
                               bg_downsample=None,
                               fg_downsample=None,
                               denoise_mask=None, timing_data=None,
                               errors=None):
    """
    Create the MRC components: mask, foreground and background

    Args:

    * image (PIL.Image): Image to be decomposed
    * hocr_word_data: OCR data about found text on the page
    * downsample (int): factor by which the OCR data is to be downsampled
    * bg_downsample (int): if the background image should be downscaled
    * denoise_mask (bool): Whether to denoise the image if it is deemed too
      noisy
    * timing_data: Optional timing data to log individual timing data to.
    * errors: Optional argument (of type set) with encountered runtime errors

    Returns a tuple of the components, as numpy arrays: (mask, foreground,
    background)
    """
    grayimg = image
    if image.mode != 'L':
        t = time()
        grayimg = image.convert('L')
        if timing_data is not None:
            timing_data.append(('grey_conversion', time() - t))

    width_, height_ = image.size

    mask_arr = np.array(Image.new('1', image.size))

    # Modifies mask_arr in place
    create_hocr_mask(grayimg, mask_arr, hocr_word_data, downsample=downsample,
                     dpi=dpi, timing_data=timing_data)
    grayimgf = np.array(grayimg, dtype=np.float32)

    MIX_THRESHOLD = True
    if MIX_THRESHOLD:
        # XXX: this nukes the hocr threshold, testing only
        # mask_arr = np.zeros(mask_arr.shape, dtype=np.bool)

        # Modifies mask_arr in place
        create_threshold_mask(mask_arr, grayimgf, dpi=dpi,
                              denoise_mask=denoise_mask,
                              timing_data=timing_data)

    if denoise_mask != DENOISE_NONE:
        t = time()
        if denoise_mask == DENOISE_FAST:
            # XXX: We could make the mincnt parameter take the dpi into account
            fast_mask_denoise(mask_arr, width_, height_, 4, 2)
            if timing_data is not None:
                timing_data.append(('fast_denoise', time() - t))
        elif denoise_mask == DENOISE_BREGMAN:
            mask_arr = denoise_bregman(mask_arr)
            if timing_data is not None:
                timing_data.append(('denoise', time() - t))
        else:
            raise ValueError('Invalid denoise option:', denoise_mask)


    yield mask_arr

    image_arr = np.array(image)

    t = time()
    # Take foreground pixels and optimise the image by making the surrounding
    # pixels like the foreground, allowing for more optimal compression (and
    # higher quality foreground pixels as a result)
    if image.mode == 'L':
        foreground_arr = optimise_gray2(mask_arr, image_arr, width_, height_, 3)
    else:
        foreground_arr = optimise_rgb2(mask_arr, image_arr, width_, height_, 3)
    if timing_data is not None:
        # The name fg_partial_blur is kept for backwards compatibility
        timing_data.append(('fg_partial_blur', time() - t))

    if fg_downsample is not None:
        t = time()
        image2 = Image.fromarray(foreground_arr)
        w, h = image2.size
        w_downsample = int(w / fg_downsample)
        h_downsample = int(h / fg_downsample)
        if w_downsample > 0 and h_downsample > 0:
            image2.thumbnail((w_downsample, h_downsample))
            foreground_arr = np.array(image2)
        else:
            if errors is not None:
                errors.add(RECODE_RUNTIME_WARNING_TOO_SMALL_TO_DOWNSAMPLE)

        if timing_data is not None:
            timing_data.append(('fg_downsample', time() - t))

    yield foreground_arr
    foreground_arr = None

    mask_inv = mask_arr ^ np.ones(mask_arr.shape, dtype=bool)

    t = time()
    # Take background pixels and optimise the image by placing them where the
    # foreground pixels are thought to be, this has the effect of reducing
    # compression artifacts (thus improving quality) and at the same time making
    # the image easier to compress (smaller file size)
    if image.mode == 'L':
        background_arr = optimise_gray2(mask_inv, image_arr, width_, height_, 10)
    else:
        background_arr = optimise_rgb2(mask_inv, image_arr, width_, height_, 10)
    if timing_data is not None:
        # The name bg_partial_blur is kept for backwards compatibility
        timing_data.append(('bg_partial_blur', time() - t))

    if bg_downsample is not None:
        t = time()
        image2 = Image.fromarray(background_arr)
        w, h = image2.size
        w_downsample = int(w / bg_downsample)
        h_downsample = int(h / bg_downsample)
        if w_downsample > 0 and h_downsample > 0:
            image2.thumbnail((w_downsample, h_downsample))
            background_arr = np.array(image2)
        else:
            if errors is not None:
                errors.add(RECODE_RUNTIME_WARNING_TOO_SMALL_TO_DOWNSAMPLE)

        if timing_data is not None:
            timing_data.append(('bg_downsample', time() - t))

    yield background_arr
    return


def encode_mrc_mask(np_mask, tmp_dir=None, jbig2=True, embedded_jbig2=False,
                    timing_data=None):
    """
    Encode mask image either to JBIG2 or PNG.

    Args:

    * np_mask (numpy.array): Mask image array
    * tmp_dir (str): path the temporary directory to write images to
    * jbig2 (bool): Whether to encode to JBIG2 or PNG
    * embedded_jbig2 (bool): Whether to encode to JBIG2 with or without header
    * timing_data (optional): Add time information to timing_data structure

    Returns a tuple: (str, str) where the first entry is the jbig2
    path, if any, the second is the png path.
    """
    t = time()
    mask = Image.fromarray(np_mask)

    fd, mask_img_png = mkstemp(prefix='mask', suffix='.png', dir=tmp_dir)
    close(fd)
    if jbig2:
        fd, mask_img_jbig2 = mkstemp(prefix='mask', suffix='.jbig2', dir=tmp_dir)
        close(fd)

    mask.save(mask_img_png, compress_level=0)

    if jbig2:
        if embedded_jbig2:
            out = subprocess.check_output(['jbig2', '-p', mask_img_png])
        else:
            out = subprocess.check_output(['jbig2', mask_img_png])
        fp= open(mask_img_jbig2, 'wb+')
        fp.write(out)
        fp.close()

    if timing_data is not None:
        timing_data.append(('mask_jbig2', time()-t))

    if jbig2:
        return mask_img_jbig2, mask_img_png
    else:
        return None, mask_img_png


def encode_mrc_background(np_bg, bg_compression_flags, tmp_dir=None,
        jpeg2000_implementation=None, mrc_image_format=None, timing_data=None):
    """
    Encode background image as JPEG2000, with the provided compression settings
    and JPEG2000 encoder.

    Args:

    * np_bg (numpy.array): Background image array
    * bg_compression_flags (str): Compression flags
    * tmp_dir (str): path the temporary directory to write images to
    * jpeg2000_implementation (str): What JPEG2000 implementation to use
    * mrc_image_format (str): What image format to produce
    * timing_data (optional): Add time information to timing_data structure

    Returns the filepath to the JPEG2000 background image
    """
    t = time()
    # Create background
    if mrc_image_format == COMPRESSOR_JPEG:
        fd, bg_img_tiff = mkstemp(prefix='bg', suffix='.jpg', dir=tmp_dir)
    else:
        if jpeg2000_implementation in (JPEG2000_IMPL_KAKADU, JPEG2000_IMPL_GROK):
            fd, bg_img_tiff = mkstemp(prefix='bg', suffix='.tiff', dir=tmp_dir)
        else:
            fd, bg_img_tiff = mkstemp(prefix='bg', suffix='.pnm', dir=tmp_dir)

    close(fd)
    fd, bg_img_jp2 = mkstemp(prefix='bg', suffix='.jp2', dir=tmp_dir)
    close(fd)
    remove(bg_img_jp2) # XXX: Kakadu doesn't want the file to exist, so what are
                       # we even doing

    bg_img = Image.fromarray(np_bg)
    if mrc_image_format == COMPRESSOR_JPEG:
        bg_img.save(bg_img_tiff, quality=100)
    else:
        bg_img.save(bg_img_tiff)

    if mrc_image_format == COMPRESSOR_JPEG:
        output = subprocess.check_output(['jpegoptim'] + bg_compression_flags +
                [bg_img_tiff, '--stdout'])
        tmpfd=open(bg_img_jp2, 'bw+') # XXX: FIXME: this defeats the point of a tmpfile
        tmpfd.write(output)
        tmpfd.close()
    elif jpeg2000_implementation == JPEG2000_IMPL_KAKADU:
        subprocess.check_call([KDU_COMPRESS,
            '-num_threads', '0',
            '-i', bg_img_tiff, '-o', bg_img_jp2] + bg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    elif jpeg2000_implementation == JPEG2000_IMPL_OPENJPEG:
        subprocess.check_call([OPJ_COMPRESS,
            '-i', bg_img_tiff, '-o', bg_img_jp2] + bg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    elif jpeg2000_implementation == JPEG2000_IMPL_GROK:
        subprocess.check_call([GRK_COMPRESS, '-H', '1',
            '-i', bg_img_tiff, '-o', bg_img_jp2] + bg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    else:
        raise Exception('Error: invalid jpeg2000 implementation?')

    remove(bg_img_tiff)

    if timing_data is not None:
        timing_data.append(('bg_jp2', time()-t))

    return bg_img_jp2


def encode_mrc_foreground(np_fg, fg_compression_flags, tmp_dir=None,
        jpeg2000_implementation=None, mrc_image_format=None, timing_data=None):
    """
    Encode foreground image as JPEG2000, with the provided compression settings
    and JPEG2000 encoder.

    Args:

    * np_bg (numpy.array): Foreground image array
    * fg_compression_flags (str): Compression flags
    * tmp_dir (str): path the temporary directory to write images to
    * jpeg2000_implementation (str): What JPEG2000 implementation to use
    * mrc_image_format (str): What image format to produce
    * timing_data (optional): Add time information to timing_data structure

    Returns the filepath to the JPEG2000 foreground image
    """
    t = time()
    # Create foreground
    if mrc_image_format == COMPRESSOR_JPEG:
        fd, fg_img_tiff = mkstemp(prefix='fg', suffix='.jpg', dir=tmp_dir)
    else:
        if jpeg2000_implementation in (JPEG2000_IMPL_KAKADU, JPEG2000_IMPL_GROK):
            fd, fg_img_tiff = mkstemp(prefix='fg', suffix='.tiff', dir=tmp_dir)
        else:
            fd, fg_img_tiff = mkstemp(prefix='fg', suffix='.pnm', dir=tmp_dir)

    close(fd)
    fd, fg_img_jp2 = mkstemp(prefix='fg', suffix='.jp2', dir=tmp_dir)
    close(fd)
    remove(fg_img_jp2) # XXX: Kakadu doesn't want the file to exist, so what are
                       # we even doing

    fg_img = Image.fromarray(np_fg)
    if mrc_image_format == COMPRESSOR_JPEG:
        fg_img.save(fg_img_tiff, quality=100)
    else:
        fg_img.save(fg_img_tiff)

    if mrc_image_format == COMPRESSOR_JPEG:
        output = subprocess.check_output(['jpegoptim'] + fg_compression_flags +
                [fg_img_tiff, '--stdout'])
        tmpfd=open(fg_img_jp2, 'bw+') # XXX: FIXME: this defeats the point of a tmpfile
        tmpfd.write(output)
        tmpfd.close()
    elif jpeg2000_implementation == JPEG2000_IMPL_KAKADU:
        subprocess.check_call([KDU_COMPRESS,
            '-num_threads', '0',
            '-i', fg_img_tiff, '-o', fg_img_jp2] + fg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    elif jpeg2000_implementation == JPEG2000_IMPL_OPENJPEG:
        subprocess.check_call([OPJ_COMPRESS,
            '-i', fg_img_tiff, '-o', fg_img_jp2] + fg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    elif jpeg2000_implementation == JPEG2000_IMPL_GROK:
        subprocess.check_call([GRK_COMPRESS, '-H', '1',
            '-i', fg_img_tiff, '-o', fg_img_jp2] + fg_compression_flags,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    else:
        raise Exception('Error: invalid jpeg2000 implementation?')

    remove(fg_img_tiff)

    if timing_data is not None:
        timing_data.append(('fg_jp2', time()-t))

    return fg_img_jp2


def encode_mrc_images(mrc_gen, bg_compression_flags=None, fg_compression_flags=None,
                      tmp_dir=None, jbig2=True, timing_data=None,
                      jpeg2000_implementation=None, mrc_image_format=None,
                      embedded_jbig2=False):
    mask_img_jbig2, mask_img_png = encode_mrc_mask(next(mrc_gen),
            tmp_dir=tmp_dir, jbig2=jbig2, embedded_jbig2=embedded_jbig2,
            timing_data=timing_data)

    np_fg = next(mrc_gen)
    fg_img_jp2 = encode_mrc_foreground(np_fg, fg_compression_flags, tmp_dir=tmp_dir,
                                       jpeg2000_implementation=jpeg2000_implementation,
                                       mrc_image_format=mrc_image_format,
                                       timing_data=timing_data)
    fg_h, fg_w = np_fg.shape[0:2]
    np_fg = None

    np_bg = next(mrc_gen)
    bg_img_jp2 = encode_mrc_background(np_bg, bg_compression_flags, tmp_dir=tmp_dir,
                                       jpeg2000_implementation=jpeg2000_implementation,
                                       mrc_image_format=mrc_image_format,
                                       timing_data=timing_data)
    bg_h, bg_w = np_bg.shape[0:2]
    np_bg = None

    # XXX: probably don't need this
    try:
        _ = next(mrc_gen)
    except StopIteration:
        pass

    if jbig2:
        remove(mask_img_png)

    if jbig2:
        return mask_img_jbig2, bg_img_jp2, (bg_w, bg_h), fg_img_jp2, (fg_w, fg_h)
    else:
        # Return PNG which mupdf will turn into ccitt with
        # save(..., deflate=True) until mupdf fixes their JBIG2 support
        #return mask_img_png, bg_img_jp2, fg_img_jp2
        return mask_img_png, bg_img_jp2, (bg_w, bg_h), fg_img_jp2, (fg_w, fg_h)
