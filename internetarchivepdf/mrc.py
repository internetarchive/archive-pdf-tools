# Author: Merlijn Boris Wolf Wajer <merlijn@archive.org>

from os import close, remove

from glob import glob
from tempfile import mkstemp
import subprocess
from time import time

from PIL import Image, ImageEnhance, ImageOps
from skimage.filters import threshold_local, threshold_otsu, threshold_sauvola
from skimage.restoration import denoise_tv_bregman, estimate_sigma

from scipy import ndimage
import numpy as np


import fitz

fitz.TOOLS.set_icc(True) # For good measure, not required


"""
TODO:

Improve the MRC algorithm, by improving the foreground detection (either
using ocropus binarisation or using a paper that both Hank and a friend of mine
ound: https://engineering.purdue.edu/~bouman/software/Text-Seg/tip30.pdf),
improve background compression using JPEG2000 ROI
(http://summit.sfu.ca/system/files/iritems1/2784/b36288305.pdf)
"""


KDU_COMPRESS = 'kdu_compress'
KDU_EXPAND = 'kdu_expand'


def invert_mask(mask):
    return mask ^ np.ones(mask.shape, dtype=bool)


def threshold_image(pil_image, rev=False, otsu=False, block_size=9):
    """
    Apply adaptive (local) thresholding, filtering out background noise to make
    the text more readable. Tesseract uses Otsu thresholding, which in our
    testing hasn't worked all that well, so we perform better thresholding
    before passing the image to tesseract.

    Returns the thresholded PIL image
    """
    img = np.array(pil_image)

    if otsu:
        try:
            binary_otsu = threshold_otsu(img)
        except ValueError:
            binary_otsu = np.ndarray(img.shape)
            binary_otsu[:] = 0

        if rev:
            binary_img = img > binary_otsu
        else:
            binary_img = img < binary_otsu
    else:
        #binary_local = threshold_local(img, block_size, method='gaussian')
        #binary_local = threshold_local(img, block_size, offset=10, method='gaussian')
        binary_local = threshold_local(img, block_size, method='gaussian')
        if not rev:
            binary_img = img < binary_local
        else:
            binary_img = img > binary_local

    return binary_img



def threshold_image2(pil_image):
    local = threshold_image(pil_image)
    otsu = threshold_image(pil_image, otsu=True)

    return local & otsu


def threshold_image3(img, rev=False):
    window_size = 51
    thres_sauvola = threshold_sauvola(img, window_size=window_size, k=0.3)
    if rev:
        binary_img = img > thres_sauvola
    else:
        binary_img = img < thres_sauvola

    return binary_img


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




def create_mrc_components(image):
    img = image
    if image.mode != 'L':
        img = image.convert('L')
    mask = threshold_image(img)
    #imask = inverse_mask(mask)

    mask_img = Image.fromarray(mask)

    np_im = np.array(image)

    np_bg = np.copy(np_im)
    np_fg = np.copy(np_im)

    # XXX: We likely don't want this to be the 'average'. We might want it (and
    # some neighbouring pixels!) to be 'background' colour, or something like
    # that.
    np_bg[mask] = np.average(np_im)

    # We might not want to touch these pixels, but let's set them to zero for
    # now for good measure.
    # np_fg[mask] = 0

    return mask, np_bg, np_fg


def create_mrc_hocr_components(image, hocr_word_data, bg_downsample=None,
                               denoise_mask=None):
    time_data = []

    img = image
    if image.mode != 'L':
        t = time()
        img = image.convert('L')
        time_data.append(('grey_conversion', time() - t))

    image_mask = Image.new('1', image.size)
    mask_arr = np.array(image_mask)

    MIX_THRESHOLD = True

    t = time()
    for paragraphs in hocr_word_data:
        for lines in paragraphs['lines']:
            for word in lines['words']:
                if not word['text'].strip():
                    continue

                top, left, bottom, right = [int(x) for x in word['bbox']]

                wordimg = img.crop(word['bbox'])
                thres = threshold_image2(wordimg)

                ones = np.count_nonzero(thres)
                zeroes = np.count_nonzero(np.invert(thres))

                # TODO: use multichannel here? I don't think that makes sense
                sigma_est = np.mean(estimate_sigma(thres))

                if sigma_est > 0.1:
                    # Invert. (TODO: we should do this in a more efficient
                    # manner)
                    thres_i = threshold_image2(ImageOps.invert(wordimg))

                    # TODO: use multichannel here? I don't think that makes sense
                    sigma_est_i = np.mean(estimate_sigma(thres_i))

                    if sigma_est < sigma_est_i:
                        pass
                    elif sigma_est_i < sigma_est:
                        thres = thres_i

                mask_arr[left:right, top:bottom] = thres

                # TODO: Set thres values in array_mask

                intbox = [int(x) for x in word['bbox']]

    time_data.append(('hocr_mask_gen', time() - t))

    image_arr = np.array(image)

    imgf = np.array(img, dtype=np.float32)

    if MIX_THRESHOLD:
        # We don't apply any of these blurs to the hOCR mask, we want that as
        # sharp as possible.

        t = time()
        sigma_est = np.mean(estimate_sigma(imgf))
        time_data.append(('est_1', time() - t))
        if sigma_est > 1.0:
            t = time()
            imgf = ndimage.filters.gaussian_filter(imgf, sigma=sigma_est*0.1)
            time_data.append(('blur_1', time() - t))

            #t = time()
            #n_sigma_est = np.mean(estimate_sigma(imgf))
            #time_data.append(('est_2', time() - t))
            #if sigma_est > 1.0 and n_sigma_est > 1.0:
            #    t = time()
            #    imgf = ndimage.filters.gaussian_filter(imgf, sigma=sigma_est*0.5)
            #    print('Going for second blur: n_sigma_est:',n_sigma_est)
            #    time_data.append(('blur_2', time() - t))

        t = time()
        thres_arr = threshold_image3(np.array(imgf, dtype=np.uint8))
        time_data.append(('threshold', time() - t))

        if denoise_mask is not None and denoise_mask:
            t = time()
            sigma_est = np.mean(estimate_sigma(thres_arr))
            time_data.append(('est_3', time() - t))

            if sigma_est > 0.1:
                t = time()
                thres_arr = denoise_bregman(thres_arr)
                time_data.append(('denoise', time() - t))


        thres_inv = thres_arr ^ np.ones(thres_arr.shape, dtype=bool)

        mask_arr |= thres_arr

    mask_inv = mask_arr ^ np.ones(mask_arr.shape, dtype=bool)

    t = time()
    # Take foreground pixels and create a blur around them, forcing the encoder
    # to not mess with our foreground pixels/colours too much, finally, plug
    # back our original foreground pixels in the blurred image.
    foreground_arr = partial_blur(mask_arr, image_arr, sigma=3,
            mode=image.mode)
    time_data.append(('fg_partial_blur', time() - t))

    t = time()
    # Take background pixels and stuff them into our foreground pixels, then
    # restore background.
    # This really only needs to touch pixels where mask_inv = 0.
    image_arr = partial_blur(mask_inv, image_arr, sigma=3,
                             mode=image.mode)
    time_data.append(('bg_partial_blur', time() - t))

    if bg_downsample is not None:
        t = time()
        image2 = Image.fromarray(image_arr)
        w, h = image2.size
        image2.thumbnail((w/bg_downsample, h/bg_downsample))
        image_arr = np.array(image2)
        time_data.append(('bg_downsample', time() - t))

    for v in time_data:
        print('%s: %.4fs ' % (v[0], v[1]), end='')
    print('')

    return mask_arr, image_arr, foreground_arr


def encode_mrc_images(mask, np_bg, np_fg, bg_slope=0.1, fg_slope=0.05,
                      tmp_dir=None, jbig2=True):
    # Create mask
    #fd, mask_img_png = mkstemp(prefix='mask', suffix='.pgm')
    fd, mask_img_png = mkstemp(prefix='mask', suffix='.png', dir=tmp_dir)
    close(fd)
    if jbig2:
        fd, mask_img_jbig2 = mkstemp(prefix='mask', suffix='.jbig2', dir=tmp_dir)
        close(fd)

    img = Image.fromarray(mask)
    img.save(mask_img_png, compress_level=0) # XXX: Check compress_level vs compress

    if jbig2:
        out = subprocess.check_output(['jbig2', mask_img_png])
        fp= open(mask_img_jbig2, 'wb+')
        fp.write(out)
        fp.close()

    # Create background
    fd, bg_img_tiff = mkstemp(prefix='bg', suffix='.tiff', dir=tmp_dir)
    close(fd)
    fd, bg_img_jp2 = mkstemp(prefix='bg', suffix='.jp2', dir=tmp_dir)
    close(fd)
    remove(bg_img_jp2) # XXX: Kakadu doesn't want the file to exist, so what are
                       # we even doing

    bg_img = Image.fromarray(np_bg)
    bg_img.save(bg_img_tiff)

    subprocess.check_call([KDU_COMPRESS,
        '-num_threads', '0',
        '-i', bg_img_tiff, '-o', bg_img_jp2,
        '-slope', str(bg_slope),
        'Clayers=20', 'Creversible=yes', 'Rweight=220', 'Rlevels=5',
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    remove(bg_img_tiff)

    # Create foreground
    fd, fg_img_tiff = mkstemp(prefix='fg', suffix='.tiff', dir=tmp_dir)
    close(fd)
    fd, fg_img_jp2 = mkstemp(prefix='fg', suffix='.jp2', dir=tmp_dir)
    close(fd)
    remove(fg_img_jp2) # XXX: Kakadu doesn't want the file to exist, so what are
                       # we even doing

    fg_img = Image.fromarray(np_fg)
    fg_img.save(fg_img_tiff)

    subprocess.check_call(['convert', mask_img_png, mask_img_png + '.pgm'])
    subprocess.check_call([KDU_COMPRESS,
        '-num_threads', '0',
        '-i', fg_img_tiff, '-o', fg_img_jp2,
        '-slope', str(fg_slope),
        'Clayers=20', 'Creversible=yes', 'Rweight=220', 'Rlevels=5',
         '-roi', mask_img_png + '.pgm,0.5',
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    remove(mask_img_png + '.pgm')
    remove(fg_img_tiff)

    if jbig2:
        remove(mask_img_png)


    # XXX: Return PNG (which mupdf will turn into ccitt) until mupdf fixes their
    # JBIG2 support
    #print(mask_img_png, bg_img_jp2, fg_img_jp2)
    #print(mask_img_jbig2, bg_img_jp2, fg_img_jp2)
    if jbig2:
        return mask_img_jbig2, bg_img_jp2, fg_img_jp2
    else:
        return mask_img_png, bg_img_jp2, fg_img_jp2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='PDF recoder using MRC and '\
                        'hOCR for text file placement')
    parser.add_argument('--jp2-stack', help='Base path of unpacked JP2 stack',
                        default=None)
    parser.add_argument('--tesseract-text-only-pdf', help='Path to tesseract'\
                        'text-only PDF (PDF with just invisible text)',
                        default=None)
    parser.add_argument('--out-pdf', help='File to write to', default=None)

    args = parser.parse_args()

    inpath = args.jp2_stack
    tesspath = args.tesseract_text_only_pdf
    outpath = args.out_pdf

    pdf = fitz.open(tesspath)

    i = 0
    for f in sorted(glob(inpath + '*.jp2')):
        # XXX: Make this /tmp/in.tiff) a tempfile
        subprocess.check_call([KDU_EXPAND, '-i', f, '-o', '/tmp/in.tiff'],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        mask, bg, fg = create_mrc_components(Image.open('/tmp/in.tiff'))
        mask_f, bg_f, fg_f = encode_mrc_images(mask, bg, fg)

        #page = pdf.newPage(-1)
        page = pdf[i]

        bg_contents = open(bg_f, 'rb').read()
        page.insertImage(page.rect, stream=bg_contents, mask=None)

        fg_contents = open(fg_f, 'rb').read()
        mask_contents = open(mask_f, 'rb').read()

        page.insertImage(page.rect, stream=fg_contents, mask=mask_contents)

        remove(mask_f)
        remove(bg_f)
        remove(fg_f)

        i += 1
        if i % 10 == 0:
            print('Saving')
            pdf.save(outpath)

    print(fitz.TOOLS.mupdf_warnings())
    pdf.save(outpath, deflate=True)
