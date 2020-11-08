# Author: Merlijn Boris Wolf Wajer <merlijn@archive.org>

from os import close, remove

from glob import glob
from tempfile import mkstemp
import subprocess

from PIL import Image, ImageEnhance
from skimage.filters import threshold_local, threshold_otsu, threshold_sauvola
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
#KDU_COMPRESS = '/home/merlijn/archive/microfilm-issue-generator/bin/kdu_compress'
#KDU_EXPAND = '/home/merlijn/archive/microfilm-issue-generator/bin/kdu_expand'


def threshold_image(pil_image, rev=False, otsu=False):
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
        block_size = 9
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


def threshold_image3(pil_image, rev=False):
    img = np.array(pil_image)

    window_size = 101
    thres_sauvola = threshold_sauvola(img, window_size=window_size, k=0.6)
    if rev:
        binary_img = img > thres_sauvola
    else:
        binary_img = img < thres_sauvola

    return binary_img


# TODO: Rename, can be either foreground or background
def special_foreground(mask, fg, sigma=5, mode=None):
    maskf = np.array(mask, dtype=np.float)

    if mode == 'RGB' or mode == 'RGBA':
        in_r = fg[:, :, 0] * maskf
        in_g = fg[:, :, 1] * maskf
        in_b = fg[:, :, 2] * maskf
        filter_r = ndimage.filters.gaussian_filter(in_r, sigma = sigma)
        filter_g = ndimage.filters.gaussian_filter(in_g, sigma = sigma)
        filter_b = ndimage.filters.gaussian_filter(in_b, sigma = sigma)
    else:
        fgf = np.copy(fg)
        fgf = np.array(fgf, dtype=np.float)
        filter = ndimage.filters.gaussian_filter(fgf * maskf, sigma = sigma)

    weights = ndimage.filters.gaussian_filter(maskf, sigma = sigma)

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


def create_mrc_hocr_components(image, hocr_word_data, bg_downsample=None):
    img = image
    if image.mode != 'L':
        img = image.convert('L')

    image_mask = Image.new('1', image.size)
    mask_arr = np.array(image_mask)

    MIX_THRESHOLD = True

    for paragraphs in hocr_word_data:
        for lines in paragraphs['lines']:
            for word in lines['words']:
                if not word['text'].strip():
                    continue

                top, left, bottom, right = [int(x) for x in word['bbox']]

                wordimg = img.crop(word['bbox'])
                thres = threshold_image2(wordimg)
                mask_arr[left:right, top:bottom] = thres

                # TODO: Set thres values in array_mask

                intbox = [int(x) for x in word['bbox']]

    image_arr = np.array(image)

    if MIX_THRESHOLD:
        thres_arr = threshold_image3(img)
        thres_inv = thres_arr ^ np.ones(thres_arr.shape, dtype=bool)

        mask_arr |= thres_arr

    mask_inv = mask_arr ^ np.ones(mask_arr.shape, dtype=bool)

    foreground_arr = special_foreground(mask_arr, image_arr, sigma=6,
            mode=image.mode)

    ## TODO: Clean this up
    mask_arr_f = np.array(mask_arr, dtype=np.float)
    mask_blur = ndimage.filters.gaussian_filter(mask_arr_f, sigma=1)
    #mask_blur = ndimage.filters.gaussian_filter(mask_arr_f, sigma=1)
    mask_blur[mask_blur > 0.00001] = 1.
    mask_blurb = mask_blur > 0.0001
    mask_inv_blur = mask_blurb ^ np.ones(mask_blurb.shape, dtype=bool)
    #image_arr = special_foreground(mask_inv_blur, image_arr, sigma=10,
    #image_arr = special_foreground(mask_inv_blur, image_arr, sigma=10,
    image_arr = special_foreground(mask_inv, image_arr, sigma=10,
                                   mode=image.mode)

    diff_mask = mask_blurb & ~mask_arr
    diff_mask_inv = diff_mask ^ np.ones(diff_mask.shape, dtype=bool)
    bg_blur = special_foreground(diff_mask_inv, image_arr, mode=image.mode,
            sigma=30)
    image_arr = bg_blur

    ## image_arr = background
    #image_arr = special_foreground(mask_inv, image_arr, sigma=10,
    #                               mode=image.mode)

    # TODO: Turn this into a command line argument:
    if bg_downsample is not None:
        image2 = Image.fromarray(image_arr)
        w, h = image2.size
        image2.thumbnail((w/bg_downsample, h/bg_downsample))
        image_arr = np.array(image2)

    fg = Image.fromarray(foreground_arr)
    en = ImageEnhance.Brightness(fg)
    tmp = en.enhance(0.5)
    en = ImageEnhance.Contrast(tmp)
    foreground_arr = np.array(en.enhance(2.0))

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
