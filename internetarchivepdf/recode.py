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
import pkg_resources

import sys
import os
import subprocess
from os import remove
from time import time
from datetime import datetime
from tempfile import mkstemp
from os.path import join
import shutil
import json
from glob import glob
import re
import io


from PIL import Image
import numpy as np
import fitz

from hocr.parse import (hocr_page_iterator, hocr_page_to_word_data,
        hocr_page_get_dimensions, hocr_page_get_scan_res)
from internetarchivepdf.mrc import create_mrc_hocr_components, \
        encode_mrc_images, encode_mrc_mask
from internetarchivepdf.grayconvert import special_gray_convert
from internetarchivepdf.pdfhacks import fast_insert_image, write_pdfa, \
        write_page_labels, write_basic_ua, write_metadata
from internetarchivepdf.pdfrenderer import TessPDFRenderer
from internetarchivepdf.scandata import scandata_xml_get_skip_pages, \
        scandata_xml_get_page_numbers, scandata_xml_get_dpi_per_page, \
        scandata_xml_get_document_dpi
from internetarchivepdf.jpeg2000 import decode_jpeg2000, get_jpeg2000_info
from internetarchivepdf.const import (IMAGE_MODE_PASSTHROUGH, IMAGE_MODE_PIXMAP,
        IMAGE_MODE_MRC, RECODE_RUNTIME_WARNING_INVALID_PAGE_SIZE,
        RECODE_RUNTIME_WARNING_INVALID_PAGE_NUMBERS,
        RECODE_RUNTIME_WARNING_INVALID_JP2_HEADERS, JPEG2000_IMPL_KAKADU,
        JPEG2000_IMPL_OPENJPEG, JPEG2000_IMPL_GROK, JPEG2000_IMPL_PILLOW,
        COMPRESSOR_JPEG2000, COMPRESSOR_JPEG)

PDFA_MIN_UNITS = 3
PDFA_MAX_UNITS = 14400

Image.MAX_IMAGE_PIXELS = 625000000


def guess_dpi(w, h, expected_format=(8.27, 11.69), round_to=[72, 96, 150, 300, 600]):
    """
    Guesstimate DPI for a given image.

    Args:

    * w (int): width of the image
    * h (int): height of the image
    * expected_format (tuple): (width_inch, height_inch) of expected document,
                               defaults to european A4.
    * round_to (list of int): List of acceptable DPI values.
                              Defaults to (72, 96, 150, 300, 600)

    Returns an int which is the best matching DPI picked from round_to.
    """
    w_dpi = w / expected_format[0]
    h_dpi = h / expected_format[1]
    diffs = []
    for dpi in round_to:
        diff = abs(w_dpi - dpi) + abs(h_dpi - dpi)
        diffs.append((dpi, diff))
    sorted_diffs = sorted(diffs, key=lambda x: x[1])
    return sorted_diffs[0][0]


def create_tess_textonly_pdf(hocr_file, save_path, in_pdf=None,
        image_files=None, dpi=None, skip_pages=None, dpi_pages=None,
        reporter=None,
        verbose=False, debug=False, stop_after=None,
        render_text_lines=False,
        tmp_dir=None,
        jpeg2000_implementation=None,
        errors=None):
    hocr_iter = hocr_page_iterator(hocr_file)

    render = TessPDFRenderer(render_text_lines=render_text_lines)
    render.BeginDocumentHandler()

    skipped_pages = 0

    last_time = time()
    reporting_page_count = 0

    if verbose:
        print('Starting page generation at', datetime.utcnow().isoformat())

    for idx, hocr_page in enumerate(hocr_iter):
        w, h = hocr_page_get_dimensions(hocr_page)
        hocr_dpi = hocr_page_get_scan_res(hocr_page)
        # If scan_res is not found in hOCR, it returns (None, None)
        hocr_dpi = hocr_dpi[1]

        if skip_pages is not None and idx in skip_pages:
            if verbose:
                print('Skipping page %d' % idx)
            skipped_pages += 1
            continue

        if stop_after is not None and (idx - skipped_pages) >= stop_after:
            break

        if in_pdf is not None:
            page = in_pdf[idx - skipped_pages]
            width = page.rect.width
            height = page.rect.height

            scaler = page.rect.width / w
            ppi = 72 / scaler
        elif image_files is not None:
            # Do not subtract skipped pages here
            imgfile = image_files[idx]

            if imgfile.endswith('.jp2'):
                size, _ = get_jpeg2000_info(imgfile, jpeg2000_implementation, errors)
                imwidth, imheight = size
            else:
                img = Image.open(imgfile)
                imwidth, imheight = img.size
                del img

            page_dpi = dpi
            per_page_dpi = None

            if dpi_pages is not None:
                try:
                    per_page_dpi = int(dpi_pages[idx - skipped_pages])
                    page_dpi = per_page_dpi
                except:
                    pass  # Keep item-wide dpi if available

            # Both document level dpi is not available and per-page dpi is not
            # available, let's guesstimate
            # Assume european A4 (8.27",11.69") and guess DPI
            # to be one-of (72, 96, 150, 300, 600)
            if page_dpi is None:
                page_dpi = guess_dpi(imwidth, imheight,
                                     expected_format=(8.27, 11.69),
                                     round_to=(72, 96, 150, 300, 600))

            page_width = imwidth / (page_dpi / 72)
            if page_width <= PDFA_MIN_UNITS or page_width >= PDFA_MAX_UNITS:
                if verbose:
                    print('Page size invalid with current image size and dpi.')
                    print('Image size: %d, %d. DPI: %d' % (imwidth, imheight,
                                                           page_dpi))

                # First let's try without per_page_dpi, is avail, then try to
                # guess the page dpi, if that also fails, then set to min
                # or max allowed size
                if per_page_dpi is not None and dpi:
                    if verbose:
                        print('Trying document level dpi:', dpi)
                    page_width = imwidth / (dpi / 72)

                # If that didn't work, guess
                if page_width <= PDFA_MIN_UNITS or page_width >= PDFA_MAX_UNITS:
                    page_dpi = guess_dpi(imwidth, imheight,
                                         expected_format=(8.27, 11.69),
                                         round_to=(72, 96, 150, 300, 600))
                    if verbose:
                        print('Guessing DPI:', dpi)
                    page_width = imwidth / (page_dpi / 72)

                # If even guessing fails, let's just set minimal values since
                # this typically only happens for really tiny images
                if page_width <= PDFA_MIN_UNITS or page_width >= PDFA_MAX_UNITS:
                    page_width = PDFA_MIN_UNITS + 1
                    page_height = PDFA_MIN_UNITS + 1

                # Add warning/error
                if errors is not None:
                    errors.add(RECODE_RUNTIME_WARNING_INVALID_PAGE_SIZE)

            scaler = page_width / imwidth

            ppi = 72. / scaler

            width = page_width
            height = imheight * scaler

        font_scaler = 1
        if hocr_dpi is not None:
            font_scaler = hocr_dpi / ppi
        else:
            font_scaler = 72. / ppi

        word_data = hocr_page_to_word_data(hocr_page, font_scaler)
        render.AddImageHandler(word_data, width, height, ppi=ppi, hocr_ppi=hocr_dpi)

        reporting_page_count += 1


    if verbose:
        print('Finished page generation at', datetime.utcnow().isoformat())
        print('Creating text pages took %.4f seconds' % (time() - last_time))


    if reporter and reporting_page_count != 0:
        current_time = time()
        ms = int(((current_time - last_time) / reporting_page_count) * 1000)

        data = json.dumps({'text_pages': {'count': reporting_page_count,
                                              'time-per': ms}})
        subprocess.check_output(reporter, input=data.encode('utf-8'))

    render.EndDocumentHandler()

    fp = open(save_path, 'wb+')
    fp.write(render._data)
    fp.close()


def get_timing_summary(timing_data):
    sums = {}

    # We expect this to always happen per page
    image_load_c = 0

    for v in timing_data:
        key = v[0]
        val = v[1]

        if key == 'image_load':
            image_load_c += 1

        if key not in sums:
            sums[key] = 0.

        sums[key] += val

    for k in sums.keys():
        sums[k] = sums[k] / image_load_c

    for k in sums.keys():
        # For statsd, in ms
        sums[k] = int(sums[k] * 1000)

    return sums



def insert_images_mrc(to_pdf, hocr_file, from_pdf=None, image_files=None,
        dpi=None, dpi_pages=None,
        bg_compression_flags=None, fg_compression_flags=None,
        skip_pages=None, img_dir=None, jbig2=False,
        downsample=None,
        bg_downsample=None,
        fg_downsample=None,
        denoise_mask=None, reporter=None,
        hq_pages=None, hq_bg_compression_flags=None, hq_fg_compression_flags=None,
        verbose=False, debug=False, tmp_dir=None, report_every=None,
        stop_after=None, grayscale_pdf=False,
        force_1bit_output=None,
        jpeg2000_implementation=None, mrc_image_format=None, threads=None,
        errors=None):
    hocr_iter = hocr_page_iterator(hocr_file)

    skipped_pages = 0

    last_time = time()
    timing_data = []
    reporting_page_count = 0

    downsampled = False

    #for idx, page in enumerate(to_pdf):
    for idx, hocr_page in enumerate(hocr_iter):
        if skip_pages is not None and idx in skip_pages:
            skipped_pages += 1
            continue

        idx = idx - skipped_pages

        if stop_after is not None and idx >= stop_after:
            break

        picked_dpi = None

        hocr_dpi = hocr_page_get_scan_res(hocr_page)

        if dpi_pages is not None:
            picked_dpi = dpi_pages[idx]
            if picked_dpi is None:
                picked_dpi = hocr_dpi[1]

        if picked_dpi is None:
            picked_dpi = dpi

        if picked_dpi is not None:
            picked_dpi = int(picked_dpi)

        page = to_pdf[idx]

        if from_pdf is not None:
            # TODO: Support more images and their masks, if they exist (and
            # write them to the right place in the PDF)
            t = time()

            img = from_pdf[idx].get_images()[0]
            xref = img[0]
            maskxref = img[1]

            image = from_pdf.extract_image(xref)
            imgfd = io.BytesIO()
            imgfd.write(image["image"])
            image = Image.open(imgfd)
            image.load()
            imgfd.close()

            if timing_data is not None:
                timing_data.append(('image_load', time()-t))
        else:
            t = time()
            # Do not subtract skipped pages here
            imgfile = image_files[idx+skipped_pages]

            # Potentially special path
            if imgfile.endswith('.jp2') or imgfile.endswith('.jpx'):
                image = decode_jpeg2000(imgfile, reduce_=downsample,
                        impl=jpeg2000_implementation, threads=threads, debug=debug)
                if downsample:
                    downsampled = True
            else:
                image = Image.open(imgfile)
                image.load()

            if image.mode in ('RGBA', 'LA'):
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                elif image.mode == 'LA':
                    image = image.convert('L')

            if timing_data is not None:
                timing_data.append(('image_load', time()-t))

        if grayscale_pdf and image.mode not in ('L', 'LA'):
            t = time()
            image = Image.fromarray(special_gray_convert(np.array(image)))
            if timing_data is not None:
                timing_data.append(('special_gray_convert', time()-t))

        render_hq = hq_pages[idx]

        if downsample is not None and not downsampled:
            w, h = image.size
            image.thumbnail((w/downsample, h/downsample),
                            resample=Image.LANCZOS, reducing_gap=None)
            downsampled = True

        hocr_word_data = hocr_page_to_word_data(hocr_page)

        if image.mode == '1':
            ww, hh = image.size
            mask_jb2, mask_png = encode_mrc_mask(np.array(image), tmp_dir=tmp_dir,
                    jbig2=jbig2, timing_data=timing_data, debug=debug)

            t = time()

            if jbig2:
                mask_contents = open(mask_jb2, 'rb').read()
                remove(mask_jb2)
            else:
                mask_contents = open(mask_png, 'rb').read()

            # We currently always return the PNG file
            remove(mask_png)

            page.insert_image(page.rect, stream=mask_contents,
                    width=ww, height=hh, alpha=0)

            if timing_data is not None:
                timing_data.append(('page_image_insertion', time() - t))

        elif force_1bit_output == True:
            ww, hh = image.size
            mrc_gen = create_mrc_hocr_components(image, hocr_word_data,
                    dpi=picked_dpi,
                    downsample=downsample,
                    bg_downsample=None if render_hq else bg_downsample,
                    fg_downsample=None if render_hq else fg_downsample,
                    denoise_mask=denoise_mask,
                    timing_data=timing_data, errors=errors)
            np_mask = next(mrc_gen)
            np_mask = np_mask ^ np.ones(np_mask.shape, dtype=bool)
            mask_jb2, mask_png = encode_mrc_mask(np_mask, tmp_dir=tmp_dir, jbig2=jbig2,
                    timing_data=timing_data, debug=debug)

            if jbig2:
                mask_contents = open(mask_jb2, 'rb').read()
                remove(mask_jb2)
            else:
                mask_contents = open(mask_png, 'rb').read()

            # We currently always return the PNG file
            remove(mask_png)

            page.insert_image(page.rect, stream=mask_contents,
                    width=ww, height=hh, alpha=0)

            if timing_data is not None:
                timing_data.append(('page_image_insertion', time() - t))
        else:
            mrc_gen = create_mrc_hocr_components(image, hocr_word_data,
                    dpi=picked_dpi,
                    downsample=downsample,
                    bg_downsample=None if render_hq else bg_downsample,
                    fg_downsample=None if render_hq else fg_downsample,
                    denoise_mask=denoise_mask,
                    timing_data=timing_data, errors=errors)


            # TODO: keep all these files on disk, and insert them into the pager
            # later? maybe? or just saveIncr()
            # TODO: maybe call the encode_mrc_{mask,foreground,background}
            # separately from here so that we can free the arrays sooner (and even
            # get the images separately from the create_mrc_hocr_components call)

            fast_insert_image_ok = jbig2 and image.mode in ('L', 'RGB')

            mask_f, bg_f, bg_s, fg_f, fg_s = encode_mrc_images(mrc_gen,
                    bg_compression_flags=hq_bg_compression_flags if render_hq else bg_compression_flags,
                    fg_compression_flags=hq_fg_compression_flags if render_hq else fg_compression_flags,
                    tmp_dir=tmp_dir, jbig2=jbig2, timing_data=timing_data,
                    jpeg2000_implementation=jpeg2000_implementation,
                    mrc_image_format=mrc_image_format,
                    embedded_jbig2=fast_insert_image_ok,
                    threads=threads,
                    debug=debug)

            if img_dir is not None:
                shutil.copy(mask_f, join(img_dir, '%.6d_mask.jbig2' % idx))
                shutil.copy(bg_f, join(img_dir, '%.6d_bg.jp2' % idx))
                shutil.copy(fg_f, join(img_dir, '%.6d_fg.jp2' % idx))


            t = time()
            bg_contents = open(bg_f, 'rb').read()
            if not jbig2 or image.mode not in ('L', 'RGB'):
                # Tell PyMuPDF about width/height/alpha since it's faster this way
                page.insert_image(page.rect, stream=bg_contents, mask=None,
                    overlay=False, width=bg_s[0], height=bg_s[1], alpha=0)
            else:
                fast_insert_image(page, page.rect, stream=bg_contents,
                                  mask=None, width=bg_s[0], height=bg_s[1],
                                  stream_fmt=mrc_image_format,
                                  gray=image.mode == 'L')

            fg_contents = open(fg_f, 'rb').read()
            mask_contents = open(mask_f, 'rb').read()

            # Tell PyMuPDF about width/height/alpha since it's faster this way
            if not jbig2 or image.mode not in ('L', 'RGB'):
                page.insert_image(page.rect, stream=fg_contents, mask=mask_contents,
                        overlay=True, width=fg_s[0], height=fg_s[1], alpha=0)
            else:
                fast_insert_image(page, page.rect, stream=fg_contents,
                                  mask=mask_contents, width=fg_s[0], height=fg_s[1],
                                  stream_fmt=mrc_image_format,
                                  gray=image.mode == 'L')

            # Remove leftover files
            remove(mask_f)
            remove(bg_f)
            remove(fg_f)
            if timing_data is not None:
                timing_data.append(('page_image_insertion', time() - t))

        reporting_page_count += 1

        if report_every is not None and reporting_page_count % report_every == 0:
            print('Processed %d PDF pages.' % idx)
            sys.stdout.flush()

            timing_sum = get_timing_summary(timing_data)
            timing_data = []

            if reporter:
                current_time = time()
                ms = int(((current_time - last_time) / reporting_page_count) * 1000)

                data = json.dumps({'compress_pages': {'count': reporting_page_count,
                                                 'time-per': ms},
                                   'page_time_breakdown': timing_sum})
                subprocess.check_output(reporter, input=data.encode('utf-8'))

                # Reset chunk timer
                last_time = time()
                # Reset chunk counter
                reporting_page_count = 0


    if reporter and reporting_page_count != 0:
        current_time = time()
        ms = int(((current_time - last_time) / reporting_page_count) * 1000)

        timing_sum = get_timing_summary(timing_data)

        data = json.dumps({'compress_pages': {'count': reporting_page_count,
                                         'time-per': ms},
                           'page_time_breakdown': timing_sum})
        subprocess.check_output(reporter, input=data.encode('utf-8'))

    if verbose:
        summary = get_timing_summary(timing_data)
        print('MRC time breakdown:', summary)


def insert_images(from_pdf, to_pdf, mode, report_every=None, stop_after=None):
    # TODO: This hasn't been updated, should fix this up, only MRC is tested
    # really.
    # TODO: implement img_dir here

    for idx, page in enumerate(to_pdf):
        # XXX: TODO: FIXME: MEGAHACK: For some reason the _imgonly PDFs
        # generated by us have all images on all pages according to pymupdf, so
        # hack around that for now.
        img = sorted(from_pdf.getPageImageList(idx))[idx]
        #img = from_pdf.getPageImageList(idx)[0]

        xref = img[0]
        maskxref = img[1]
        if mode == IMAGE_MODE_PASSTHROUGH:
            image = from_pdf.extract_image(xref)
            page.insert_image(page.rect, stream=image["image"], overlay=False)
        elif mode == IMAGE_MODE_PIXMAP:
            pixmap = fitz.Pixmap(from_pdf, xref)
            page.insert_image(page.rect, pixmap=pixmap, overlay=False)

        if stop_after is not None and idx >= stop_after:
            break

        if report_every is not None and idx % report_every == 0:
            print('Processed %d PDF pages.' % idx)
            sys.stdout.flush()


# TODO: Document these options (like in bin/recode_pdf)
def recode(from_pdf=None, from_imagestack=None, dpi=None, hocr_file=None,
        scandata_file=None, out_pdf=None, out_dir=None,
        reporter=None,
        grayscale_pdf=False,
        force_1bit_output=False,
        image_mode=IMAGE_MODE_MRC, jbig2=False, verbose=False, debug=False,
        tmp_dir=None,
        report_every=None, stop_after=None,
        jpeg2000_implementation=JPEG2000_IMPL_PILLOW,
        bg_compression_flags=None, fg_compression_flags=None,
        mrc_image_format=None,
        downsample=None,
        bg_downsample=None,
        fg_downsample=None,
        denoise_mask=None,
        hq_pages=None,
        hq_bg_compression_flags=None, hq_fg_compression_flags=None,
        threads=None,
        render_text_lines=False,
        metadata_url=None, metadata_title=None, metadata_author=None,
        metadata_creator=None, metadata_language=None,
        metadata_subject=None, metadata_creatortool=None):
    # TODO: document that the scandata document dpi will override the dpi arg
    # TODO: Take hq-pages and reporter arg and change format (as lib call we
    # don't want to pass that as one string, I guess?)

    errors = set()

    in_pdf = None
    if from_pdf:
        in_pdf = fitz.open(from_pdf)

    image_files = None
    if from_imagestack:
        image_files = sorted(glob(from_imagestack))

    hocr_file = hocr_file
    outfile = out_pdf

    stop = stop_after
    if stop is not None:
        stop -= 1

    if verbose:
        from numpy.core._multiarray_umath import __cpu_features__ as cpu_have
        cpu = cpu_have
        for k, v in cpu.items():
            if v:
                print('\t', k)


    reporter = reporter.split(' ') if reporter else None # TODO: overriding

    start_time = time()

    scandata_doc_dpi = None

    # Figure out if we have scandata, and figure out if we want to skip pages
    # based on scandata.
    skip_pages = []
    dpi_pages = None
    if scandata_file is not None:
        skip_pages = scandata_xml_get_skip_pages(scandata_file)
        dpi_pages = scandata_xml_get_dpi_per_page(scandata_file)
        scandata_doc_dpi = scandata_xml_get_document_dpi(scandata_file)

        if scandata_doc_dpi is not None:
            # Let's prefer the DPI in the scandata file over the provided DPI
            dpi = scandata_doc_dpi

    # XXX: Maybe use a buffer, since the file is typically quite small
    fd, tess_tmp_path = mkstemp(prefix='pdfrenderer', suffix='.pdf', dir=tmp_dir)
    os.close(fd)

    if verbose:
        print('Creating text only PDF')

    # 1. Create text-only PDF from hOCR first, but honour page sizes of in_pdf
    create_tess_textonly_pdf(hocr_file, tess_tmp_path, in_pdf=in_pdf,
            image_files=image_files, dpi=dpi,
            skip_pages=skip_pages, dpi_pages=dpi_pages,
            reporter=reporter,
            verbose=verbose, debug=debug, stop_after=stop,
            render_text_lines=render_text_lines,
            tmp_dir=tmp_dir,
            jpeg2000_implementation=jpeg2000_implementation,
            errors=errors)

    if verbose:
        print('Inserting (and compressing) images')
    # 2. Load tesseract PDF and stick images in the PDF
    # We open the generated file but do not modify it in place
    outdoc = fitz.open(tess_tmp_path)

    HQ_PAGES = [False for x in range(outdoc.page_count)]
    if hq_pages is not None:
        index_range = map(int, hq_pages.split(','))
        for i in index_range:
            # We want 0-indexed, not 1-indexed, but not negative numbers we want
            # to remain 1-indexed.
            if i > 0:
                i = i - 1

            if abs(i) >= len(HQ_PAGES):
                # Page out of range, silently ignore for automation purposes.
                # We don't want scripts that call out tool to worry about how
                # many a PDF has exactly. E.g. if 1,2,3,4,-4,-3,-2,-1 is passed,
                # and a PDF has only three pages, let's just set them all to HQ
                # and not complain about 4 and -4 being out of range.
                continue

            # Mark page as HQ
            HQ_PAGES[i] = True


    if verbose:
        print('Converting with image mode:', image_mode)
    if image_mode == 2:
        insert_images_mrc(outdoc, hocr_file,
                          from_pdf=in_pdf,
                          image_files=image_files,
                          dpi=dpi,
                          dpi_pages=dpi_pages,
                          bg_compression_flags=bg_compression_flags,
                          fg_compression_flags=fg_compression_flags,
                          skip_pages=skip_pages,
                          img_dir=out_dir,
                          jbig2=jbig2,
                          downsample=downsample,
                          bg_downsample=bg_downsample,
                          fg_downsample=fg_downsample,
                          denoise_mask=denoise_mask,
                          reporter=reporter,
                          hq_pages=HQ_PAGES,
                          hq_bg_compression_flags=hq_bg_compression_flags,
                          hq_fg_compression_flags=hq_fg_compression_flags,
                          verbose=verbose,
                          debug=debug,
                          tmp_dir=tmp_dir,
                          report_every=report_every,
                          stop_after=stop,
                          grayscale_pdf=grayscale_pdf,
                          force_1bit_output=force_1bit_output,
                          jpeg2000_implementation=jpeg2000_implementation,
                          mrc_image_format=mrc_image_format,
                          threads=threads,
                          errors=errors)
    elif image_mode in (0, 1):
        # TODO: Update this codepath
        insert_images(in_pdf, outdoc, mode=image_mode,
                report_every=report_every, stop_after=stop)
    elif image_mode == 3:
        # 3 = skip
        pass

    # 3. Add PDF/A compliant data
    write_pdfa(outdoc)

    if scandata_file is not None:
        # XXX: we parse scandata twice now, let's not do that
        # 3b. Write page labels from scandata file, if present
        write_page_labels(outdoc, scandata_file, errors=errors)


    lang_if_any = metadata_language[0] if metadata_language else None
    write_basic_ua(outdoc, language=lang_if_any)

    # 4. Write metadata
    extra_metadata = {}
    if metadata_url:
        extra_metadata['url'] = metadata_url
    if metadata_title:
        extra_metadata['title'] = metadata_title
    if metadata_creator:
        extra_metadata['creator'] = metadata_creator
    if metadata_author:
        extra_metadata['author'] = metadata_author
    if metadata_language:
        extra_metadata['language'] = metadata_language
    if metadata_subject:
        extra_metadata['subject'] = metadata_subject
    if metadata_creatortool:
        extra_metadata['creatortool'] = metadata_creatortool
    write_metadata(in_pdf, outdoc, extra_metadata=extra_metadata)

    # 5. Save
    mupdf_warnings = fitz.TOOLS.mupdf_warnings()
    if mupdf_warnings:
        print('mupdf warnings:', repr(mupdf_warnings))
    if verbose:
        print('Saving PDF now')

    t = time()
    outdoc.save(outfile, deflate=True, pretty=True)
    save_time_ms = int((time() - t)*1000)
    if reporter:
        data = json.dumps({'time_to_save': {'time': save_time_ms}})
        subprocess.check_output(reporter, input=data.encode('utf-8'))

    end_time = time()
    print('Processed %d pages at %.2f seconds/page' % (len(outdoc),
        (end_time - start_time) / len(outdoc)))

    if from_pdf is not None:
        oldsize = os.path.getsize(from_pdf)
    else:
        bytesum = 0
        skipped_pages = 0
        for idx, fname in enumerate(image_files):
            if skip_pages is not None and idx in skip_pages:
                skipped_pages += 1
                continue

            if stop_after is not None and (idx - skipped_pages) > stop_after:
                break

            bytesum += os.path.getsize(fname)

        oldsize = bytesum

    newsize = os.path.getsize(out_pdf)
    compression_ratio  = oldsize / newsize
    if verbose:
        print('Compression ratio: %f' % (compression_ratio))

    # 5. Remove leftover files
    outdoc.close()
    remove(tess_tmp_path)

    return {'errors': errors,
            'compression_ratio': compression_ratio}
