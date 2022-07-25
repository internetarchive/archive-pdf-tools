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
#
# Helper functions for efficient JPEG2000 handing using Pillow and/or external
# binaries.

import sys
from os import close, remove
from subprocess import check_call, DEVNULL
from tempfile import mkstemp
from ast import literal_eval

from PIL import Image
from PIL import Jpeg2KImagePlugin

from internetarchivepdf.const import RECODE_RUNTIME_WARNING_INVALID_JP2_HEADERS
from internetarchivepdf.const import (JPEG2000_IMPL_KAKADU,
        JPEG2000_IMPL_OPENJPEG, JPEG2000_IMPL_GROK,
        JPEG2000_IMPL_PILLOW, JPEG2000_IMPLS)


KDU_COMPRESS = 'kdu_compress'
KDU_EXPAND = 'kdu_expand'
OPJ_COMPRESS = 'opj_compress'
OPJ_DECOMPRESS = 'opj_decompress'
GRK_COMPRESS = 'grk_compress'
GRK_DECOMPRESS = 'grk_decompress'

def encode_jpeg2000(image, outpath, impl, flags, tmp_dir=None, imgtype=None,
        threads=None, debug=False):
    """ Encode PIL image to JPEG2000 file

    Args:

    * image (PIL.Image): image to compress
    * impl (str): JPEG2000 implementation
    * flags list of str: encoding flags
    * threads (int): How many threads to use
    """
    if impl not in JPEG2000_IMPLS:
        raise Exception('Error: invalid jpeg2000 implementation?')

    if impl == JPEG2000_IMPL_PILLOW:
        kwargs = _jpeg2000_pillow_str_to_kwargs(flags[0])
        image.save(outpath, **kwargs)
    elif impl in (JPEG2000_IMPL_KAKADU, JPEG2000_IMPL_GROK,
            JPEG2000_IMPL_OPENJPEG):
        if impl in (JPEG2000_IMPL_KAKADU, JPEG2000_IMPL_GROK):
            fd, img_tiff = mkstemp(prefix=imgtype, suffix='.tif', dir=tmp_dir)
        else:
            fd, img_tiff = mkstemp(prefix=imgtype, suffix='.pnm', dir=tmp_dir)
        close(fd)

        image.save(img_tiff)

        args = ['-i', img_tiff, '-o', outpath]
        args += flags

        # Specify threads at the end, because some older versions of OpenJPEG do
        # not support threads, but if we specify the flags at the end, it will
        # just ignore the threads and not the variables passed before that (see
        # https://github.com/internetarchive/archive-pdf-tools/issues/41)
        args = add_impl_args(args, impl, encode=True, threads=threads)

        if debug:
            print('check_call: %s' % args, file=sys.stderr)
        check_call(args, stdout=DEVNULL, stderr=DEVNULL)

        remove(img_tiff)


def decode_jpeg2000(infile, reduce_=None, impl=JPEG2000_IMPL_PILLOW,
        tmp_dir=None, threads=None, debug=False):
    """ Decode JPEG2000 file to PIL image

    Args:

    * infile (str): Path of image to load
    * reduce_ (bool or None): Optional, reduction factor
    * impl (str): JPEG2000 implementation
    * tmp_dir (str): Temporary directory to use, if any
    * threads (int): How many theads to use

    Returns: loaded image (PIL.Image)
    """
    if impl not in JPEG2000_IMPLS:
        raise Exception('Error: invalid jpeg2000 implementation?')

    if reduce_ is not None:
        # TODO: Check if reduce_ is an int? (not a float, etc)
        reduce_ = int(reduce_ - 1)
        if reduce_ == 1:
            # Don't reduce when downsample = 1, it means we don't want to change
            # it, and it also complicates the arg handling below
            reduce_ = None

    img = None
    img_tiff = None

    if impl == JPEG2000_IMPL_PILLOW:
        img = Image.open(infile)
        if reduce_ is not None:
            img = img.reduce(reduce_)
    else:
        fd, img_tiff = mkstemp(suffix='.tif', dir=tmp_dir)
        close(fd)

        args = ['-i', infile, '-o', img_tiff]
        if reduce_ is not None:
            if impl in (JPEG2000_IMPL_KAKADU,):
                args += ['-reduce', str(reduce_ - 1)]
            if impl in (JPEG2000_IMPL_OPENJPEG, JPEG2000_IMPL_GROK):
                args += ['-r', str(reduce_ - 1)]

        args = add_impl_args(args, impl, encode=False, threads=threads)

        if debug:
            print('check_call: %s' % args, file=sys.stderr)
        check_call(args, stdout=DEVNULL, stderr=DEVNULL)

        img = Image.open(img_tiff)

    img.load()

    if sys.platform == 'win32':
        new_img = img.copy()
        img.close()
        img = new_img

    if img_tiff is not None:
        remove(img_tiff)

    return img


def get_jpeg2000_info(infile, impl, errors=None):
    size = None
    mode = None

    # Pillow reads the entire file for JPEG2000 images - just to get
    # the image size - so let's use some internal functions to speed that up.
    fd = open(infile, 'rb')
    try:
        size, mode, mimetype, dpi = Jpeg2KImagePlugin._parse_jp2_header(fd)
    except Exception:
        # JP2 lacks some info and PIL doesn't like it (Image.open
        # will not work, so use kdu_expand to create a tiff)
        if errors is not None:
            errors.add(RECODE_RUNTIME_WARNING_INVALID_JP2_HEADERS)

        img = decode_jpeg2000(infile, impl=impl)
        size = img.size
        mode = img.mode
        img = None
    finally:
        fd.close()

    return size, mode


def add_impl_args(args, impl, encode=False, threads=None):
    threads = str(threads) if threads else '1'

    # Use just one core
    if impl in (JPEG2000_IMPL_KAKADU,):
        # From kdu_expand/kdu_compress:
        # The special value of 0 may be used to specify that you want to use the
        # conventional single-threaded processing machinery -- i.e., you don't
        # want to create or use a threading environment.
        if threads == '1':
            threads = '0'
        args += ['-num_threads', threads]
        if encode:
            args = [KDU_COMPRESS] + args
        else:
            args = [KDU_EXPAND] + args
    if impl in (JPEG2000_IMPL_OPENJPEG,):
        args += ['-threads', threads]
        if encode:
            args = [OPJ_COMPRESS] + args
        else:
            args = [OPJ_DECOMPRESS] + args
    if impl in (JPEG2000_IMPL_GROK,):
        args += ['-H', threads]
        if encode:
            args = [GRK_COMPRESS] + args
        else:
            args = [GRK_DECOMPRESS] + args

    return args

def _jpeg2000_pillow_str_to_kwargs(s):
    kwargs = {}
    for en in s.split(';'):
        k, v = en.split(':', maxsplit=1)
        kwargs[k] = literal_eval(v)

    return kwargs
