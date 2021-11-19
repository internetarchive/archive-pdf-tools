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
# Functions to (more quickly) create and modify PDFs.
#
# Some of this code is contributed by Jorj X. McKie <jorj.x.mckie@outlook.de>
# License in this file if AGPL-3 (like most of the project)
#
# For fast_insert_image, see this for more background:
# https://github.com/pymupdf/PyMuPDF/issues/1408

from internetarchivepdf.const import COMPRESSOR_JPEG, COMPRESSOR_JPEG2000, \
        COMPRESSOR_JBIG2


JPX_TEMPL = """<<
  /Type /XObject
  /Subtype /Image
  /BitsPerComponent 8
  /Width &width
  /Height &height
  /ColorSpace /&colourspace
  /Length &len
>>"""

JPEG_TEMPL = """<<
  /Type /XObject
  /Subtype /Image
  /BitsPerComponent 8
  /Width &width
  /Height &height
  /ColorSpace /&colourspace
  /Length &len
>>"""

JBIG2_TEMPL = """<<
  /Type /XObject
  /Subtype /Image
  /BitsPerComponent 1
  /Width &width
  /Height &height
  /ColorSpace /DeviceGray
  /Length &len
>>"""


def jpx_string(stream=None, width=0, height=0, gray=True):
    if any((stream == None, width == 0, height == 0)):
        raise ValueError("invalid args")
    jpx = (
        JPX_TEMPL.replace("&width", str(width))
        .replace("&height", str(height))
        .replace("&colourspace", 'DeviceGray' if gray else 'DeviceRGB')
        .replace("&len", str(len(stream)))
    )
    return jpx


def jpg_string(stream=None, width=0, height=0, gray=True):
    if any((stream == None, width == 0, height == 0)):
        raise ValueError("invalid args")
    jpg = (
        JPX_TEMPL.replace("&width", str(width))
        .replace("&height", str(height))
        .replace("&colourspace", 'DeviceGray' if gray else 'DeviceRGB')
        .replace("&len", str(len(stream)))
    )
    return jpg


def jbig2_string(stream=None, width=0, height=0):
    if any((stream == None, width == 0, height == 0)):
        raise ValueError("invalid args")
    jbig2 = (
        JBIG2_TEMPL.replace("&width", str(width))
        .replace("&height", str(height))
        .replace("&len", str(len(stream)))
    )
    return jbig2


def fast_insert_image(page, rect=None, width=0, height=0, stream=None,
                      mask=None, stream_fmt=COMPRESSOR_JPEG2000,
                      mask_fmt=COMPRESSOR_JBIG2, gray=True):
    """Fast image insertion

    Args:
    * page: output fitz.Page
    * rect: rectangle to use
    * width: image width
    * height: image height
    * stream: image stream
    * mask: mask image stream (if any)
    * stream_fmt: COMPRESSOR_JPEG2000 or COMPRESSOR_JPEG
    * mask_fmt: COMPRESSOR_JBIG2 or None
    * gray: if the image is grayscale (otherwise RGB is assumed)
    """
    # We encode jbig2 ourselves using jbig2enc, we can't do that for ccitt
    # currently, so we rely on mupdf to do it for us, so let's not support that
    # in this code path now
    if mask_fmt not in (COMPRESSOR_JBIG2,):
        raise ValueError('mask_fmt can only be jbig2')

    # We can't handle other formats (yet)
    if stream_fmt not in (COMPRESSOR_JPEG, COMPRESSOR_JPEG2000):
        raise ValueError('stream_fmt can only be jpeg or jpeg2000')

    doc = page.parent
    nxref = doc.get_new_xref()  # make image xref in output page
    xref_stream = stream
    mask_stream = mask

    # Make object string for target page
    if stream_fmt == COMPRESSOR_JPEG2000:
        jpx_obj = jpx_string(stream=xref_stream, width=width, height=height,
                             gray=gray)
    elif stream_fmt == COMPRESSOR_JPEG:
        jpx_obj = jpg_string(stream=xref_stream, width=width, height=height,
                             gray=gray)

    doc.update_object(nxref, jpx_obj)  # give it the object definition

    # give it the image stream - unchanged compression
    doc.update_stream(nxref, stream=xref_stream, new=True, compress=False)

    # adjust image definition with correct compression info
    # this must happen AFTER stream insertion!
    if stream_fmt == COMPRESSOR_JPEG2000:
        doc.xref_set_key(nxref, "Filter", "/JPXDecode")
    elif stream_fmt == COMPRESSOR_JPEG:
        doc.xref_set_key(nxref, "Filter", "/DCTDecode")

    # if input image had a mask, we need further adjustments ...
    if mask_stream:
        nmask = doc.get_new_xref()  # need another xref in target doc

        # make smask object definition
        mask_obj = jbig2_string(stream=mask_stream, width=width, height=height)
        # and put it in mask object xref
        doc.update_object(nmask, mask_obj)

        # now insert raw mask image stream
        doc.update_stream(nmask, stream=mask_stream, new=True, compress=False)

        # and also adjust the compression filer ... AFTER stream insertion
        doc.xref_set_key(nmask, "Filter", "/JBIG2Decode")

        # we also need to tell the main image that it has a mask:
        doc.xref_set_key(nxref, "SMask", "%i 0 R" % nmask)

    # now we are ready to insert the image
    return page.insert_image(rect, xref=nxref)
