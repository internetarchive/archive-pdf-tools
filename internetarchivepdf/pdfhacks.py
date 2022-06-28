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
# Functions to (more quickly) create and modify PDFs, as well as some other low
# level PDF writing code.
#
# Some of this code is contributed by Jorj X. McKie <jorj.x.mckie@outlook.de>
# License in this file if AGPL-3 (like most of the project)
#
# For fast_insert_image, see this for more background:
# https://github.com/pymupdf/PyMuPDF/issues/1408

import pkg_resources
from math import ceil
from datetime import datetime
from xml.sax.saxutils import escape as xmlescape

from internetarchivepdf.const import COMPRESSOR_JPEG, COMPRESSOR_JPEG2000, \
        COMPRESSOR_JBIG2, PRODUCER, RECODE_RUNTIME_WARNING_INVALID_PAGE_NUMBERS
from internetarchivepdf.pagenumbers import parse_series, series_to_pdf
from internetarchivepdf.scandata import scandata_xml_get_page_numbers


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


# XXX: tmp.icc - pick proper one and ship it with the tool, or embed it
def write_pdfa(to_pdf):
    srgbxref = to_pdf.get_new_xref()
    to_pdf.update_object(srgbxref, """
<<
      /Alternate /DeviceRGB
      /N 3
>>
""")
    icc = pkg_resources.resource_string('internetarchivepdf', "data/tmp.icc")
    to_pdf.update_stream(srgbxref, icc, new=True)

    intentxref = to_pdf.get_new_xref()
    to_pdf.update_object(intentxref, """
<<
  /Type /OutputIntent
  /S /GTS_PDFA1
  /OutputConditionIdentifier (Custom)
  /Info (sRGB IEC61966-2.1)
  /DestOutputProfile %d 0 R
>>
""" % srgbxref)

    catalogxref = to_pdf.pdf_catalog()
    s = to_pdf.xref_object(to_pdf.pdf_catalog())
    s = s[:-2]
    s += '  /OutputIntents [ %d 0 R ]' % intentxref
    s += '>>'
    to_pdf.update_object(catalogxref, s)


def write_page_labels(to_pdf, scandata, errors=None):
    page_numbers = scandata_xml_get_page_numbers(scandata)
    res, all_ok = parse_series(page_numbers)

    # Add warning/error
    if errors is not None and not all_ok:
        errors.add(RECODE_RUNTIME_WARNING_INVALID_PAGE_NUMBERS)

    catalogxref = to_pdf.pdf_catalog()
    s = to_pdf.xref_object(to_pdf.pdf_catalog())
    s = s[:-2]
    s += series_to_pdf(res)
    s += '>>'
    to_pdf.update_object(catalogxref, s)



def write_basic_ua(to_pdf, language=None):
    # Create StructTreeRoot and descendants, allocate new xrefs as needed
    structtreeroot_xref = to_pdf.get_new_xref()
    parenttree_xref = to_pdf.get_new_xref()
    page_info_xrefs = []
    page_info_a_xrefs = []
    parenttree_kids_xrefs = []
    parenttree_kids_indirect_xrefs = []

    kids_cnt = ceil(to_pdf.page_count / 32)
    for _ in range(kids_cnt):
        kid_xref = to_pdf.get_new_xref()
        parenttree_kids_xrefs.append(kid_xref)

    # Parent tree contains a /Kids entry with a list of xrefs, that each contain
    # a list of xrefs (limited to 32 per), and each entry in that list of list
    # of xrefs contains a single reference that points to the page info xref.
    for idx, page in enumerate(to_pdf):
        page_info_xref = to_pdf.get_new_xref()
        page_info_xrefs.append(page_info_xref)

        page_info_a_xref = to_pdf.get_new_xref()
        page_info_a_xrefs.append(page_info_a_xref)

        parenttree_kids_indirect_xref = to_pdf.get_new_xref()
        parenttree_kids_indirect_xrefs.append(parenttree_kids_indirect_xref)


    for idx in range(kids_cnt):
        start = idx*32
        stop = (idx+1)*31
        if stop > to_pdf.page_count:
            stop = to_pdf.page_count- 1

        s = """<<
  /Limits [ %d %d ]
""" % (start, stop - 1)
        s += '  /Nums [ '

        for pidx in range(start, stop):
            s += '%d %d 0 R ' % (pidx, parenttree_kids_indirect_xrefs[pidx])

            if idx % 7 == 0:
                s = s[:-1] + '\n' + '      '

        s += ']\n>>'

        to_pdf.update_object(parenttree_kids_xrefs[idx], s)


    for idx, page in enumerate(to_pdf):
        intrect = tuple([int(x) for x in page.rect])

        s = """<<
  /BBox [ %d %d %d %d ]
  /InlineAlign /Center
  /O /Layout
  /Placement /Block
>>
""" % intrect
        to_pdf.update_object(page_info_a_xrefs[idx], s)

        s = """ <<
  /A %d 0 R
  /K 0
  /P %d 0 R
  /Pg %d 0 R
  /S /Figure
>>""" % (page_info_a_xrefs[idx], structtreeroot_xref, page.xref)

        to_pdf.update_object(page_info_xrefs[idx], s)


    for idx, page in enumerate(to_pdf):
        s = '[ %d 0 R ]' % page_info_a_xrefs[idx]
        to_pdf.update_object(parenttree_kids_indirect_xrefs[idx], s)


    K = '  /Kids [ '
    for idx in range(kids_cnt):
        K += '%d 0 R ' % parenttree_kids_xrefs[idx]

        if idx % 7 == 0:
            K = K[:-1] + '\n' + '      '

    K += ']'
    s = """<<
%s
>>
""" % K

    to_pdf.update_object(parenttree_xref, s)

    K = '  /K [ '
    for idx, xref in enumerate(page_info_xrefs):
        K += '%d 0 R ' % xref

        if idx % 7 == 0:
            K = K[:-1] + '\n' + '      '

    K += ']'

    to_pdf.update_object(structtreeroot_xref, """
<<
""" + K + """
  /Type /StructTreeRoot
  /ParentTree %d 0 R
>>
""" % parenttree_xref)

    #  TODO? /ClassMap 1006 0 R
    #  TODO? /ParentTreeNextKey 198


    # Update pages, add back xrefs
    for idx, page in enumerate(to_pdf):
        page_data = to_pdf.xref_object(page.xref)
        page_data = page_data[:-2]

        page_data += """
  /StructParents %d
""" % idx

        page_data += """
  /CropBox [ 0 0 %.1f %.1f ]
""" % (page.rect[2], page.rect[3])

        page_data += """
  /Rotate 0
"""
        page_data += """
  /Tabs /S
"""
        page_data += '>>'
        to_pdf.update_object(page.xref, page_data)

    catalogxref = to_pdf.pdf_catalog()
    s = to_pdf.xref_object(to_pdf.pdf_catalog())
    s = s[:-2]
    s += """
  /ViewerPreferences <<
    /FitWindow true
    /DisplayDocTitle true
  >>
"""
    if language:
        s += """
  /Lang (%s)
""" % language

    s += """
  /MarkInfo <<
    /Marked true
  >>
"""
    s += """
  /StructTreeRoot %d 0 R
""" % structtreeroot_xref

    s += '>>'
    to_pdf.update_object(catalogxref, s)


def write_metadata(from_pdf, to_pdf, extra_metadata):
    """
    Write document and XMP metadata.

    Args:

    * from_pdf (fitz.Document or None): metadata to copy from input PDF, can be omitted
    * to_pdf: (fitz.Document): PDF to write metadata to
    * extra_metadata (dict): dictionary with extra metadata values

    Allowed values for extra_metadata:

    * 'url'
    * 'title'
    * 'author'
    * 'creator'
    * 'subject'
    * 'creatortool'
    * 'language' (can be a list)
    """
    doc_md = from_pdf.metadata if from_pdf is not None else {}

    doc_md['producer'] = PRODUCER

    if 'url' in extra_metadata:
        doc_md['keywords'] = extra_metadata['url']
    if 'title' in extra_metadata:
        doc_md['title'] = extra_metadata['title']
    if 'author' in extra_metadata:
        doc_md['author'] = extra_metadata['author']
    if 'creator' in extra_metadata:
        doc_md['creator'] = extra_metadata['creator']
    if 'subject' in extra_metadata:
        doc_md['subject'] = extra_metadata['subject']

    current_time = 'D:' + datetime.utcnow().strftime('%Y%m%d%H%M%SZ')
    if from_pdf is not None:
        doc_md['creationDate'] = from_pdf.metadata['creationDate']
    else:
        doc_md['creationDate'] = current_time
    doc_md['modDate'] = current_time

    # Set PDF basic metadata
    to_pdf.set_metadata(doc_md)

    have_xmlmeta = (from_pdf is not None) and (from_pdf.xref_xml_metadata() > 0)
    if have_xmlmeta:
        xml_xref = from_pdf.xref_xml_metadata()

        # Just copy the existing XML, perform no validity checks
        xml_bytes = from_pdf.xref_stream(xml_xref)
        to_pdf.set_xml_metadata(xml_bytes.decode('utf-8'))
    else:
        current_time = datetime.utcnow().isoformat(timespec='seconds') + 'Z'

        stream='''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
              <xmp:CreateDate>{createdate}</xmp:CreateDate>
              <xmp:MetadataDate>{metadatadate}</xmp:MetadataDate>
              <xmp:ModifyDate>{modifydate}</xmp:ModifyDate>
              <xmp:CreatorTool>{creatortool}</xmp:CreatorTool>'''.format(creatortool=xmlescape(extra_metadata.get('creatortool', PRODUCER)),
           createdate=current_time, metadatadate=current_time,
           modifydate=current_time)

        stream += '''
            </rdf:Description>
            <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">'''

        if 'url' in extra_metadata:
            stream += '''
              <pdf:Keywords>{keywords}</pdf:Keywords>'''.format(keywords=xmlescape(extra_metadata['url']))

        stream += '''
              <pdf:Producer>{producer}</pdf:Producer>'''.format(producer=xmlescape(PRODUCER))

        stream += '''
            </rdf:Description>
            <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'''

        if extra_metadata.get('title'):
            stream += '''
              <dc:title>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">{title}</rdf:li>
                </rdf:Alt>
              </dc:title>'''.format(title=xmlescape(extra_metadata.get('title')))

        # "An entity responsible for making the resource."
        # https://www.dublincore.org/specifications/dublin-core/dcmi-terms/#http://purl.org/dc/terms/creator
        # So should be author...
        if extra_metadata.get('author'):
            stream += '''
              <dc:creator>
                <rdf:Seq>
                  <rdf:li>{author}</rdf:li>
                </rdf:Seq>
              </dc:creator>'''.format(author=xmlescape(extra_metadata.get('author')))

        # TODO: Support multiple languages here?

        if extra_metadata.get('language'):
        # Empty language field means unknown language
            stream += '''
              <dc:language>
                <rdf:Bag>'''

            for language in extra_metadata.get('language', []):
                stream += '''
                  <rdf:li>{language}</rdf:li>'''.format(language=xmlescape(language))

            stream += '''
                </rdf:Bag>
              </dc:language>'''

        stream += '''
            </rdf:Description>
            <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
              <pdfaid:part>3</pdfaid:part>
              <pdfaid:conformance>B</pdfaid:conformance>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="r"?>'''

        to_pdf.set_xml_metadata(stream)
