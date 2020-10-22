# Straight port for tesseract/src/api/pdfrenderer.cpp
# at commit 2d6f38eebf9a14d9fbe65d785f0d7bd898ff46cb
#
# Please look at the code in there for documentation, etc.
# TODO: Code is probably Apache Licensed as well

from math import atan, atan2, cos, sin
import numpy as np
import zlib

## BEGIN EXTRA DEFS
# TODO: fix these
WRITING_DIRECTION_RIGHT_TO_LEFT = 42
WRITING_DIRECTION_TOP_TO_BOTTOM = 43
WRITING_DIRECTION_LEFT_TO_RIGHT = 0
## END EXTRA DEFS

K_CHAR_WIDTH = 2
K_MAX_BYTES_PER_CODEPOINTS = 20

class TessPDFRenderer(object):

    def __init__(self, textonly=True, image_list=None):
        self.textonly = textonly

        self._obj = 0
        self._offsets = [0]

        self._pages = []

        self._data = b''

    def AppendPDFObjectDIY(self, object_size):
        self._offsets.append(object_size + self._offsets[-1])
        self._obj += 1

    def AppendPDFObject(self, data):
        self.AppendPDFObjectDIY(len(data))
        self.AppendString(data)

    def AppendString(self, s):
        self._data += s

    def AppendData(self, s):
        self._data += s

    def GetPDFTextObjects(self, word_data, width, height, ppi):
        # Stub values
        old_x = 0.0
        old_y = 0.0
        old_fontsize = 0
        old_writing_direction = WRITING_DIRECTION_LEFT_TO_RIGHT
        new_block = True
        fontsize = 0
        a = 1.
        b = 0.
        c = 0.
        d = 1.

        pdf_str = bytes()
        pdf_str += b'q ' + floatbytes(prec(width), prec=3) + b' 0 0 ' + floatbytes(prec(height), prec=3) + b' 0 0 cm'

        if not self.textonly:
            pdf_str += b' /Im1 Do'

        pdf_str += b' Q\n';

        line_x1 = 0
        line_y1 = 0
        line_x2 = 0
        line_y2 = 0

        for paragraph in word_data:
            # TODO: change this to 3 to make text invisible again
            pdf_str += b'BT\n0 Tr'
            #pdf_str += b'BT\n3 Tr'
            old_fontsize = 0
            new_block = True

            for line in paragraph['lines']:
                first_word_of_line = True
                for word in line['words']:
                    if first_word_of_line:
                        x1, y1, x2, y2 = line['bbox']

                        # TODO: I am not sure if this baseline code makes sense yet
                        slope, constant = line['baseline']
                        angle = atan(slope)
                        diff = cos(angle) * constant
                        diff = diff * 72 / ppi
                        diff = (y2-y1) - diff

                        #print('slope:', slope, 'constant:', constant)
                        #print('pts:', x1, y1, x2, y2)
                        #print('diff:', diff, 'normdiff:', y2-y1)

                        line_x1, line_y1, line_x2, line_y2 = \
                                ClipBaseline(ppi, x1, y1, x2, y2 - diff)
                                #ClipBaseline(ppi, x1, y1, x2, y2)

                        # TODO: Get writing direction from hOCR files (see hocrrenderer.cpp)
                        writing_direction = WRITING_DIRECTION_LEFT_TO_RIGHT

                    word_x1, word_y1, word_x2, word_y2 = word['bbox']

                    word_height = word_y2 - word_y1

                    x, y, word_length = GetWordBaseline(writing_direction, ppi, height,
                            word_x1, word_y1, word_x2, word_y2,
                            line_x1, line_y1 + word_height,
                            line_x2, line_y2 + word_height)

                    if (writing_direction != old_writing_direction) or new_block:
                        a, b, c, d = \
                                AffineMatrix(writing_direction, line_x1, line_y1, line_x2, line_y2)
                        pdf_str += (b' ' + floatbytes(prec(a)) +
                                    b' ' + floatbytes(prec(b)) +
                                    b' ' + floatbytes(prec(c)) +
                                    b' ' + floatbytes(prec(d)) +
                                    b' ' + floatbytes(prec(x)) +
                                    b' ' + floatbytes(prec(y)) +
                                    b' Tm ')

                        new_block = False
                    else:
                        dx = x - old_x
                        dy = y - old_y
                        pdf_str += b' ' + floatbytes(prec(dx * a + dy * b))
                        pdf_str += b' ' + floatbytes(prec(dx * c + dy * d))
                        pdf_str += b' Td '

                        first_word_of_line = False

                    old_x = x;
                    old_y = y;
                    old_writing_direction = writing_direction

                    fontsize = word['fontsize']
                    kDefaultFontsize = 8;
                    if fontsize <= 0:
                        fontsize = kDefaultFontsize

                    if fontsize != old_fontsize:
                        pdf_str += b'/f-0-0 ' + str(fontsize).encode('ascii') + b' Tf ';
                        old_fontsize = fontsize;

                    pdf_word = b''
                    pdf_word_len = 0

                    for char in word['text']:
                        codepoint = ord(char)
                        ok, utf16 = CodepointToUtf16be(codepoint)
                        if ok:
                            pdf_word += utf16
                            pdf_word_len += 1

                    if True: # res_is->IsAtBeginningOf(RIL_WORD)
                        pdf_word += b'0020'
                        pdf_word_len += 1

                    if word_length > 0 and pdf_word_len > 0:
                        h_stretch = K_CHAR_WIDTH * prec(100.0 * word_length / (fontsize * pdf_word_len))
                        pdf_str += floatbytes(h_stretch) + b' Tz'
                        pdf_str += b' [ <' + pdf_word
                        pdf_str += b'> ] TJ'

                # Last word in the line
                pdf_str += b' \n';

            # Last line the block
            pdf_str += b'ET\n'


        return pdf_str

    def BeginDocumentHandler(self):
        self.AppendPDFObject(b'%PDF-1.5\n%\xDE\xAD\xBE\xEB\n');
        self.AppendPDFObject(b'1 0 obj\n'
                             b'<<\n'
                             b'  /Type /Catalog\n'
                             b'  /Pages 2 0 R\n'
                             b'>>\nendobj\n')
        self.AppendPDFObject(b'');
        self.AppendPDFObject(b'3 0 obj\n'
                             b'<<\n'
                             b'  /BaseFont /GlyphLessFont\n'
                             b'  /DescendantFonts [ 4 0 R ]\n'
                             b'  /Encoding /Identity-H\n'
                             b'  /Subtype /Type0\n'
                             b'  /ToUnicode 6 0 R\n'
                             b'  /Type /Font\n'
                             b'>>\n'
                             b'endobj\n')

        self.AppendPDFObject(b'4 0 obj\n'
                             b"<<\n"
                             b"  /BaseFont /GlyphLessFont\n"
                             b"  /CIDToGIDMap 5 0 R\n"
                             b"  /CIDSystemInfo\n"
                             b"  <<\n"
                             b"     /Ordering (Identity)\n"
                             b"     /Registry (Adobe)\n"
                             b"     /Supplement 0\n"
                             b"  >>\n"
                             b"  /FontDescriptor 7 0 R\n"
                             b"  /Subtype /CIDFontType2\n"
                             b"  /Type /Font\n"
                             b"  /DW " + str(1000 // K_CHAR_WIDTH).encode('ascii') + b'\n' +
                             b">>\n"
                             b"endobj\n")

        kCIDToGIDMapSize = 2 * (1 << 16);
        cidtogidmap = np.ndarray(kCIDToGIDMapSize, dtype='<u1')
        cidtogidmap[:] = 0
        cidtogidmap[1::2] = 1

        compressed = zlib.compress(cidtogidmap.tobytes())
        complen = len(compressed)
        stream = bytes()

        stream += b'5 0 obj\n'
        stream += b'<<\n'
        stream += b'  /Length ' + str(complen).encode('ascii') + b' /Filter /FlateDecode\n'
        stream += b'>>\n'
        stream += b'stream\n'
        self.AppendString(stream)

        objsize = len(stream)
        self.AppendString(compressed)
        objsize += complen
        endstream_obj = b'endstream\nendobj\n'
        self.AppendString(endstream_obj)
        objsize += len(endstream_obj)
        self.AppendPDFObjectDIY(objsize)

        stream2 = (b'/CIDInit /ProcSet findresource begin\n'
                  b'12 dict begin\n'
                  b'begincmap\n'
                  b'/CIDSystemInfo\n'
                  b'<<\n'
                  b'  /Registry (Adobe)\n'
                  b'  /Ordering (UCS)\n'
                  b'  /Supplement 0\n'
                  b'>> def\n'
                  b'/CMapName /Adobe-Identify-UCS def\n'
                  b'/CMapType 2 def\n'
                  b'1 begincodespacerange\n'
                  b'<0000> <FFFF>\n'
                  b'endcodespacerange\n'
                  b'1 beginbfrange\n'
                  b'<0000> <FFFF> <0000>\n'
                  b'endbfrange\n'
                  b'endcmap\n'
                  b'CMapName currentdict /CMap defineresource pop\n'
                  b'end\n'
                  b'end\n')

        stream = b'6 0 obj\n'
        stream += b'<< /Length ' + str(len(stream2)).encode('ascii') + b' >>\n'
        stream += b'stream\n' + stream2
        stream += b'endstream\n'
        stream += b'endobj\n'
        self.AppendPDFObject(stream)

        stream = (
          b'7 0 obj\n'
          b'<<\n'
          b'  /Ascent 1000\n'
          b'  /CapHeight 1000\n'
          b'  /Descent -1\n'
          b'  /Flags 5\n'
          b'  /FontBBox  [ 0 0 ' + str(1000 // K_CHAR_WIDTH).encode('ascii') + b' 1000 ]\n'
          b'  /FontFile2 8 0 R\n'
          b'  /FontName /GlyphLessFont\n'
          b'  /ItalicAngle 0\n'
          b'  /StemV 80\n'
          b'  /Type /FontDescriptor\n'
          b'>>\n'
          b'endobj\n')
        self.AppendPDFObject(stream)

        fontstream = open('pdf.ttf', 'rb').read()
        stream = (
          b'8 0 obj\n'
          b'<<\n'
          b'  /Length ' + str(len(fontstream)).encode('ascii') + b'\n'
          b'  /Length1 ' + str(len(fontstream)).encode('ascii') + b'\n'
          b'>>\n'
          b'stream\n')
        self.AppendString(stream)
        objsize  = len(stream)
        self.AppendData(fontstream)
        objsize += len(fontstream)
        self.AppendString(endstream_obj)
        objsize += len(endstream_obj)
        self.AppendPDFObjectDIY(objsize)

    def EndDocumentHandler(self):
        kPagesObjectNumber = 2

        self._offsets[kPagesObjectNumber] = self._offsets[-1]

        stream = bytes()
        stream += str(kPagesObjectNumber).encode('ascii') + b' 0 obj\n<<\n  /Type /Pages\n  /Kids [ '
        self.AppendString(stream)

        pages_objsize = len(stream)
        for i in range(len(self._pages)):
            stream = bytes()
            stream += str(self._pages[i]).encode('ascii') + b' 0 R '
            self.AppendString(stream)
            pages_objsize += len(stream)

        stream = bytes()
        stream += b']\n  /Count ' + str(len(self._pages)).encode('ascii') + b'\n>>\nendobj\n'
        self.AppendString(stream)
        pages_objsize += len(stream)
        self._offsets[-1] += pages_objsize

        utf16_title = b'FEFF'
        title = "TEST" # FIXME
        for c in title:
            codepoint = ord(c)
            ok, utf16 = CodepointToUtf16be(codepoint)
            if not ok:
                raise Exception('Wait what')
            if ok:
                utf16_title += utf16

        stream = bytes()
        stream += (
                str(self._obj).encode('ascii') + b' 0 obj\n'
                b'<<\n'
                b'  /Producer (Tesseract ' + b'4.1.1' + b')\n'  # XXX: TODO: version
                b'  /CreationDate (D:' + b'TODO' + b')\n'
                b'  /Title <' + utf16_title + b'>\n'
                b'>>\n'
                b'endobj\n')
        self.AppendPDFObject(stream)
        stream = bytes()
        stream += b'xref\n0 ' + str(self._obj).encode('ascii') + b'\n0000000000 65535 f \n';
        self.AppendString(stream)
        for i in range(1, self._obj):
            stream = bytes()
            stream += ('%.10d' % self._offsets[i]).encode('ascii') + b' 00000 n \n'
            self.AppendString(stream)

        stream = bytes()
        stream += (b'trailer\n<<\n  /Size ' + str(self._obj).encode('ascii') + b'\n'
                   b'  /Root 1 0 R\n'
                   b'  /Info ' + str(self._obj - 1).encode('ascii') + b' 0 R\n'
                   b'>>\nstartxref\n' + str(self._offsets[-1]).encode('ascii') +
                   b'\n%%EOF\n')
        self.AppendString(stream)

    def AddImageHandler(self, word_data, width, height, ppi):
        xobject = bytes()
        stream = bytes()

        if False: # if !textonly
            xobject += b'/XObject << /Im1 ' + str((obj_ + 2)).encode('ascii') + b' 0 R >>\n'

        stream += (
          str(self._obj).encode('ascii') + b' 0 obj\n'
          b'<<\n'
          b'  /Type /Page\n'
          b'  /Parent 2 0 R\n'
          b'  /MediaBox [0 0 ' + str(width).encode('ascii')+ b' ' + str(height).encode('ascii') + b']\n'
          b'  /Contents ' + str(self._obj + 1).encode('ascii') + b' 0 R\n'
          b'  /Resources\n'
          b'  <<\n'
          b'    ' + xobject +
          b'    /ProcSet [ /PDF /Text /ImageB /ImageI /ImageC ]\n'
          b'    /Font << /f-0-0 3 0 R >>\n'
          b'  >>\n'
          b'>>\n'
          b'endobj\n')
        self._pages.append(self._obj)
        self.AppendPDFObject(stream)

        pdftext = self.GetPDFTextObjects(word_data, width, height, ppi)
        comp_pdftext = zlib.compress(pdftext)
        stream = bytes()

        stream += (
          str(self._obj).encode('ascii') + b' 0 obj\n'
          b'<<\n'
          b'  /Length ' + str(len(comp_pdftext)).encode('ascii') + b' /Filter /FlateDecode\n'
          b'>>\n'
          b'stream\n')
        self.AppendString(stream)
        objsize = len(stream)

        self.AppendData(comp_pdftext)
        objsize += len(comp_pdftext)

        b2 = (b"endstream\n"
             b"endobj\n")
        self.AppendString(b2)
        objsize += len(b2)
        self.AppendPDFObjectDIY(objsize)

        if False: # textonly
            pass

        return True




# Helper function to prevent from writing scientific notation to PDF file
def prec(x):
    kPrecision = 1000.0
    a = round(x * kPrecision) / kPrecision
    if a == -0:
        return 0.
    return a


def dist2(x1, y1, x2, y2):
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)


def GetWordBaseline(writing_direction, ppi, height,
                    word_x1, word_y1, word_x2, word_y2,
                    line_x1, line_y1, line_x2, line_y2):
    if (writing_direction == WRITING_DIRECTION_RIGHT_TO_LEFT):
        tmp = word_x1
        word_x1 = word_x2
        word_x2 = tmp

        tmp = word_y1
        word_y1 = word_y2
        word_y2 = tmp

    word_length = 0.
    x = 0.
    y = 0.
    px = word_x1
    py = word_y1
    l2 = float(dist2(line_x1, line_y1, line_x2, line_y2))
    if (l2 == 0):
        x = line_x1
        y = line_y1
    else:
          t = ((px - line_x2) * (line_x2 - line_x1) +
                      (py - line_y2) * (line_y2 - line_y1)) / l2
          x = line_x2 + t * (line_x2 - line_x1)
          y = line_y2 + t * (line_y2 - line_y1)

    word_length = float(dist2(word_x1, word_y1, word_x2, word_y2) ** 0.5)
    word_length = word_length * 72.0 / ppi
    x = x * 72 / ppi
    y = height - (y * 72.0 / ppi)

    return x, y, word_length

def AffineMatrix(writing_direction, line_x1, line_y1, line_x2, line_y2):
    theta = atan2(float(line_y1 - line_y2), float(line_x2 - line_x1))
    a = cos(theta)
    b = sin(theta)
    c = -sin(theta)
    d = cos(theta)

    if writing_direction == WRITING_DIRECTION_RIGHT_TO_LEFT:
        a = -a;
        b = -b;
    elif writing_direction == WRITING_DIRECTION_TOP_TO_BOTTOM:
        # From original Tess code: TODO(jbreiden) Consider using the vertical PDF writing mode.
        pass
    else:
        pass
        #raise Exception('Unknown writing direction!')

    return a, b, c, d


def ClipBaseline(ppi, x1, y1, x2, y2):
    line_x1 = x1
    line_y1 = y1
    line_x2 = x2
    line_y2 = y2
    rise = abs(y2 - y1) * 72
    run = abs(x2 - x1) * 72
    if (rise < 2 * ppi) and (2 * ppi < run):
        line_y1 = line_y2 = (y1 + y2) / 2

    return line_x1, line_y1, line_x2, line_y2


def CodepointToUtf16be(code):
    res = None

    if (((code > 0xD7FF) and (code < 0xE000)) or (code > 0x10FFFF)):
        tprintf("Dropping invalid codepoint %d\n", code);
        return False, res

    if (code < 0x10000):
        res = '%04X' % code
    else:
        a = code - 0x010000
        high_surrogate = (0x03FF & (a >> 10)) + 0xD800
        low_surrogate = (0x03FF & a) + 0xDC00
        res = '%04X04X' % (high_surrogate, low_surrogate)

    return True, res.encode('ascii')


def floatbytes(v, prec=8):
    fmt_str = '{:.%df}' % prec
    return fmt_str.format(v).encode('ascii')


# XXX: Perhaps move this parsing elsewhere
from lxml import etree, html
import re

BBOX_REGEX = re.compile(r'bbox((\s+\d+){4})')
BASELINE_REGEX = re.compile(r'baseline((\s+[\d\.\-]+){2})')
X_SIZE_REGEX = re.compile(r'x_size((\s+[\d\.\-]+){1})')
X_FSIZE_REGEX = re.compile(r'x_fsize((\s+[\d\.\-]+){1})')

def hocr_page_iterator(hocrfile):
    hocr = etree.parse(hocrfile, html.XHTMLParser())
    hocr_pages = hocr.xpath("//*[@class='ocr_page']")
    for hocr_page in hocr_pages:
        pagebox = BBOX_REGEX.search(hocr_page.attrib['title']).group(1).split()
        w, h = int(pagebox[2]), int(pagebox[3])

        yield hocr_page, (w,h )

def hocr_to_word_data(hocr_page):
    paragraphs = []

    for par in hocr_page.xpath('.//*[@class="ocr_par"]'):
        paragraph_data = {'lines': []}

        for line in par.getchildren():
            line_data = {}

            linebox = BBOX_REGEX.search(line.attrib['title']).group(1).split()
            baseline = BASELINE_REGEX.search(line.attrib['title'])
            if baseline is not None:
                baseline = baseline.group(1).split()
            else:
                baseline = [0, 0]

            linebox = [float(i) for i in linebox]
            baseline = [float(i) for i in baseline]

            line_data['bbox'] = linebox
            line_data['baseline'] = baseline

            word_data = []
            for word in line.xpath('.//*[@class="ocrx_word"]'):
                # XXX: if no ocrx_cinfo, then just read word.text
                rawtext = ''
                for char in word.xpath('.//*[@class="ocrx_cinfo"]'):
                    rawtext += char.text

                box = BBOX_REGEX.search(word.attrib['title']).group(1).split()
                box = [float(i) for i in box]

                f_sizeraw = X_FSIZE_REGEX.search(word.attrib['title'])
                if f_sizeraw:
                    x_fsize = float(f_sizeraw.group(1))
                else:
                    x_fsize = 0. # Will get fixed later on

                # TODO: writing direction
                word_data.append({'bbox': box, 'text': rawtext, 'fontsize': x_fsize})


            line_data['words'] = word_data
            #print('Line words:', word_data)
            paragraph_data['lines'].append(line_data)

        paragraphs.append(paragraph_data)

    return paragraphs


if __name__ == '__main__':
    # TODO improve
    import sys
    hocrfile = sys.argv[1]

    render = TessPDFRenderer()

    render.BeginDocumentHandler()

    scaler = 1

    PPI = 72

    #idx = 0
    for page, (width, height) in hocr_page_iterator(hocrfile):
        width /= scaler
        height /= scaler
        ppi = PPI * scaler
        word_data = hocr_to_word_data(page)
        render.AddImageHandler(word_data, width, height, ppi=ppi)
        #idx += 1
        #if idx > 2:
        #    break

    render.EndDocumentHandler()

    fp = open('tessout.pdf', 'wb+')
    fp.write(render._data)
    fp.close()

