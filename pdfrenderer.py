# Straight port for tesseract/src/api/pdfrenderer.cpp
# at commit 2d6f38eebf9a14d9fbe65d785f0d7bd898ff46cb
#
# Please look at the code in there for documentation, etc.
# TODO: Code is probably Apache Licensed as well
#
# Port:
# - BeginDocumentHandler
# - AddImageHandler(hOCR page res)
# - EndDocumentHandler
#
# Usage of TessBaseAPI:
# - getSourceYResolution (ppi)
# - Get Iterator for words//lines
# - GetInputImage
# - GetInputName
# - Get output jpg quality

# TODO:
# - Code can already output multiple PDFs, let's see if we can use that (I
# suppose it uses that in batch processing mode
# - Get DPI/PPI from somewhere

from math import atan2, cos, sin
import numpy as np
import zlib

## BEGIN EXTRA DEFS
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

    def GetPDFTextObjects(self, word_data, width, height):
        ppi = 400. # TODO: Get DPI from hOCR or from scandata, etc?

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
        #std::stringstream pdf_str;
        #// Use "C" locale (needed for double values prec()).
        #pdf_str.imbue(std::locale::classic());
        #// Use 8 digits for double values.
        #pdf_str.precision(8);

        pdf_str += b'q ' + floatbytes(prec(width), prec=3) + b' 0 0 ' + floatbytes(prec(height), prec=3) + b' 0 0 cm'

        if not self.textonly:
            pdf_str += b' /Im1 Do'

        pdf_str += b' Q\n';

        line_x1 = 0
        line_y1 = 0
        line_x2 = 0
        line_y2 = 0

        # ocr_carea == RIL_BLOCK I think

        # TESTING
        fontsize = 20
        pdf_str += b'BT\n0 Tr'
        #pdf_str += b'BT\n3 Tr'

        x = 100
        y = 100
        a, b, c, d = AffineMatrix(-9001, 100, 100, 100, 100)
        print(a,b,c,d)
        a,b,c,d =1,0,0,1
        pdf_str += b' ' + floatbytes(prec(a), prec=3)
        pdf_str += b' ' + floatbytes(prec(b), prec=3)
        pdf_str += b' ' + floatbytes(prec(c), prec=3)
        pdf_str += b' ' + floatbytes(prec(d), prec=3)
        pdf_str += b' ' + floatbytes(prec(x), prec=3)
        pdf_str += b' ' + floatbytes(prec(y), prec=3)
        pdf_str += b' Tm '
        pdf_str += b'/f-0-0 ' + str(fontsize).encode('ascii') + b' Tf ';

        word = 'HELLO'
        word_length = len(word)

        pdf_word = b''
        pdf_word_len = 0
        for char in word:
            codepoint = ord(char)
            ok, utf16 = CodepointToUtf16be(codepoint)
            if ok:
                pdf_word += utf16
                pdf_word_len += 1

        pdf_word += b'0020'
        pdf_word_len += 1

        h_stretch = K_CHAR_WIDTH * prec(100.0 * word_length / (fontsize * pdf_word_len))
        pdf_str += floatbytes(h_stretch) + b' Tz'
        pdf_str += b' [ <' + pdf_word
        pdf_str += b'> ] TJ'

        pdf_str += b' \n';
        pdf_str += b'ET\n'
        # END TESTING

        return pdf_str # XXX


        for word in ocr_careas:
            # I think this loop run on every word

            if word_is_at_beginning_of_carea:
                pdf_str += b'BT\n3 Tr'
                old_fontsize = 0
                new_block = True

            word_is_at_beginning_of_ril_textline = False # TODO: implement
            # baseline parsing
            if word_is_at_beginning_of_ril_textline:
                x1, y1, x2, y2 = GET_LINE_BASELINE
                line_x1, line_y1, line_x2, line_y2 = ClipBaseline(ppi, x1, y1,
                        x2, y2)

            # TODO: Get writing direction from hOCR files (see hocrrenderer.cpp)
            writing_direction = WRITING_DIRECTION_LEFT_TO_RIGHT

            if writing_direction != WRITING_DIRECTION_TOP_TO_BOTTOM:
                # TODO: get word writing direction (parse "dir-'ltr'" etc)
                # word_dir = get_our_word_direction()
                word_dir = DIR_LEFT_TO_RIGHT

                if word_dir == DIR_LEFT_TO_RIGHT:
                    writing_direction = WRITING_DIRECTION_LEFT_TO_RIGHT
                elif word_dir == DIR_RIGHT_TO_LEFT:
                    writing_direction = WRITING_DIRECTION_RIGHT_TO_LEFT
                else:
                    writing_direction = old_writing_direction

            x = 0.
            y = 0.
            word_length = 0.

            # XXX: MERLIJN: I think we can get away with getting the 
            # bottom of the word bounding box here instead, since the  function
            # GetWordBaseline projects all words onto the line baseline
            word_x1, word_y1, word_x2, word_y2 = get_baseline_for_word()
            x, y, word_length = GetWordBaseline(writing_direction, ppi, height,
                    word_x1, word_y1, word_x2, word_y2, line_x1, line_y1,
                    line_x2, line_y2)

            if (writing_direction != old_writing_direction) or new_block:
                a, b, c, d = AffineMatrix(writing_direction, line_x1, line_y1,
                        line_x2, line_y2)
                pdf_str += b' ' + floatbytes(prec(a))
                pdf_str += b' ' + floatbytes(prec(b))
                pdf_str += b' ' + floatbytes(prec(c))
                pdf_str += b' ' + floatbytes(prec(d))
                pdf_str += b' ' + floatbytes(prec(x))
                pdf_str += b' ' + floatbytes(prec(y))
                pdf_str += b' Tm '

                new_block = False
            else:
                dx = x - old_x
                dy = y - old_y
                pdf_str += b' ' + floatbytes(prec(dx * a + dy * b))
                pdf_str += b' ' + floatbytes(prec(dx * c + dy * d))
                pdf_str += b' Td '

            old_x = x;
            old_y = y;
            old_writing_direction = writing_direction;

            # // Adjust font size on a per word granularity. Pay attention to
            # // fontsize, old_fontsize, and pdf_str. We've found that for
            # // in Arabic, Tesseract will happily return a fontsize of zero,
            # // so we make up a default number to protect ourselves.
            fontsize = TODO_GET_FONTSIZE # = x_fsize on word
            kDefaultFontsize = 8;
            if fontsize <= 0:
                fontsize = kDefaultFontsize

            if fontsize != old_fontsize:
                pdf_str += b'/f-0-0 ' + str(fontsize).encode('ascii') + b' Tf ';
                old_fontsize = fontsize;

            last_word_in_line = False
            last_word_in_block = Fales
            #bool last_word_in_line = res_it->IsAtFinalElement(RIL_TEXTLINE, RIL_WORD);
            #bool last_word_in_block = res_it->IsAtFinalElement(RIL_BLOCK, RIL_WORD);

            pdf_word = b''
            pdf_word_len = 0

            for char in word:
                codepoint = ord(char)
                ok, utf16 = CodepointToUtf16be(codepoint)
                if ok:
                    pdf_word += utf16
                    pdf_word_len += 1

            if is_at_beginning_of_word:
                pdf_word += b'0020'
                pdf_word_len += 1

            if word_length > 0 and pdf_word_len > 0:
                h_stretch = kCharWidth * prec(100.0 * word_length / (fontsize * pdf_word_len))
                pdf_str += floatbytes(h_stretch) + b' Tz'
                pdf_str += b' [ <' + pdf_word
                pdf_str += b'> ] TJ'

            if last_word_in_line:
                pdf_str += b' \n';
            if last_word_in_block:
                pdf_str += 'ET\n'

        # XXX: done looping
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

    def AddImageHandler(self, word_data, image):
        # TODO: Let's just add a single page without image and without text, and
        # see if that can work with our document end handler as well.
        # Then look at hocr parsing and text placement next.
        ppi = 400. # TODO
        # TODO: get width/height from hOCR
        #width = image.size[0]
        #height = image.size[1]
        width = 500
        height = 500

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

        pdftext = self.GetPDFTextObjects(word_data, width, height)
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



# TODO: Reverse this transformation
"""
static void AddBaselineCoordsTohOCR(const PageIterator* it,
                                    PageIteratorLevel level,
                                    std::stringstream& hocr_str) {
  tesseract::Orientation orientation = GetBlockTextOrientation(it);
  if (orientation != ORIENTATION_PAGE_UP) {
    hocr_str << "; textangle " << 360 - orientation * 90;
    return;
  }

  int left, top, right, bottom;
  it->BoundingBox(level, &left, &top, &right, &bottom);

  // Try to get the baseline coordinates at this level.
  int x1, y1, x2, y2;
  if (!it->Baseline(level, &x1, &y1, &x2, &y2)) return;
  // Following the description of this field of the hOCR spec, we convert the
  // baseline coordinates so that "the bottom left of the bounding box is the
  // origin".
  x1 -= left;
  x2 -= left;
  y1 -= bottom;
  y2 -= bottom;

  // Now fit a line through the points so we can extract coefficients for the
  // equation:  y = p1 x + p0
  if (x1 == x2) {
    // Problem computing the polynomial coefficients.
    return;
  }
  double p1 = (y2 - y1) / static_cast<double>(x2 - x1);
  double p0 = y1 - p1 * x1;

  hocr_str << "; baseline " << round(p1 * 1000.0) / 1000.0 << " "
           << round(p0 * 1000.0) / 1000.0;
}
"""

if __name__ == '__main__':
    render = TessPDFRenderer()
    render.BeginDocumentHandler()
    render.AddImageHandler(None, None)
    render.EndDocumentHandler()

    fp = open('tessout.pdf', 'wb+')
    fp.write(render._data)
    fp.close()

