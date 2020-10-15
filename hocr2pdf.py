#!/usr/bin/env python
import argparse
import re

import fitz


# TODO: Perhaps not use lxml, but just python built-in functions.
from lxml import etree, html


FONT_SIZE = 8

BBOX_REGEX = re.compile(r'bbox((\s+\d+){4})')
BASELINE_REGEX = re.compile(r'baseline((\s+[\d\.\-]+){2})')
X_SIZE_REGEX = re.compile(r'x_size((\s+[\d\.\-]+){1})')
WORDCONF_REGEX = re.compile(r'x_wconf((\s+[\d\.\-]+){1})')

def create_pdf(hocrfile):
    doc = fitz.open()

    hocr = etree.parse(hocrfile, html.XHTMLParser())
    hocr_pages = hocr.xpath("//*[@class='ocr_page']")
    for hocr_page in hocr_pages:
        pagebox = BBOX_REGEX.search(hocr_page.attrib['title']).group(1).split()
        w, h = int(pagebox[2]), int(pagebox[3])

        pdfpage = doc.newPage(width=w, height=h)

        add_text_layer(pdfpage, hocr_page)

    return doc


def add_text_layer(doc, hocr_page):
    for par in hocr_page.xpath('.//*[@class="ocr_par"]'):

        x_sizes = []
        # Calculate size per paragraph
        for line in par.getchildren():
            for word in line.xpath('.//*[@class="ocrx_word"]'):
                x_size = float(X_SIZE_REGEX.search(line.attrib['title']).group(1))
                x_sizes.append(x_size)

        sz = (sum(x_sizes) / len(x_sizes)) / float(FONT_SIZE)


        for line in par.getchildren():
            linebox = BBOX_REGEX.search(line.attrib['title']).group(1).split()
            baseline = BASELINE_REGEX.search(line.attrib['title'])
            if baseline is not None:
                baseline = baseline.group(1).split()
            else:
                baseline = [0, 0]

            linebox = [float(i) for i in linebox]
            baseline = [float(i) for i in baseline]

            # TODO: Use same size per word
            rawtext = ''
            for word in line.xpath('.//*[@class="ocrx_word"]'):

                rawtext = ''
                for char in word.xpath('.//*[@class="ocrx_cinfo"]'):
                    rawtext += char.text

                box = BBOX_REGEX.search(word.attrib['title']).group(1).split()
                box = [int(i) for i in box]

                b = polyval(baseline, (box[0] + box[2]) / 2 - linebox[0]) + linebox[3]
                rect = fitz.Rect(box[0], b, box[2], b + box[3]-box[1])
                box_width = box[2] - box[0]
                spacing_per = box_width / len(rawtext)

                pt = fitz.Point(box[0], b)

                scale_matrix = fitz.Matrix(sz, sz)

                rot_matrix = fitz.Matrix(baseline[0])

                scale_matrix = scale_matrix * rot_matrix


                if rect[0] == rect[2] or rect[1] == rect[3]:
                    print('Skipping word (%s) due to invalid word box: %s' % (repr(rawtext), rect))
                    continue

                doc.insertTextbox(rect, rawtext,
                        fontsize=8,
                        morph=(pt, scale_matrix),
                        render_mode=0,
                        )#fontfile='/glyphless')




# from hocr-pdf from hocr-tools
def polyval(poly, x):
    return x * poly[0] + poly[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a PDF with a hidden text layer from a hOCR file')
    parser.add_argument("hocrfile",
        help=('hOCR file containing all the pages')
    )
    args = parser.parse_args()

    doc = create_pdf(args.hocrfile)
    doc.save('/tmp/hocr2pdf.pdf')
