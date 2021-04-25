# Tool to extract each MRC layer and give it its own page in the outfile pdf.
import fitz
import sys

from io import BytesIO

from PIL import Jpeg2KImagePlugin, Image

if len(sys.argv) < 2:
    print('Usage: python mrcview.py infile outfile')


doc = fitz.open(sys.argv[1])
newdoc = fitz.open()

def add_image_page(inpage, outdoc, img_raw, stream=True):
    p = outdoc.newPage(width=inpage.rect.width, height=inpage.rect.height)
    p.insertImage(p.rect, stream=img_raw)


for idx, page in enumerate(doc):
    imgs = doc.getPageImageList(pno=page.number)

    for imgidx, img in enumerate(imgs):
        img_xref = img[0]
        img_maskxref = img[1]

        image = doc.xrefStreamRaw(img_xref)

        add_image_page(page, newdoc, image)
        if img_maskxref > 0:
            mask = doc.extractImage(img_maskxref)
            stream = mask['image']
            add_image_page(page, newdoc, stream)


newdoc.save(sys.argv[2], deflate=True, pretty=True)
