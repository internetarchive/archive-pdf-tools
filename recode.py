import fitz
import sys

from pdfrenderer import TessPDFRenderer, hocr_page_iterator, hocr_to_word_data

STOP = 10000

in_doc = fitz.open(sys.argv[1])
hocr_iter = hocr_page_iterator(sys.argv[2])
outfile = sys.argv[3]

# TODO: read scandata and use it to skip pages

# Create text-only PDF first, and stick images on it
render = TessPDFRenderer()
render.BeginDocumentHandler()
for idx, page in enumerate(in_doc):
    hocr_page, (w, h) = hocr_iter.__next__()
    print('Page width, height:', page.rect.width, page.rect.height)
    print('hocr width, height:', w, h)

    scaler = page.rect.width / w
    print('scaler:', scaler)

    width = page.rect.width
    height = page.rect.height
    ppi = 72 / scaler

    word_data = hocr_to_word_data(hocr_page)
    render.AddImageHandler(word_data, width, height, ppi=ppi)
    if idx > STOP:
        break

render.EndDocumentHandler()

fp = open('/tmp/tess.pdf', 'wb+')
fp.write(render._data)
fp.close()

outdoc = fitz.open('/tmp/tess.pdf')


mode = 2

for idx, page in enumerate(outdoc):
    print('IDX:', idx)
    # TODO: pixmaps support colourspaces, so let's see if we can get those set
    # somehow.

    img = sorted(in_doc.getPageImageList(idx))[idx]
    xref = img[0]
    maskxref = img[1]
    if mode == 0:
        image = in_doc.extractImage(xref)
        page.insertImage(page.rect, stream=image["image"])
    elif mode == 1:
        pixmap = fitz.Pixmap(in_doc, xref)
        page.insertImage(page.rect, pixmap=pixmap)
    elif mode == 2:
        # mrc
        image = in_doc.extractImage(xref)
        jpx = image["image"]
        fp = open('/tmp/img.jpx', 'wb+')
        fp.write(jpx)
        fp.close()

        from mrc import KDU_EXPAND, create_mrc_components, encode_mrc_images
        from os import remove
        import subprocess
        from PIL import Image
        subprocess.check_call([KDU_EXPAND, '-i', '/tmp/img.jpx', '-o', '/tmp/in.tiff'])
        mask, bg, fg = create_mrc_components(Image.open('/tmp/in.tiff'))
        mask_f, bg_f, fg_f = encode_mrc_images(mask, bg, fg)

        bg_contents = open(bg_f, 'rb').read()
        page.insertImage(page.rect, stream=bg_contents, mask=None)

        fg_contents = open(fg_f, 'rb').read()
        mask_contents = open(mask_f, 'rb').read()

        page.insertImage(page.rect, stream=fg_contents, mask=mask_contents)
        remove(mask_f)
        remove(bg_f)
        remove(fg_f)

    if idx > STOP:
        break

print('Metadata:', in_doc.metadata)
doc_md = in_doc.metadata

#Metadata: {'format': 'PDF 1.5', 'title': None, 'author': None, 'subject': None, 'keywords': None, 'creator': None, 'producer': None, 'creationDate': None, 'modDate': None, 'encryption': None}

doc_md['producer'] = 'Internet Archive Recoder 0.0.1' # TODO
outdoc.setMetadata(doc_md)

xmlxref = outdoc._getNewXref()
stream=b'''<?xpacket begin="..." id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xmp:CreateDate>2020-10-15T01:06:14+00:00</xmp:CreateDate>
      <xmp:MetadataDate>2020-10-15T01:06:14+00:00</xmp:MetadataDate>
      <xmp:ModifyDate>2020-10-15T01:06:14+00:00</xmp:ModifyDate>
      <xmp:CreatorTool>Internet Archive</xmp:CreatorTool>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">a delightful collection of various items for testing</rdf:li>
        </rdf:Alt>
      </dc:title>
      <dc:creator>
        <rdf:Seq>
          <rdf:li>Example, Joe</rdf:li>
        </rdf:Seq>
      </dc:creator>
      <dc:language>
        <rdf:Bag>
          <rdf:li>en</rdf:li>
        </rdf:Bag>
      </dc:language>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="r"?> '''

outdoc.updateObject(xmlxref, '<<\n  /Type /Metadata\n/Subtype /XML>>')
outdoc.updateStream(xmlxref, stream, new=True)


catalogxref = outdoc.PDFCatalog()
print('catalogxref:', catalogxref)
s = outdoc.xrefObject(outdoc.PDFCatalog())
s = s[:-2]
s += '  /Metadata %d 0 R' % xmlxref
s += '>>'
outdoc.updateObject(catalogxref, s)


#trailerxref = outdoc.PDFTrailer()
#print('trailerxref:', trailerxref)
#s = outdoc.xrefObject(outdoc.PDFTrailer())
#s = s[:-2]
#s += '  /ID [ <5326FB76C88D9B75E16E613188ACE1B5> <5326FB76C88D9B75E16E613188ACE1B5> ]'
#s += '>>'
#outdoc.updateObject(catalogxref, s)

# TODO: For ID writing, we can just open the PDF and fix the trailer manually,
# all we need to do is insert a literal string with two hashes.

print(fitz.TOOLS.mupdf_warnings())
outdoc.save(outfile, deflate=True)
