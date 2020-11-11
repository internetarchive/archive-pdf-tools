# Script to generate definitions for CJK fonts

import fitz
d = fitz.open()
p = d.insertPage(0)
d[0].insertFont('china-s')
d.save('/tmp/cjk.pdf')
