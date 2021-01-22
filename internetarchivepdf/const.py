VERSION = '0.0.1'

SOFTWARE_URL = 'https://git.archive.org/merlijn/archive-pdf-tools'
PRODUCER = 'Internet Archive PDF creator and recoder %s; %s; written by '\
           'Merlijn B. W. Wajer. Powered by Tesseract, mupdf and '\
           'Python (pymupdf/skimage).' % (VERSION, SOFTWARE_URL)


IMAGE_MODE_PASSTHROUGH = 0
IMAGE_MODE_PIXMAP = 1
IMAGE_MODE_MRC = 2
IMAGE_MODE_SKIP = 3


RECODE_RUNTIME_WARNING_INVALID_PAGE_SIZE = 'invalid-page-size'
RECODE_RUNTIME_WARNING_INVALID_PAGE_NUMBERS = 'invalid-page-numbers'

RECODE_RUNTIME_WARNINGS = {
    RECODE_RUNTIME_WARNING_INVALID_PAGE_SIZE,
    RECODE_RUNTIME_WARNING_INVALID_PAGE_NUMBERS,
}
