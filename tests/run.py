import sys
from os.path import join

import fitz

from internetarchivepdf.recode import recode
from internetarchivepdf.const import IMAGE_MODE_MRC, \
        JPEG2000_IMPL_KAKADU, COMPRESSOR_JPEG2000, DENOISE_NONE, DENOISE_FAST



from testimages import test_images



fg_slope = 44500
bg_slope = 44250

if __name__ == '__main__':
    out_dir = sys.argv[1]

    for test_image in test_images:
        out_file = join(out_dir, test_image['identifier'] + '.pdf')

        hocr_path = join('files', test_image['identifier'], test_image['hocr'])
        image_path = join('files', test_image['identifier'], test_image['glob'])

        recode(from_imagestack=image_path,
               hocr_file=hocr_path,
               dpi=test_image['dpi'],
               out_pdf=out_file,
               jbig2=True,
               bg_compression_flags=['-slope', str(bg_slope)],
               fg_compression_flags=['-slope', str(fg_slope)],
               bg_downsample=3,
               image_mode=IMAGE_MODE_MRC,
               mrc_image_format=COMPRESSOR_JPEG2000,
               jpeg2000_implementation=JPEG2000_IMPL_KAKADU,
               denoise_mask=DENOISE_FAST,
        )

        # Open pdf, create png from pdf page render, so that we can compare
        # against baseline
        png_path = join(out_dir, test_image['identifier'] + '.png') 
        doc = fitz.open(out_file)
        pix = doc[0].get_pixmap(matrix=fitz.Identity * (test_image['dpi'] / 72))
        pix.save(png_path)

