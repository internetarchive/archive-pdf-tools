#!/bin/bash
set -e
echo "This is not a production tool, just a debug tool!"

img=$1
prefix=$2

#time python tools/threshold-image --with-pregaussblur $img /tmp/$2-mask.png
#time python tools/threshold-image --with-pregaussblur $img /tmp/$2-maski.png --invert-mask
time python tools/threshold-image --with-invert-image --with-pregaussblur $img /tmp/$2-mask.png
time python tools/threshold-image --with-invert-image --with-pregaussblur $img /tmp/$2-maski.png --invert-mask

tools/partial-blur --sigma 3 $img /tmp/$2-maski.png /tmp/$2-bg.png
tools/partial-blur --sigma 3 $img /tmp/$2-mask.png /tmp/$2-fg.png

tools/compress /tmp/$2-mask.png /tmp/$2-fg.png /tmp/$2-bg.png /tmp/$2-fg.jp2 /tmp/$2-bg.jp2
#tools/compress /tmp/$2-mask.png /tmp/$2-fg.png /tmp/$2-bg.png /tmp/$2-fg.jp2 /tmp/$2-bg.jp2 --bg-downscale 3


tools/merge /tmp/$2-mask.png /tmp/$2-fg.jp2 /tmp/$2-bg.jp2 /tmp/$2-z-combined.png
#tools/merge /tmp/$2-mask.png /tmp/$2-fg.png /tmp/$2-bg.png /tmp/$2-z-combined.png
