#!/usr/bin/env bash
# Builds this branch's private copy of jpeg-9e, plus roi_jpeg_codec (the
# Qveg/Qbg tiling codec, algo/jpeg-qveg-qbg and algo/jpeg-qveg-qbg-4x1x4).
# Run once after copying your jpeg-9e/ source folder into compression/lib/.
set -e

cd "$(dirname "$0")"

if [ ! -f jpeg-9e/Makefile ]; then
    echo "Configuring jpeg-9e..."
    (cd jpeg-9e && ./configure)
fi

echo "Building cjpeg and djpeg..."
(cd jpeg-9e && make cjpeg djpeg)

echo
echo "Done. Binaries:"
echo "  $(pwd)/jpeg-9e/cjpeg"
echo "  $(pwd)/jpeg-9e/djpeg"

if [ -f roi_jpeg_codec.c ]; then
    echo
    echo "Building roi_jpeg_codec (linked against this branch's own jpeg-9e, not the system libjpeg)..."
    gcc -O2 -I jpeg-9e -o roi_jpeg_codec roi_jpeg_codec.c jpeg-9e/.libs/libjpeg.a
    echo "Done. Binary:"
    echo "  $(pwd)/roi_jpeg_codec"
fi
