#!/usr/bin/env bash
# Builds this branch's private copy of jpeg-9e.
# Run once after copying your jpeg-9e/ source folder into compression/lib/.
set -e

cd "$(dirname "$0")/jpeg-9e"

if [ ! -f Makefile ]; then
    echo "Configuring jpeg-9e..."
    ./configure
fi

echo "Building cjpeg and djpeg..."
make cjpeg djpeg

echo
echo "Done. Binaries:"
echo "  $(pwd)/cjpeg"
echo "  $(pwd)/djpeg"
