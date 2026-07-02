#!/usr/bin/env python3
"""
ADRES decoder (corrected v2)
------------------------------------------------------------------
Reads the two lossless PNG streams written by encode.cpp -- the
full-resolution ROI canvas (zero outside the ROI) and the sub-sampled
background -- and fuses them using the transmitted block-level ROI
mask. Matches Section 4.2 of the paper: bicubic sub-sampling of ROI
and background, PNG lossless compression of the quantized data.
"""
import io
import sys

import numpy as np
from PIL import Image

BLOCK_SIZE = 16


def read_compressed(filename):
    with open(filename, 'rb') as f:
        magic = f.readline().decode().strip()
        if magic != "PROTO2v3":
            raise ValueError(f"Format invalide (attendu PROTO2v3, reçu {magic})")

        width, height = map(int, f.readline().decode().split())
        q_roi, q_bg, subsample = map(int, f.readline().decode().split())
        bw, bh = map(int, f.readline().decode().split())
        roi_png_size = int(f.readline().decode().strip())
        bg_w, bg_h, bg_png_size = map(int, f.readline().decode().split())
        marker = f.readline().decode().strip()
        if marker != "DATA":
            raise ValueError(f"Marqueur DATA manquant (reçu: {marker})")

        roi_mask = np.frombuffer(f.read(bw * bh), dtype=np.uint8).reshape(bh, bw)
        roi_png_bytes = f.read(roi_png_size)
        bg_png_bytes = f.read(bg_png_size)

    return roi_mask, roi_png_bytes, bg_png_bytes, (width, height, q_roi, q_bg, subsample, bw, bh, bg_w, bg_h)


def reconstruct_image(roi_mask, roi_png_bytes, bg_png_bytes, params):
    width, height, q_roi, q_bg, subsample, bw, bh, bg_w, bg_h = params
    del q_roi, q_bg, subsample, bg_w, bg_h  # kept in header for traceability only

    roi_canvas = np.array(Image.open(io.BytesIO(roi_png_bytes)).convert('RGB'))
    bg_small = Image.open(io.BytesIO(bg_png_bytes)).convert('RGB')
    bg_full = np.array(bg_small.resize((width, height), Image.BICUBIC))

    roi_full_mask = np.zeros((height, width), dtype=bool)
    for by in range(bh):
        for bx in range(bw):
            if roi_mask[by, bx]:
                y0, y1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, height)
                x0, x1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, width)
                roi_full_mask[y0:y1, x0:x1] = True

    reconstructed = bg_full.copy()
    reconstructed[roi_full_mask] = roi_canvas[roi_full_mask]

    return reconstructed


def main():
    if len(sys.argv) != 4:
        print("Usage: decode.py <compressed.p2> <output.png> <output_dir>")
        sys.exit(1)

    roi_mask, roi_png_bytes, bg_png_bytes, params = read_compressed(sys.argv[1])
    reconstructed = reconstruct_image(roi_mask, roi_png_bytes, bg_png_bytes, params)

    Image.fromarray(reconstructed, 'RGB').save(sys.argv[2], "PNG")
    print(f"Reconstruction terminée: {sys.argv[2]}")


if __name__ == "__main__":
    main()
