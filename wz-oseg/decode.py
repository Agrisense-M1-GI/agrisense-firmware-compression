#!/usr/bin/env python3
"""
WZ-OSEG decoder (corrected v2)
------------------------------------------------------------------
Reverses the RLE + canonical Huffman entropy coding added in encode.cpp
to recover the exact same quantized DCT coefficient array as v1, then
reuses the original block-wise IDCT reconstruction unchanged.
"""
import struct
import sys

import numpy as np
from PIL import Image
from scipy.fftpack import idct

BLOCK_SIZE = 8
ESCAPE_MARKER = -32768


# ------------------------------------------------------------------ #
# Canonical Huffman reconstruction
# ------------------------------------------------------------------ #
def build_canonical_codes(entries):
    """entries: list of (symbol, length). Returns {(length, code): symbol}."""
    entries = sorted(entries, key=lambda e: (e[1], e[0]))
    decode_map = {}
    code = 0
    for i, (sym, length) in enumerate(entries):
        decode_map[(length, code)] = sym
        code += 1
        if i + 1 < len(entries) and entries[i + 1][1] > length:
            code <<= (entries[i + 1][1] - length)
    return decode_map


def huffman_decode(packed, total_bits, token_count, decode_map):
    tokens = []
    code = 0
    length = 0
    bit_index = 0
    for byte in packed:
        for shift in range(7, -1, -1):
            if bit_index >= total_bits or len(tokens) == token_count:
                break
            bit = (byte >> shift) & 1
            code = (code << 1) | bit
            length += 1
            bit_index += 1
            key = (length, code)
            if key in decode_map:
                tokens.append(decode_map[key])
                code = 0
                length = 0
        if len(tokens) == token_count:
            break
    if len(tokens) != token_count:
        raise ValueError(
            f"Décodage Huffman incomplet: {len(tokens)}/{token_count} tokens récupérés"
        )
    return tokens


def rle_decode(tokens, coeff_count):
    coeffs = np.zeros(coeff_count, dtype=np.int16)
    ci = 0
    i = 0
    n = len(tokens)
    while i < n and ci < coeff_count:
        if tokens[i] == ESCAPE_MARKER:
            run = tokens[i + 1]
            i += 2
            end = min(ci + run, coeff_count)
            coeffs[ci:end] = 0
            ci = end
        else:
            coeffs[ci] = tokens[i]
            ci += 1
            i += 1
    return coeffs


# ------------------------------------------------------------------ #
# File reading
# ------------------------------------------------------------------ #
def read_compressed(filename):
    with open(filename, 'rb') as f:
        magic = f.readline().decode().strip()
        if magic != "PROTO1_WZv2":
            raise ValueError(f"Format invalide (attendu PROTO1_WZv2, reçu {magic})")

        width, height = map(int, f.readline().decode().split())
        bw, bh = map(int, f.readline().decode().split())
        coeff_count = int(f.readline().decode().strip())
        token_count = int(f.readline().decode().strip())
        nsym = int(f.readline().decode().strip())
        total_bits = int(f.readline().decode().strip())
        table_marker = f.readline().decode().strip()
        if table_marker != "HUFFTABLE":
            raise ValueError(f"Table Huffman manquante (marqueur reçu: {table_marker})")

        entries = []
        for _ in range(nsym):
            sym = struct.unpack('<h', f.read(2))[0]
            length = f.read(1)[0]
            entries.append((sym, length))

        data_marker = f.readline().decode().strip()
        if data_marker != "DATA":
            raise ValueError(f"Marqueur DATA manquant (reçu: {data_marker})")

        size = width * height
        otsu_data = np.frombuffer(f.read(size // 4), dtype=np.uint8)
        hue_data = np.frombuffer(f.read(size // 4), dtype=np.uint8)

        packed_bytes = (total_bits + 7) // 8
        packed = f.read(packed_bytes)

    decode_map = build_canonical_codes(entries)
    tokens = huffman_decode(packed, total_bits, token_count, decode_map)
    dct_coeffs = rle_decode(tokens, coeff_count)

    return otsu_data, hue_data, dct_coeffs, (width, height, bw, bh, coeff_count)


# ------------------------------------------------------------------ #
# Reconstruction (unchanged from v1)
# ------------------------------------------------------------------ #
def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def reconstruct_from_dct_sparse(dct_coeffs, width, height, bw, bh, side_info):
    reconstructed = side_info.copy().astype(float)

    coeff_idx = 0
    coeffs_per_block = 21  # triangle x+y < 6 in an 8x8 block

    for by in range(bh):
        for bx in range(bw):
            block_coeffs = dct_coeffs[coeff_idx:coeff_idx + coeffs_per_block]
            coeff_idx += coeffs_per_block

            if coeff_idx > len(dct_coeffs):
                break

            block_dct = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
            c_idx = 0
            for y in range(BLOCK_SIZE):
                for x in range(BLOCK_SIZE):
                    if x + y < 6:
                        q = 8 if (x + y < 3) else 16
                        block_dct[y, x] = block_coeffs[c_idx] * q
                        c_idx += 1

            y_start = by * BLOCK_SIZE
            y_end = min((by + 1) * BLOCK_SIZE, height)
            x_start = bx * BLOCK_SIZE
            x_end = min((bx + 1) * BLOCK_SIZE, width)

            try:
                block_spatial = idct_2d(block_dct) + 128
                block_spatial = np.clip(block_spatial, 0, 255)
            except Exception:
                block_spatial = side_info[y_start:y_end, x_start:x_end]

            h_block = y_end - y_start
            w_block = x_end - x_start

            alpha = 0.6  # 60% DCT, 40% side info
            reconstructed[y_start:y_end, x_start:x_end] = (
                alpha * block_spatial[:h_block, :w_block] +
                (1 - alpha) * side_info[y_start:y_end, x_start:x_end]
            )

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def gray_to_rgb(gray_channel):
    return np.stack([gray_channel, gray_channel, gray_channel], axis=2)


def main():
    if len(sys.argv) != 5:
        print("Usage: decode.py <compressed.p1> <side_info.ppm> <output.png> <output_dir>")
        sys.exit(1)

    side_img = Image.open(sys.argv[2])
    side_array = np.array(side_img)
    side_gray = (0.299 * side_array[:, :, 0] +
                 0.587 * side_array[:, :, 1] +
                 0.114 * side_array[:, :, 2]).astype(np.uint8)

    otsu_data, hue_data, dct_coeffs, params = read_compressed(sys.argv[1])
    width, height, bw, bh, coeff_count = params

    y_reconstructed = reconstruct_from_dct_sparse(dct_coeffs, width, height, bw, bh, side_gray)
    rgb_reconstructed = gray_to_rgb(y_reconstructed)

    img_out = Image.fromarray(rgb_reconstructed, 'RGB')
    img_out.save(sys.argv[3], "PNG")

    print(f"Reconstruction terminée: {sys.argv[3]}")


if __name__ == "__main__":
    main()
