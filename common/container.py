"""
common/container.py
====================
Binary container format for algo/jpeg-qveg-qbg (and algo/jpeg-qveg-qbg-4x1x4):
transmits the block map (must be counted in output_size_bytes -- it's
required to reconstruct the image, so it's part of the real cost) plus
the two compact JPEG streams (vegetation, background). bw/bh are NOT
transmitted -- they're deterministically recomputed from orig_w/orig_h
on the station side (ceil(dim / 8)), avoiding a few redundant bytes.

Layout (all integers little-endian uint32):
    [orig_w][orig_h][len_mask][mask bytes][len_veg][veg jpg][len_bg][bg jpg]
"""

import struct

_HEADER = struct.Struct("<I")  # one little-endian uint32


def pack_qveg_qbg(orig_w: int, orig_h: int, mask_bytes: bytes,
                   veg_jpg: bytes, bg_jpg: bytes) -> bytes:
    parts = [
        _HEADER.pack(orig_w),
        _HEADER.pack(orig_h),
        _HEADER.pack(len(mask_bytes)), mask_bytes,
        _HEADER.pack(len(veg_jpg)), veg_jpg,
        _HEADER.pack(len(bg_jpg)), bg_jpg,
    ]
    return b"".join(parts)


def unpack_qveg_qbg(data: bytes) -> dict:
    offset = 0

    def read_u32():
        nonlocal offset
        (value,) = _HEADER.unpack_from(data, offset)
        offset += 4
        return value

    def read_bytes(n):
        nonlocal offset
        chunk = data[offset:offset + n]
        offset += n
        return chunk

    orig_w = read_u32()
    orig_h = read_u32()
    mask_len = read_u32()
    mask_bytes = read_bytes(mask_len)
    veg_len = read_u32()
    veg_jpg = read_bytes(veg_len)
    bg_len = read_u32()
    bg_jpg = read_bytes(bg_len)

    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "mask_bytes": mask_bytes,
        "veg_jpg": veg_jpg,
        "bg_jpg": bg_jpg,
    }
