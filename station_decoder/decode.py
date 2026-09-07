"""
station_decoder/decode.py — algo/agrijpeg-light
===================================================
Reverses encode.py: rans_jpeg_codec transcode-decode (rANS -> standard
Huffman JPEG, byte-identical quantized coefficients) then plain djpeg
(no roi_jpeg_codec, no tiling -- there was never more than one JPEG
here). bw/n_class_blocks come from CAPTURE_WIDTH/CAPTURE_HEIGHT, not
from anything transmitted.

Contract: decode() returns (image, mask) like every other branch, mask
is always None here (no segmentation).
"""

import math
import os
import subprocess
import tempfile

import cv2

from common import config


def decode(compressed_bytes: bytes):
    bw = config.CAPTURE_WIDTH // 8
    bh = config.CAPTURE_HEIGHT // 8
    n_class_blocks = bw * bh

    if not os.path.exists(config.RANS_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"rans_jpeg_codec not found at {config.RANS_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )
    if not os.path.exists(config.DJPEG_BIN):
        raise RuntimeError(
            f"djpeg not found at {config.DJPEG_BIN} -- build it first: "
            f"compression/lib/build.sh"
        )

    with tempfile.TemporaryDirectory() as tmp:
        jpg_path = _transcode_decode_to_jpg(compressed_bytes, bw, n_class_blocks, tmp)

        ppm_path = os.path.join(tmp, "out.ppm")
        result = subprocess.run(
            [config.DJPEG_BIN, "-outfile", ppm_path, jpg_path], capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"djpeg failed (code {result.returncode}): "
                f"{result.stderr.decode(errors='replace')}"
            )

        img_bgr = cv2.imread(ppm_path)
        if img_bgr is None:
            raise RuntimeError("djpeg produced a PPM OpenCV could not read.")

    return img_bgr, None


def _transcode_decode_to_jpg(compressed_bytes: bytes, bw: int, n_class_blocks: int, tmp_dir: str) -> str:
    rans_path = os.path.join(tmp_dir, "in.rans")
    with open(rans_path, "wb") as f:
        f.write(compressed_bytes)

    jpg_path = os.path.join(tmp_dir, "out.jpg")
    cmd = [
        config.RANS_JPEG_CODEC_BIN, "transcode-decode",
        rans_path, str(bw), str(n_class_blocks), config.QTABLE_LIGHT_PATH, jpg_path,
        "--sample", config.JPEG_SAMPLE_FACTORS,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rans_jpeg_codec transcode-decode failed (code {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return jpg_path


def save_example(compressed_bytes, decoded_bgr, mask, output_dir: str, image_id: str) -> str:
    """
    Re-derives the intermediate standard JPEG (transcode-decode's own
    output, before the final djpeg->pixels step) and saves it as-is --
    a real, directly-openable JPEG, no re-encoding.
    """
    bw = config.CAPTURE_WIDTH // 8
    bh = config.CAPTURE_HEIGHT // 8
    n_class_blocks = bw * bh

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, image_id + ".jpg")
    with tempfile.TemporaryDirectory() as tmp:
        jpg_path = _transcode_decode_to_jpg(compressed_bytes, bw, n_class_blocks, tmp)
        with open(jpg_path, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
    return path
