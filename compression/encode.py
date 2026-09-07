"""
compression/encode.py — algo/agrijpeg-light
===============================================
No segmentation, no tiling, no mask -- a single standard JPEG (custom
global luminance table Q_light, calibrated via sorted random search
around the standard IJG table; 4:1:4 chroma sub-sampling, unchanged
from jpeg-4x1x4) with its Huffman stage replaced by rANS
(rans_jpeg_codec, same tool as agrijpeg-core).

compressed_bytes is the raw .rans bytes -- no container, no embedded
metadata at all. The station reconstructs bw/n_class_blocks from
CAPTURE_WIDTH/CAPTURE_HEIGHT (fixed, known on both sides), so nothing
beyond the .rans stream itself needs to be transmitted.
"""

import os
import subprocess
import tempfile
import time

import cv2

from common import config


def encode(img_bgr) -> tuple[bytes, dict]:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = os.path.join(tmp, "input.ppm")
        cv2.imwrite(ppm_path, img_bgr)
        compressed_bytes = _encode_core(ppm_path, tmp)
    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def encode_from_ppm(ppm_path: str) -> tuple[bytes, dict]:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        compressed_bytes = _encode_core(ppm_path, tmp)
    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def _encode_core(ppm_path: str, tmp_dir: str) -> bytes:
    jpg_path = os.path.join(tmp_dir, "out.jpg")

    cmd = [
        config.CJPEG_BIN,
        "-quality", str(config.JPEG_QUALITY),
        "-qtables", config.QTABLE_LIGHT_PATH,
        "-qslots", "0,1,1",
        "-sample", config.JPEG_SAMPLE_FACTORS,
        "-outfile", jpg_path,
        ppm_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cjpeg failed (code {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    if not os.path.exists(config.RANS_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"rans_jpeg_codec not found at {config.RANS_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )

    rans_path = os.path.join(tmp_dir, "out.rans")
    result = subprocess.run(
        [config.RANS_JPEG_CODEC_BIN, "transcode-encode", jpg_path, rans_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rans_jpeg_codec transcode-encode failed (code {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    with open(rans_path, "rb") as f:
        return f.read()
