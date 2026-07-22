"""
compression/encode.py — algo/jpeg-baseline
===========================================
Standard JPEG (libjpeg IJG v9e), quality 75, subsampling 4:2:0.
This is Experience A of the ablation study (Section 4.3): the baseline
against which every other configuration is compared.

Two entry points:
- encode(img_bgr): NORMAL mode, live webcam frame (numpy array) — no
  lossless twin exists on disk, so we still write a temporary PPM for
  cjpeg to read (it doesn't read PNG/raw arrays directly).
- encode_from_ppm(ppm_path): TEST mode — the reference dataset ships a
  lossless PPM twin next to every PNG specifically so this step can skip
  decode + re-encode entirely and hand cjpeg the existing file directly.

Branch name and compression parameters (quality, subsampling) are read
from common/config.py, the single source of truth for this branch.
"""

import os
import subprocess
import tempfile
import time

import cv2

from common import config


def encode(img_bgr) -> tuple[bytes, dict]:
    """
    Compresses a live BGR frame (numpy array, NORMAL mode) with this
    branch's cjpeg. Returns (compressed_bytes, extra_metrics) where
    extra_metrics is a dict of branch-specific fields worth keeping
    around (empty here, since baseline has no extra parameters beyond
    quality/subsampling, already fixed by the protocol).
    """
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = os.path.join(tmp, "input.ppm")
        jpg_path = os.path.join(tmp, "output.jpg")

        # Lossless intermediate format cjpeg can actually read.
        cv2.imwrite(ppm_path, img_bgr)

        compressed_bytes = _run_cjpeg(ppm_path, jpg_path)

    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def encode_from_ppm(ppm_path: str) -> tuple[bytes, dict]:
    """
    Compresses an existing PPM file (TEST mode) directly — no decode, no
    intermediate write. The dataset provides this PPM twin ahead of time
    for exactly this purpose, so the measured window only ever contains
    the compression step itself, not a PNG->PPM conversion.
    """
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        jpg_path = os.path.join(tmp, "output.jpg")
        compressed_bytes = _run_cjpeg(ppm_path, jpg_path)

    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def _run_cjpeg(ppm_path: str, jpg_path: str) -> bytes:
    cmd = [
        config.CJPEG_BIN,
        "-quality", str(config.JPEG_QUALITY),
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

    with open(jpg_path, "rb") as f:
        return f.read()
