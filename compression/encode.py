"""
compression/encode.py — algo/jpeg-baseline
===========================================
Standard JPEG (libjpeg IJG v9e), quality 75, subsampling 4:2:0.
This is Experience A of the ablation study (Section 4.3): the baseline
against which every other configuration is compared.

cjpeg (from this branch's private copy of jpeg-9e, see compression/lib/)
does not read PNG directly — it reads PPM/BMP/GIF/Targa. We write a
temporary lossless PPM, call cjpeg, read back the compressed bytes, and
clean up. This conversion step is lossless, so it does not affect the
comparison.
"""

import os
import subprocess
import tempfile
import time

import cv2

from common import config

# Human-readable name shown in logs/README, and used by pipeline.py to
# decide behaviour that differs between branches (see common/config.py).
BRANCH_NAME = "algo/jpeg-baseline"


def encode(img_bgr) -> tuple[bytes, dict]:
    """
    Compresses a BGR image (numpy array) with this branch's cjpeg.

    Returns (compressed_bytes, extra_metrics) where extra_metrics is a
    dict of branch-specific fields worth keeping around (empty here,
    since baseline has no extra parameters beyond quality/subsampling,
    already fixed by the protocol).
    """
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = os.path.join(tmp, "input.ppm")
        jpg_path = os.path.join(tmp, "output.jpg")

        # Lossless intermediate format cjpeg can actually read.
        cv2.imwrite(ppm_path, img_bgr)

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
            compressed_bytes = f.read()

    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}
