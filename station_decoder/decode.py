"""
station_decoder/decode.py — algo/jpeg-baseline
=================================================
Branch-specific decoder, called generically by the standalone station
receiver (same pattern as startup.py -> pipeline.py on the Pi: the
runner doesn't know or care which branch is checked out, it just calls
this module's decode()).

Standard JPEG -> decode with THIS branch's own compiled djpeg
(common/config.py: DJPEG_BIN), the same IJG-9e build used to encode.
"""

import os
import subprocess
import tempfile

import cv2

from common import config


def decode(compressed_bytes: bytes):
    """Returns a BGR numpy array decoded from compressed_bytes via djpeg."""
    if not os.path.exists(config.DJPEG_BIN):
        raise RuntimeError(
            f"djpeg not found at {config.DJPEG_BIN} -- build it first: "
            f"compression/lib/build.sh"
        )

    with tempfile.TemporaryDirectory() as tmp:
        jpg_path = os.path.join(tmp, "input.jpg")
        ppm_path = os.path.join(tmp, "output.ppm")
        with open(jpg_path, "wb") as f:
            f.write(compressed_bytes)

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
        return img_bgr
