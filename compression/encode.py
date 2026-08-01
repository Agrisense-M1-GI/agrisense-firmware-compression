"""
compression/encode.py — algo/jpeg-qveg-qbg
=============================================
Tiling approach ("Option B", Section 4.3): the image is split into
vegetation/background 8x8 blocks (common.vari.classify_blocks_composite),
each class packed into its own compact rectangle and encoded as a
separate JPEG stream with its own quantization table (roi_jpeg_codec.c,
luminance only -- chroma stays standard). The block map is transmitted
alongside both streams (common.container) and counted in
output_size_bytes -- it is required to reconstruct the image on the
station side, so it is a real cost of this algorithm, not overhead.

Q_veg/Q_bg tables are PLACEHOLDERS (CALIBRATE, common/config.py) -- both
identical to the standard IJG luminance table until the sorted random
search calibration (Sampson et al., arXiv:2003.02874) is done.

Two entry points, same contract as every other branch:
- encode(img_bgr): NORMAL mode, live webcam frame.
- encode_from_ppm(ppm_path): TEST mode, dataset's lossless PPM twin.

Both entry points measure the FULL algorithm -- mask computation,
packing, and both JPEG encodes are all part of what "compressing with
qveg-qbg" means, so all of it sits inside the caller's energy
measurement window (unlike jpeg-baseline's gate/segmentation, which were
unrelated to JPEG itself).
"""

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from common import config, container, vari


def encode(img_bgr) -> tuple[bytes, dict]:
    """Compresses a live BGR frame (numpy array, NORMAL mode)."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = os.path.join(tmp, "input.ppm")
        cv2.imwrite(ppm_path, img_bgr)  # cv2 writes correct RGB byte order for .ppm
        compressed_bytes = _encode_core(img_bgr, ppm_path, tmp)
    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def encode_from_ppm(ppm_path: str) -> tuple[bytes, dict]:
    """
    Compresses an existing PPM file (TEST mode). Unlike jpeg-baseline,
    the pixel data still needs to be read into memory here -- computing
    the composite ROI mask requires pixel access, an unavoidable part of
    this branch's own compression algorithm, so it's measured too.
    """
    t0 = time.time()
    img_bgr = cv2.imread(ppm_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read {ppm_path}")
    with tempfile.TemporaryDirectory() as tmp:
        compressed_bytes = _encode_core(img_bgr, ppm_path, tmp)
    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def _encode_core(img_bgr, ppm_path: str, tmp_dir: str) -> bytes:
    h, w = img_bgr.shape[:2]
    block_size = config.QVEG_QBG_BLOCK_SIZE

    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Image dimensions {w}x{h} are not multiples of {block_size} -- "
            f"roi_jpeg_codec pads internally (edge replication) but "
            f"common.vari's block classification uses floor division and "
            f"would silently miss the remainder pixels. This branch assumes "
            f"multiple-of-{block_size} capture dimensions (Section 2.2: "
            f"1920x1080)."
        )

    # --- Composite ROI mask (block granularity, Section: common/vari.py) --
    block_mask = vari.classify_blocks_composite(img_bgr, block_size=block_size)
    mask_bytes = block_mask.astype(np.uint8).tobytes()  # row-major, 1=veg/0=bg
    bh, bw = block_mask.shape

    blockmap_path = os.path.join(tmp_dir, "blockmap.bin")
    with open(blockmap_path, "wb") as f:
        f.write(mask_bytes)

    veg_path = os.path.join(tmp_dir, "veg.jpg")
    bg_path = os.path.join(tmp_dir, "bg.jpg")

    cmd = [
        config.ROI_JPEG_CODEC_BIN, "encode",
        ppm_path, blockmap_path, str(bw), str(bh),
        config.QVEG_TABLE_PATH, config.QBG_TABLE_PATH,
        veg_path, bg_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"roi_jpeg_codec encode failed (code {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    with open(veg_path, "rb") as f:
        veg_jpg = f.read()
    with open(bg_path, "rb") as f:
        bg_jpg = f.read()

    return container.pack_qveg_qbg(w, h, mask_bytes, veg_jpg, bg_jpg)
