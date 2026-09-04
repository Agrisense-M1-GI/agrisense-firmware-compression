"""
compression/encode.py — algo/agrijpeg-core
=============================================
Same tiling as jpeg-qveg-qbg-4x1x4 (roi_jpeg_codec: mask -> veg/bg
canvases -> Qveg/Qbg + 4:1:4 JPEG), plus one more step: each compact
JPEG's Huffman entropy stage is replaced by rANS (rans_jpeg_codec,
calibrated static tables, rans_tables.h). The container transmits the
.rans bytes, NOT the .jpg bytes.
"""

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from common import config, container, vari


def encode(img_bgr) -> tuple[bytes, dict]:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = os.path.join(tmp, "input.ppm")
        cv2.imwrite(ppm_path, img_bgr)
        compressed_bytes = _encode_core(img_bgr, ppm_path, tmp)
    compression_time_ms = (time.time() - t0) * 1000.0
    return compressed_bytes, {"compression_time_ms": compression_time_ms}


def encode_from_ppm(ppm_path: str) -> tuple[bytes, dict]:
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
        raise ValueError(f"Image dimensions {w}x{h} are not multiples of {block_size}.")

    block_mask = vari.classify_blocks_composite(img_bgr, block_size=block_size)
    mask_bytes = block_mask.astype(np.uint8).tobytes()
    bh, bw = block_mask.shape

    blockmap_path = os.path.join(tmp_dir, "blockmap.bin")
    with open(blockmap_path, "wb") as f:
        f.write(mask_bytes)

    veg_jpg = os.path.join(tmp_dir, "veg.jpg")
    bg_jpg = os.path.join(tmp_dir, "bg.jpg")

    if not os.path.exists(config.ROI_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"roi_jpeg_codec not found at {config.ROI_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )

    cmd = [
        config.ROI_JPEG_CODEC_BIN, "encode",
        ppm_path, blockmap_path, str(bw), str(bh),
        config.QVEG_TABLE_PATH, config.QBG_TABLE_PATH,
        veg_jpg, bg_jpg,
        "--sample", config.ROI_JPEG_SAMPLE_FACTORS,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"roi_jpeg_codec encode failed (code {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    # --- rANS transcode: Huffman -> rANS on each compact JPEG stream ---
    if not os.path.exists(config.RANS_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"rans_jpeg_codec not found at {config.RANS_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )

    veg_rans = os.path.join(tmp_dir, "veg.rans")
    bg_rans = os.path.join(tmp_dir, "bg.rans")
    for src, dst in [(veg_jpg, veg_rans), (bg_jpg, bg_rans)]:
        result = subprocess.run(
            [config.RANS_JPEG_CODEC_BIN, "transcode-encode", src, dst],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rans_jpeg_codec transcode-encode failed on {src} "
                f"(code {result.returncode}): {result.stderr.decode(errors='replace')}"
            )

    with open(veg_rans, "rb") as f:
        veg_bytes = f.read()
    with open(bg_rans, "rb") as f:
        bg_bytes = f.read()

    return container.pack_qveg_qbg(w, h, mask_bytes, veg_bytes, bg_bytes)
