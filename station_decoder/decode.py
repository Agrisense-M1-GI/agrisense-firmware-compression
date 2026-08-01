"""
station_decoder/decode.py — algo/jpeg-qveg-qbg
=================================================
Branch-specific decoder, called generically by the standalone station
receiver (repo_link.load_station_decoder). Unpacks the container
(common/container.py), decodes both compact JPEG streams with
roi_jpeg_codec (this branch's own, linked against the private jpeg-9e),
and reassembles the full image using the block map -- the one that was
TRANSMITTED, never recomputed from the reference dataset (a station
in real deployment never has that reference).

Contract: decode() returns (image, mask) -- mask is a 2D boolean array
(block granularity) so station_decoder/quality.py can zone PSNR/SSIM.
Branches without segmentation (e.g. algo/jpeg-baseline) return
(image, None) instead.
"""

import math
import os
import subprocess
import tempfile

import cv2
import numpy as np

from common import config, container


def decode(compressed_bytes: bytes):
    """Returns (image_bgr, block_mask) decoded via this branch's roi_jpeg_codec."""
    if not os.path.exists(config.ROI_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"roi_jpeg_codec not found at {config.ROI_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )

    parts = container.unpack_qveg_qbg(compressed_bytes)
    orig_w, orig_h = parts["orig_w"], parts["orig_h"]
    block_size = config.QVEG_QBG_BLOCK_SIZE
    bw = math.ceil(orig_w / block_size)
    bh = math.ceil(orig_h / block_size)

    with tempfile.TemporaryDirectory() as tmp:
        veg_path = os.path.join(tmp, "veg.jpg")
        bg_path = os.path.join(tmp, "bg.jpg")
        blockmap_path = os.path.join(tmp, "blockmap.bin")
        out_ppm = os.path.join(tmp, "out.ppm")

        with open(veg_path, "wb") as f:
            f.write(parts["veg_jpg"])
        with open(bg_path, "wb") as f:
            f.write(parts["bg_jpg"])
        with open(blockmap_path, "wb") as f:
            f.write(parts["mask_bytes"])

        cmd = [
            config.ROI_JPEG_CODEC_BIN, "decode",
            veg_path, bg_path, blockmap_path,
            str(bw), str(bh), str(orig_w), str(orig_h),
            out_ppm,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"roi_jpeg_codec decode failed (code {result.returncode}): "
                f"{result.stderr.decode(errors='replace')}"
            )

        img_bgr = cv2.imread(out_ppm)
        if img_bgr is None:
            raise RuntimeError("roi_jpeg_codec produced a PPM OpenCV could not read.")

    block_mask = np.frombuffer(parts["mask_bytes"], dtype=np.uint8).reshape(bh, bw).astype(bool)
    return img_bgr, block_mask


def save_example(compressed_bytes, decoded_bgr, mask, output_dir: str, image_id: str) -> str:
    """
    No single JPEG file represents "the compressed image" here (there
    are two compact streams + a mask) -- save the reassembled
    reconstruction instead, as a lossless PNG.
    """
    path = os.path.join(output_dir, image_id + ".png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(path, decoded_bgr)
    return path
