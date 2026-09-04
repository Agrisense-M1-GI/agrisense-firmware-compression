"""
station_decoder/decode.py — algo/agrijpeg-core
==================================================
Reverses encode.py: rans_jpeg_codec transcode-decode (rANS -> standard
Huffman JPEG, byte-identical quantized coefficients to what
roi_jpeg_codec originally produced) then roi_jpeg_codec decode
(UNCHANGED, reuses the already-tested tiling reassembly).

Contract: decode() returns (image, mask), same as jpeg-qveg-qbg-4x1x4.
"""

import math
import os
import subprocess
import tempfile

import cv2
import numpy as np

from common import config, container


def decode(compressed_bytes: bytes):
    parts = container.unpack_qveg_qbg(compressed_bytes)
    orig_w, orig_h = parts["orig_w"], parts["orig_h"]
    block_size = config.QVEG_QBG_BLOCK_SIZE
    bw = math.ceil(orig_w / block_size)
    bh = math.ceil(orig_h / block_size)

    block_mask = np.frombuffer(parts["mask_bytes"], dtype=np.uint8).reshape(bh, bw).astype(bool)
    n_veg_blocks = int(block_mask.sum())
    n_bg_blocks = int((~block_mask).sum())

    if not os.path.exists(config.RANS_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"rans_jpeg_codec not found at {config.RANS_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )
    if not os.path.exists(config.ROI_JPEG_CODEC_BIN):
        raise RuntimeError(
            f"roi_jpeg_codec not found at {config.ROI_JPEG_CODEC_BIN} -- "
            f"build it first: compression/lib/build.sh"
        )

    with tempfile.TemporaryDirectory() as tmp:
        veg_rans = os.path.join(tmp, "veg.rans")
        bg_rans = os.path.join(tmp, "bg.rans")
        with open(veg_rans, "wb") as f:
            f.write(parts["veg_jpg"])  # container field name kept generic; holds .rans bytes here
        with open(bg_rans, "wb") as f:
            f.write(parts["bg_jpg"])

        veg_jpg = os.path.join(tmp, "veg.jpg")
        bg_jpg = os.path.join(tmp, "bg.jpg")

        for rans_path, jpg_path, qtable_path, n_blocks in [
            (veg_rans, veg_jpg, config.QVEG_TABLE_PATH, n_veg_blocks),
            (bg_rans, bg_jpg, config.QBG_TABLE_PATH, n_bg_blocks),
        ]:
            cmd = [
                config.RANS_JPEG_CODEC_BIN, "transcode-decode",
                rans_path, str(bw), str(n_blocks), qtable_path, jpg_path,
                "--sample", config.ROI_JPEG_SAMPLE_FACTORS,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"rans_jpeg_codec transcode-decode failed on {rans_path} "
                    f"(code {result.returncode}): {result.stderr.decode(errors='replace')}"
                )

        blockmap_path = os.path.join(tmp, "blockmap.bin")
        with open(blockmap_path, "wb") as f:
            f.write(parts["mask_bytes"])
        out_ppm = os.path.join(tmp, "out.ppm")

        cmd = [
            config.ROI_JPEG_CODEC_BIN, "decode",
            veg_jpg, bg_jpg, blockmap_path,
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

    return img_bgr, block_mask


def save_example(compressed_bytes, decoded_bgr, mask, output_dir: str, image_id: str) -> str:
    path = os.path.join(output_dir, image_id + ".png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(path, decoded_bgr)
    return path
