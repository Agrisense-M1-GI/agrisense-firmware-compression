"""
reference_jpeg_decode.py -- server-side decoders for the branch reference/jpeg.

Two methods are handled here:
  - "JPEG": standard, decoded directly by Pillow (no custom code needed --
    it's a plain, unmodified JPEG file).
  - "JPEG2000-ROI-2stream": the AGRIJ2K_ROI container produced by
    jpeg2000_roi_test.py (see that file's module docstring for exactly
    what this is and, just as importantly, what it is NOT -- it is not
    the JPEG2000 standard's native Annex H ROI feature).

Requires opj_decompress on PATH (same package as opj_compress:
libopenjp2-tools).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

MAGIC = b"AGRIJ2K_ROI\n"


def decode_jpeg(compressed_path) -> np.ndarray:
    """Plain JPEG: Pillow decodes it directly, nothing custom here."""
    img = Image.open(compressed_path).convert("RGB")
    return np.array(img)


def _run_opj_decompress(src_jp2_path, dst_ppm_path):
    if shutil.which("opj_decompress") is None:
        raise RuntimeError(
            "opj_decompress introuvable dans le PATH du serveur. "
            "Installe libopenjp2-tools sur la machine qui exécute main.py."
        )
    cmd = ["opj_decompress", "-i", str(src_jp2_path), "-o", str(dst_ppm_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not os.path.exists(dst_ppm_path):
        raise RuntimeError(
            f"opj_decompress a échoué (code {result.returncode}). "
            f"stderr: {result.stderr.strip()}"
        )


def read_compressed(container_path):
    """Parse the AGRIJ2K_ROI container. Returns
    (block_mask (bh,bw) uint8 0/1, roi_jp2_bytes, bg_jp2_bytes,
     (width, height, block_size, bw, bh, subsample))."""
    with open(container_path, "rb") as f:
        magic = f.readline()
        if magic != MAGIC:
            raise ValueError(f"Format invalide (magic={magic!r}), attendu {MAGIC!r}")

        width, height = map(int, f.readline().decode().split())
        block_size, bw, bh = map(int, f.readline().decode().split())
        subsample = int(f.readline().decode().strip())
        roi_len, bg_len = map(int, f.readline().decode().split())
        data_marker = f.readline()
        if data_marker.strip() != b"DATA":
            raise ValueError("Marqueur DATA manquant")

        mask_flat = np.frombuffer(f.read(bw * bh), dtype=np.uint8)
        mask = mask_flat.reshape(bh, bw)

        roi_jp2_bytes = f.read(roi_len)
        bg_jp2_bytes = f.read(bg_len)
        if len(roi_jp2_bytes) != roi_len or len(bg_jp2_bytes) != bg_len:
            raise ValueError("Fichier tronqué: tailles de flux incohérentes")

    return mask, roi_jp2_bytes, bg_jp2_bytes, (width, height, block_size, bw, bh, subsample)


def reconstruct_jpeg2000_roi(container_path) -> tuple[np.ndarray, np.ndarray]:
    """Decode both JP2 streams with the real OpenJPEG decoder and
    composite: ROI blocks come from the full-res ROI stream, everything
    else from the bicubic-upsampled background stream.

    Returns (reconstructed_rgb, pixel_level_roi_mask uint8 0/255) --
    the mask is returned too so the caller can compute IoU/Dice against
    the reference Otsu mask, same as ADRES/WZ-OSEG."""
    mask, roi_jp2_bytes, bg_jp2_bytes, (width, height, block_size, bw, bh, subsample) = \
        read_compressed(container_path)

    with tempfile.TemporaryDirectory() as tmp:
        roi_jp2 = os.path.join(tmp, "roi.jp2")
        bg_jp2 = os.path.join(tmp, "bg.jp2")
        roi_ppm = os.path.join(tmp, "roi.ppm")
        bg_ppm = os.path.join(tmp, "bg.ppm")

        with open(roi_jp2, "wb") as f:
            f.write(roi_jp2_bytes)
        with open(bg_jp2, "wb") as f:
            f.write(bg_jp2_bytes)

        _run_opj_decompress(roi_jp2, roi_ppm)
        _run_opj_decompress(bg_jp2, bg_ppm)

        roi_rgb = np.array(Image.open(roi_ppm).convert("RGB"))
        bg_rgb_small = np.array(Image.open(bg_ppm).convert("RGB"))

    bg_rgb = np.array(
        Image.fromarray(bg_rgb_small).resize((width, height), Image.BICUBIC)
    )

    reconstructed = bg_rgb.copy()
    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    for by in range(bh):
        y0, y1 = by * block_size, min((by + 1) * block_size, height)
        for bx in range(bw):
            if mask[by, bx]:
                x0, x1 = bx * block_size, min((bx + 1) * block_size, width)
                reconstructed[y0:y1, x0:x1] = roi_rgb[y0:y1, x0:x1]
                pixel_mask[y0:y1, x0:x1] = 255

    return reconstructed, pixel_mask
