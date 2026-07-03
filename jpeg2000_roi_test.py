#!/usr/bin/env python3
"""
jpeg2000_roi_test.py -- branch reference/jpeg
------------------------------------------------------------------
"JPEG2000-ROI" node-side encoder, HONESTLY documented.

WHY THIS FILE EXISTS IN THIS FORM (read before touching thresholds)
---------------------------------------------------------------------
The reference JPEG2000 implementation used here (OpenJPEG, via the
`opj_compress` / `opj_decompress` CLI tools) does NOT implement the
standard's native spatial Region-of-Interest feature (Annex H,
"Maxshift" method). OpenJPEG's own `-ROI` flag only up-shifts an
entire colour COMPONENT (R, G or B) during quantisation -- it has no
notion of an arbitrary-shaped spatial mask. This is documented by the
OpenJPEG project itself:
    "This option does not implement the usual ROI (Region of
     Interest). It should be understood as a 'Component of Interest'."
A genuine Annex H implementation (per-coefficient Maxshift, driven by
a spatial mask) has never been merged into OpenJPEG.

So instead of pretending to do something the library cannot do, this
script implements what is realistically achievable and STILL a fair,
real use of JPEG2000: a **two-stream, region-differentiated encoder**.

    1. A block-level ROI mask is computed with the same Otsu
       convention used by the ADRES/WZ-OSEG branches
       (gray > global_Otsu_threshold = foreground/vegetation-like,
       16x16 blocks, block flagged ROI if >25% of its pixels are
       foreground). This keeps the segmentation convention identical
       across all branches of the study, which is what makes
       cross-algorithm comparison meaningful.
    2. Two *full, independently valid* JPEG2000 codestreams are
       produced with the real OpenJPEG encoder:
         - "roi.jp2":  full resolution, but pixels outside ROI blocks
           are flattened to a flat mid-grey. This does not reserve
           bits the way Annex H would; it works because OpenJPEG's
           rate control (-r) allocates its bit budget according to
           image content/entropy, and a flattened background
           contributes near-zero entropy, so under a fixed target
           ratio the encoder ends up spending its budget on the ROI
           structure. This is a known, standard workaround for
           encoders without native spatial ROI, not an invented trick.
         - "bg.jp2": the full image, spatially sub-sampled (bicubic),
           encoded at a coarser target ratio.
    3. Both streams + the block-level ROI mask are packed into one
       container file. The sink reconstructs by decoding both
       streams and compositing: ROI blocks come from roi.jp2,
       everything else from the up-sampled bg.jp2.

This is a legitimate, reproducible use of real JPEG2000 encoding with
genuine region-differentiated quality allocation. It is NOT the
codestream-native Annex H ROI feature, and this script (and the
branch README) says so explicitly, every time, to avoid any
overstatement of what was measured.

Requires: opj_compress and opj_decompress on PATH.
    sudo apt install libopenjp2-tools
(or build OpenJPEG from source; both are C, consistent with the
project's C/C++/Python/Bash constraint -- we call the C encoder as a
subprocess, we do not reimplement JPEG2000 in Python.)
"""
import sys
import os
import time
import shutil
import struct
import subprocess
import tempfile
import resource
import numpy as np
from PIL import Image

BLOCK_SIZE = 16          # same convention as ADRES (16x16 blocks)
ROI_BLOCK_FRACTION = 0.25  # same convention as ADRES (>25% fg pixels -> ROI block)
SUBSAMPLE_BG = 2          # background linear downscale factor
ROI_TARGET_RATIO = 6      # opj_compress -r for the ROI stream (lower = higher quality)
BG_TARGET_RATIO = 40      # opj_compress -r for the background stream (higher = lower quality)
FLATTEN_VALUE = 128       # neutral grey used to flatten non-ROI pixels in the ROI stream

MAGIC = b"AGRIJ2K_ROI\n"


def estimate_energy(cpu_ms, mem_kb, tx_bytes):
    """Same literature-derived model used across the whole study (see
    context doc / article Eq.1). This is an order-of-magnitude estimate,
    not a hardware measurement -- see the paper's Section 5.3."""
    e_cpu = cpu_ms * 0.0054
    e_mem = mem_kb * 0.00001
    e_tx = tx_bytes * 0.18
    return e_cpu + e_mem + e_tx


def get_memory_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def otsu_threshold(gray):
    """Global Otsu threshold, implemented by hand (no extra dependency
    beyond numpy) to match exactly the convention already used by the
    C encoders (adres/encode.cpp: otsu_threshold()). Same maths."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def compute_roi_mask(rgb):
    """Block-level ROI mask, identical convention to ADRES:
    gray > Otsu threshold = foreground; block is ROI if >25% of its
    pixels are foreground. Returns (mask_bh_x_bw uint8 0/1, bw, bh, otsu_full uint8 HxW)."""
    h, w = rgb.shape[:2]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    th = otsu_threshold(gray)
    fg = (gray > th)

    bh = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    bw = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
    mask = np.zeros((bh, bw), dtype=np.uint8)
    for by in range(bh):
        y0, y1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, h)
        for bx in range(bw):
            x0, x1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, w)
            block = fg[y0:y1, x0:x1]
            if block.size and (block.sum() / block.size) > ROI_BLOCK_FRACTION:
                mask[by, bx] = 1

    otsu_full = (fg.astype(np.uint8) * 255)
    return mask, bw, bh, otsu_full


def build_roi_canvas(rgb, mask, bw, bh):
    """Full-res canvas: ROI blocks keep real pixels, everything else
    flattened to a neutral grey (see module docstring for why)."""
    h, w = rgb.shape[:2]
    canvas = np.full_like(rgb, FLATTEN_VALUE)
    for by in range(bh):
        y0, y1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, h)
        for bx in range(bw):
            if mask[by, bx]:
                x0, x1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, w)
                canvas[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    return canvas


def run_opj_compress(src_ppm, dst_jp2, ratio):
    cmd = ["opj_compress", "-i", str(src_ppm), "-o", str(dst_jp2), "-r", str(ratio)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not os.path.exists(dst_jp2):
        raise RuntimeError(
            f"opj_compress a échoué (code {result.returncode}). "
            f"stderr: {result.stderr.strip()}. "
            f"Vérifie que libopenjp2-tools est installé (opj_compress introuvable "
            f"ou plantage)."
        )


def save_container(out_path, mask, bw, bh, roi_jp2_bytes, bg_jp2_bytes, width, height):
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{BLOCK_SIZE} {bw} {bh}\n".encode())
        f.write(f"{SUBSAMPLE_BG}\n".encode())
        f.write(f"{len(roi_jp2_bytes)} {len(bg_jp2_bytes)}\n".encode())
        f.write(b"DATA\n")
        f.write(mask.tobytes())          # bw*bh bytes, 0/1
        f.write(roi_jp2_bytes)
        f.write(bg_jp2_bytes)


def main():
    if len(sys.argv) != 3:
        print("Usage: jpeg2000_roi_test.py <input.ppm> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    if shutil.which("opj_compress") is None:
        print("Erreur: opj_compress introuvable dans le PATH. "
              "Installe libopenjp2-tools (apt) ou compile OpenJPEG depuis les sources.")
        sys.exit(1)

    mem_before = get_memory_kb()
    t_start = time.time()

    original_size = os.path.getsize(input_file)
    img = Image.open(input_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.array(img)
    height, width = rgb.shape[:2]

    mask, bw, bh, _otsu_full = compute_roi_mask(rgb)
    # (the reference Otsu mask used for IoU/Dice is recomputed independently
    # server-side from the uploaded original -- see main.py -- so we don't
    # need to transmit otsu_full here; only the block-level ROI mask matters
    # for reconstruction.)

    roi_canvas = build_roi_canvas(rgb, mask, bw, bh)

    bg_w = max(1, width // SUBSAMPLE_BG)
    bg_h = max(1, height // SUBSAMPLE_BG)
    bg_canvas = np.array(
        Image.fromarray(rgb).resize((bg_w, bg_h), Image.BICUBIC)
    )

    with tempfile.TemporaryDirectory() as tmp:
        roi_ppm = os.path.join(tmp, "roi.ppm")
        bg_ppm = os.path.join(tmp, "bg.ppm")
        roi_jp2 = os.path.join(tmp, "roi.jp2")
        bg_jp2 = os.path.join(tmp, "bg.jp2")

        Image.fromarray(roi_canvas, "RGB").save(roi_ppm, "PPM")
        Image.fromarray(bg_canvas, "RGB").save(bg_ppm, "PPM")

        run_opj_compress(roi_ppm, roi_jp2, ROI_TARGET_RATIO)
        run_opj_compress(bg_ppm, bg_jp2, BG_TARGET_RATIO)

        with open(roi_jp2, "rb") as f:
            roi_jp2_bytes = f.read()
        with open(bg_jp2, "rb") as f:
            bg_jp2_bytes = f.read()

    compressed_path = os.path.join(output_dir, "compressed.jp2roi")
    save_container(compressed_path, mask, bw, bh, roi_jp2_bytes, bg_jp2_bytes, width, height)

    # Mask image for visualisation / server-side IoU-Dice, same convention as ADRES
    mask_img_path = os.path.join(output_dir, "roi_mask.ppm")
    mask_full = np.zeros((height, width), dtype=np.uint8)
    for by in range(bh):
        y0, y1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, height)
        for bx in range(bw):
            x0, x1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, width)
            mask_full[y0:y1, x0:x1] = 255 if mask[by, bx] else 0
    Image.fromarray(np.stack([mask_full] * 3, axis=2), "RGB").save(mask_img_path, "PPM")

    t_end = time.time()
    cpu_time_ms = (t_end - t_start) * 1000
    mem_after = get_memory_kb()
    mem_used = max(mem_after - mem_before, 512)

    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    energy = estimate_energy(cpu_time_ms, mem_used, compressed_size)

    metrics_path = os.path.join(output_dir, "node_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"cpu_time_ms,{cpu_time_ms:.3f}\n")
        f.write(f"memory_kb,{int(mem_used)}\n")
        f.write(f"compressed_bytes,{compressed_size}\n")
        f.write(f"compression_ratio,{compression_ratio:.2f}\n")
        f.write(f"energy_mj,{energy:.3f}\n")
        f.write(f"roi_bytes,{len(roi_jp2_bytes)}\n")
        f.write(f"bg_bytes,{len(bg_jp2_bytes)}\n")
        f.write(f"roi_block_fraction,{mask.mean():.4f}\n")

    print(f"JPEG2000-ROI (2-stream): {compression_ratio:.2f}x, {compressed_size} bytes "
          f"(ROI: {len(roi_jp2_bytes)}B, BG: {len(bg_jp2_bytes)}B), "
          f"blocs ROI: {100*mask.mean():.1f}%")


if __name__ == "__main__":
    main()
