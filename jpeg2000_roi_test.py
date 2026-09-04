#!/usr/bin/env python3
"""
jpeg2000_roi_test.py -- branche reference/jpeg — version 2.1
------------------------------------------------------------------
Changements par rapport à v1 :
  - Lecture PNG directe (plus de PPM en entrée).
  - Masque ROI sauvegardé en PNG (roi_mask.png) au lieu de PPM,
    cohérent avec ADRES et WZ-OSEG, lisible directement par Pillow.
  - Suppression du modèle énergie (Eq. 1) : INA219 non fonctionnel.
  - Métriques retenues : cpu_time_ms, memory_kb, compressed_bytes,
    compression_ratio, roi_bytes, bg_bytes, roi_block_fraction.

Ce que ce script fait et ne fait PAS (lire avant de modifier) :
  OpenJPEG (opj_compress) n'implémente PAS le ROI spatial natif
  de la norme JPEG2000 (Annexe H, méthode Maxshift). Son flag -ROI
  agit sur un composant couleur entier, pas sur une zone spatiale.
  Ce script implémente donc un encodeur deux-flux légitimement
  documenté : ROI pleine résolution (pixels hors ROI aplatis à 128),
  fond sous-échantillonné, les deux encodés avec opj_compress à des
  ratios différents, puis fusionnés dans un conteneur AGRIJ2K_ROI.
  Ce N'EST PAS le ROI natif JPEG2000 Annexe H.

Requires : opj_compress dans le PATH.
    sudo apt install libopenjp2-tools

Usage :
    python3 jpeg2000_roi_test.py <input.png> <output_dir>
"""
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

BLOCK_SIZE        = 16     # même convention qu'ADRES
ROI_BLOCK_FRACTION = 0.25  # même convention qu'ADRES
SUBSAMPLE_BG      = 2
ROI_TARGET_RATIO  = 6      # opj_compress -r (bas = haute qualité)
BG_TARGET_RATIO   = 40
FLATTEN_VALUE     = 128    # gris neutre pour pixels hors ROI dans le flux ROI
MAGIC             = b"AGRIJ2K_ROI\n"


def get_memory_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu identique à la convention des encodeurs C (ADRES, WZ-OSEG)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist    = hist.astype(np.float64)
    total   = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b   = 0.0; w_b = 0.0; max_var = 0.0; threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var  = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var: max_var = var; threshold = t
    return threshold


def compute_roi_mask(rgb: np.ndarray):
    """Masque ROI bloc-level, convention identique à ADRES."""
    h, w = rgb.shape[:2]
    gray = (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]).astype(np.uint8)
    th   = otsu_threshold(gray)
    fg   = gray > th
    bh   = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    bw   = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
    mask = np.zeros((bh, bw), dtype=np.uint8)
    for by in range(bh):
        y0, y1 = by*BLOCK_SIZE, min((by+1)*BLOCK_SIZE, h)
        for bx in range(bw):
            x0, x1 = bx*BLOCK_SIZE, min((bx+1)*BLOCK_SIZE, w)
            block = fg[y0:y1, x0:x1]
            if block.size and (block.sum() / block.size) > ROI_BLOCK_FRACTION:
                mask[by, bx] = 1
    return mask, bw, bh


def build_roi_canvas(rgb: np.ndarray, mask: np.ndarray, bw: int, bh: int) -> np.ndarray:
    h, w   = rgb.shape[:2]
    canvas = np.full_like(rgb, FLATTEN_VALUE)
    for by in range(bh):
        y0, y1 = by*BLOCK_SIZE, min((by+1)*BLOCK_SIZE, h)
        for bx in range(bw):
            if mask[by, bx]:
                x0, x1 = bx*BLOCK_SIZE, min((bx+1)*BLOCK_SIZE, w)
                canvas[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    return canvas


def run_opj_compress(src_ppm: str, dst_jp2: str, ratio: int) -> None:
    if shutil.which("opj_compress") is None:
        raise RuntimeError("opj_compress introuvable. sudo apt install libopenjp2-tools")
    cmd    = ["opj_compress", "-i", src_ppm, "-o", dst_jp2, "-r", str(ratio)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not os.path.exists(dst_jp2):
        raise RuntimeError(
            f"opj_compress a échoué (code {result.returncode}): {result.stderr.strip()}")


def save_container(out_path: str, mask: np.ndarray, bw: int, bh: int,
                   roi_jp2_bytes: bytes, bg_jp2_bytes: bytes,
                   width: int, height: int) -> None:
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{BLOCK_SIZE} {bw} {bh}\n".encode())
        f.write(f"{SUBSAMPLE_BG}\n".encode())
        f.write(f"{len(roi_jp2_bytes)} {len(bg_jp2_bytes)}\n".encode())
        f.write(b"DATA\n")
        f.write(mask.tobytes())
        f.write(roi_jp2_bytes)
        f.write(bg_jp2_bytes)


def save_mask_png(out_path: str, mask: np.ndarray,
                  bw: int, bh: int, width: int, height: int) -> None:
    """Masque ROI pleine résolution sauvegardé en PNG niveaux de gris
    (0 = fond, 255 = ROI). Remplace save roi_mask.ppm."""
    mask_full = np.zeros((height, width), dtype=np.uint8)
    for by in range(bh):
        y0, y1 = by*BLOCK_SIZE, min((by+1)*BLOCK_SIZE, height)
        for bx in range(bw):
            x0, x1 = bx*BLOCK_SIZE, min((bx+1)*BLOCK_SIZE, width)
            mask_full[y0:y1, x0:x1] = 255 if mask[by, bx] else 0
    Image.fromarray(mask_full, "L").save(out_path, "PNG")


def main():
    if len(sys.argv) != 3:
        print("Usage: jpeg2000_roi_test.py <input.png> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    if shutil.which("opj_compress") is None:
        print("Erreur: opj_compress introuvable. sudo apt install libopenjp2-tools")
        sys.exit(1)

    mem_before = get_memory_kb()
    t_start    = time.time()

    img          = Image.open(input_file).convert("RGB")
    rgb          = np.array(img)
    height, width = rgb.shape[:2]
    original_size = width * height * 3   # taille brute RGB non compressée

    mask, bw, bh = compute_roi_mask(rgb)
    roi_canvas   = build_roi_canvas(rgb, mask, bw, bh)

    bg_w = max(1, width  // SUBSAMPLE_BG)
    bg_h = max(1, height // SUBSAMPLE_BG)
    bg_canvas = np.array(Image.fromarray(rgb).resize((bg_w, bg_h), Image.BICUBIC))

    with tempfile.TemporaryDirectory() as tmp:
        roi_ppm = os.path.join(tmp, "roi.ppm")
        bg_ppm  = os.path.join(tmp, "bg.ppm")
        roi_jp2 = os.path.join(tmp, "roi.jp2")
        bg_jp2  = os.path.join(tmp, "bg.jp2")

        # opj_compress accepte PPM en entrée — conversion interne uniquement
        Image.fromarray(roi_canvas, "RGB").save(roi_ppm, "PPM")
        Image.fromarray(bg_canvas,  "RGB").save(bg_ppm,  "PPM")

        run_opj_compress(roi_ppm, roi_jp2, ROI_TARGET_RATIO)
        run_opj_compress(bg_ppm,  bg_jp2,  BG_TARGET_RATIO)

        with open(roi_jp2, "rb") as f: roi_jp2_bytes = f.read()
        with open(bg_jp2,  "rb") as f: bg_jp2_bytes  = f.read()

    compressed_path = os.path.join(output_dir, "compressed.jp2roi")
    save_container(compressed_path, mask, bw, bh,
                   roi_jp2_bytes, bg_jp2_bytes, width, height)

    # Masque en PNG (remplace roi_mask.ppm)
    save_mask_png(os.path.join(output_dir, "roi_mask.png"),
                  mask, bw, bh, width, height)

    cpu_time_ms       = (time.time() - t_start) * 1000
    mem_used          = max(get_memory_kb() - mem_before, 512)
    compressed_size   = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    metrics_path = os.path.join(output_dir, "node_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"cpu_time_ms,{cpu_time_ms:.3f}\n")
        f.write(f"memory_kb,{int(mem_used)}\n")
        f.write(f"compressed_bytes,{compressed_size}\n")
        f.write(f"compression_ratio,{compression_ratio:.2f}\n")
        f.write(f"roi_bytes,{len(roi_jp2_bytes)}\n")
        f.write(f"bg_bytes,{len(bg_jp2_bytes)}\n")
        f.write(f"roi_block_fraction,{mask.mean():.4f}\n")

    print(f"JPEG2000-ROI 2-flux: {compression_ratio:.2f}x, {compressed_size} bytes "
          f"(ROI: {len(roi_jp2_bytes)}B, BG: {len(bg_jp2_bytes)}B), "
          f"blocs ROI: {100*mask.mean():.1f}%")


if __name__ == "__main__":
    main()
