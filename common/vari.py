"""
vari.py
=======
VARI (Visible Atmospherically Resistant Index) computation and per-block
vegetation classification — Section 4.2, step 4.

NOTE for this branch (algo/jpeg-baseline): this module is NOT used by
pipeline.py. Per the protocol, block-level VARI classification only runs
"si la branche utilise Qveg/Qbg" (Section 4.2) — this branch does not.
It is kept here, identical to every other branch, purely so the branch
stays self-contained and consistent (see project README). It IS used by:
algo/jpeg-qveg-qbg, algo/jpeg-qveg-qbg-4x1x4, algo/agrijpeg-core,
algo/agrijpeg-final, algo/agrijpeg-deploy.

VARI = (G - R) / (G + R - B)
"""

import numpy as np

from . import config


def compute_vari(img_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel VARI map. Input is an HxWx3 uint8 BGR image."""
    img = img_bgr.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    denom = g + r - b
    # Avoid division by zero on flat/degenerate pixels.
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return (g - r) / denom


def classify_blocks(img_bgr: np.ndarray, block_size: int = 8):
    """
    Splits the image into block_size x block_size blocks and classifies
    each as "vegetation" or "background" using the two-part rule
    (Section 4.2, step 4):
        - mean VARI of the block > VARI_MEAN_THRESHOLD, AND
        - fraction of "vegetation pixels" in the block > VARI_VEG_FRACTION
          (a pixel counts as vegetation if its own VARI > VARI_MEAN_THRESHOLD)

    Returns a 2D boolean array of shape (H // block_size, W // block_size),
    True where the block is classified as vegetation.
    """
    vari_map = compute_vari(img_bgr)
    h, w = vari_map.shape
    h_blocks, w_blocks = h // block_size, w // block_size

    veg_mask = np.zeros((h_blocks, w_blocks), dtype=bool)
    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = vari_map[by * block_size:(by + 1) * block_size,
                              bx * block_size:(bx + 1) * block_size]
            block_mean = block.mean()
            veg_pixel_fraction = np.mean(block > config.VARI_MEAN_THRESHOLD)
            veg_mask[by, bx] = (block_mean > config.VARI_MEAN_THRESHOLD and
                                 veg_pixel_fraction > config.VARI_VEG_FRACTION)
    return veg_mask
