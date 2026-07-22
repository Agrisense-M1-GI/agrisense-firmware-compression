"""
vari.py
=======
VARI (Visible Atmospherically Resistant Index) computation and per-block
vegetation classification — Section 4.2, step 4. Also combines this with
the pixel-level Otsu mask (segmentation.py) into a single composite ROI
mask, used both to select the Qveg/Qbg quantization table per block during
compression and to zone PSNR/SSIM (végétation/fond) on the station side —
the same mask drives both, so quality is always measured on exactly the
regions that were compressed differently.

NOTE for this branch (algo/jpeg-baseline): none of this module is used by
pipeline.py. Per the protocol, block-level classification only runs
"si la branche utilise Qveg/Qbg" (Section 4.2) — this branch does not.
It is kept here, identical to every other branch, purely so the branch
stays self-contained and consistent (see project README). It IS used by:
algo/jpeg-qveg-qbg, algo/jpeg-qveg-qbg-4x1x4, algo/agrijpeg-core,
algo/agrijpeg-final, algo/agrijpeg-deploy.

VARI = (G - R) / (G + R - B)
"""

import numpy as np

from . import config, segmentation


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


def classify_blocks_otsu(mask_otsu: np.ndarray, block_size: int = 8):
    """
    Converts the pixel-level Otsu ROI mask (segmentation.segment_otsu,
    class 255 = vegetation) into a block-level mask at the same
    granularity as classify_blocks(): a block counts as vegetation if at
    least OTSU_ROI_BLOCK_FRACTION of its pixels are Otsu-class 255.

    Returns a 2D boolean array of shape (H // block_size, W // block_size).
    """
    h, w = mask_otsu.shape
    h_blocks, w_blocks = h // block_size, w // block_size

    veg_mask = np.zeros((h_blocks, w_blocks), dtype=bool)
    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = mask_otsu[by * block_size:(by + 1) * block_size,
                               bx * block_size:(bx + 1) * block_size]
            roi_fraction = np.mean(block == 255)
            veg_mask[by, bx] = roi_fraction >= config.OTSU_ROI_BLOCK_FRACTION
    return veg_mask


def classify_blocks_composite(img_bgr: np.ndarray, block_size: int = 8):
    """
    Composite ROI mask: a block is classified vegetation if EITHER the
    Otsu-derived block mask OR the VARI-based block classification says
    so (logical OR). This is the single mask used both for Qveg/Qbg table
    selection at compression time and for zone-based PSNR/SSIM on the
    station side.

    Returns a 2D boolean array of shape (H // block_size, W // block_size).
    """
    mask_otsu = segmentation.segment_otsu(img_bgr)
    otsu_blocks = classify_blocks_otsu(mask_otsu, block_size=block_size)
    vari_blocks = classify_blocks(img_bgr, block_size=block_size)
    return otsu_blocks | vari_blocks


def upsample_block_mask(block_mask: np.ndarray, block_size: int, shape: tuple):
    """
    Expands a block-level boolean mask back to pixel resolution (each
    block's value repeated over its block_size x block_size footprint),
    so the exact same composite mask used for compression can also be
    used to zone PSNR/SSIM pixel-by-pixel on the station side.

    `shape` is the target (H, W) — the source image's actual resolution,
    which may exceed block_mask's implied H_blocks*block_size /
    W_blocks*block_size if H or W wasn't a multiple of block_size
    (edge blocks are simply not covered and stay False).
    """
    h, w = shape
    pixel_mask = np.zeros((h, w), dtype=bool)
    h_blocks, w_blocks = block_mask.shape
    for by in range(h_blocks):
        for bx in range(w_blocks):
            pixel_mask[by * block_size:(by + 1) * block_size,
                       bx * block_size:(bx + 1) * block_size] = block_mask[by, bx]
    return pixel_mask
