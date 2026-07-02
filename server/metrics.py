"""
metrics.py
------------------------------------------------------------------
Quality metrics used to evaluate the reconstructed images and
segmentation masks against the originals. Every formula here is a
standard, citable implementation -- nothing home-rolled or
unverifiable, per the "no unverifiable claim" rule for this revision.

- PSNR / SSIM: scikit-image's `peak_signal_noise_ratio` and
  `structural_similarity`. SSIM follows Wang et al., "Image quality
  assessment: From error visibility to structural similarity," IEEE
  TIP 2004 (already in the paper's bibliography as wang2004).
- IoU / Dice: standard set-overlap definitions for binary masks.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> float:
    """PSNR in dB between two uint8 RGB arrays of identical shape."""
    if original_rgb.shape != reconstructed_rgb.shape:
        raise ValueError(f"Shape mismatch: {original_rgb.shape} vs {reconstructed_rgb.shape}")
    return float(peak_signal_noise_ratio(original_rgb, reconstructed_rgb, data_range=255))


def compute_ssim(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> float:
    """SSIM (Wang et al. 2004) between two uint8 RGB arrays of identical shape."""
    if original_rgb.shape != reconstructed_rgb.shape:
        raise ValueError(f"Shape mismatch: {original_rgb.shape} vs {reconstructed_rgb.shape}")
    return float(structural_similarity(original_rgb, reconstructed_rgb, data_range=255, channel_axis=2))


def compute_iou_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """IoU and Dice coefficient between two binary masks (any nonzero = foreground)."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    area_sum = a.sum() + b.sum()

    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / area_sum) if area_sum > 0 else 1.0
    return iou, dice
