"""
segmentation.py
================
Otsu thresholding on the luminance (Y) channel, with morphological
refinement (opening then closing, 3x3 kernel) — Section 4.2, step 3.

Produces a binary mask (uint8, 0/255) the same size as the input image.
Class 255 = ROI (vegetation, on this dataset). Combined with the VARI
block classification in vari.py (classify_blocks_composite, logical OR
at 8x8 block granularity) into the single composite mask used both for
Qveg/Qbg table selection at compression time and for zone-based
PSNR/SSIM on the station side.
"""

import cv2
import numpy as np


def compute_y_channel(img_bgr: np.ndarray) -> np.ndarray:
    """Extract the Y (luminance) channel, BT.601 weighting, from a BGR image."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0]


def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """
    Otsu threshold on Y, then 3x3 morphological opening followed by
    3x3 closing to remove speckle noise and fill small holes.

    Returns a binary mask (0 or 255), same H x W as the input image.
    """
    y = compute_y_channel(img_bgr)
    _, mask = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
