"""
common/quality.py
==================
Generic image-quality math (Section 6.2): PSNR and SSIM. Dependency-light
(OpenCV + numpy only, SSIM implemented directly -- Wang et al. 2004,
11x11 Gaussian window, single scale -- validated against scikit-image
to floating-point precision).

This module only computes numbers on two same-shape images. Whether a
branch applies this whole-image, or zoned via common/vari.py's composite
mask, is decided in that branch's own station_decoder/quality.py -- not
here. Kept in common/ because the math itself is identical across every
branch; only its application differs.
"""

import numpy as np
import cv2


def compute_psnr(reference_bgr: np.ndarray, decoded_bgr: np.ndarray) -> float:
    return float(cv2.PSNR(reference_bgr, decoded_bgr))


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel_1d = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel_1d, kernel_1d)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def compute_ssim(reference_bgr: np.ndarray, decoded_bgr: np.ndarray) -> float:
    """Mean SSIM across the 3 BGR channels, over the full images given."""
    return float(np.mean([
        _ssim_single_channel(reference_bgr[..., c], decoded_bgr[..., c])
        for c in range(3)
    ]))
