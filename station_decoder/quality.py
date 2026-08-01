"""
station_decoder/quality.py — algo/jpeg-qveg-qbg
==================================================
Whole-image PSNR/SSIM (always) plus vegetation/fond-zoned PSNR/SSIM
(this branch has segmentation), using the block mask TRANSMITTED by the
node -- never recomputed from the reference dataset (see
common/vari.py's composite mask; recomputing station-side would let the
station "cheat" with information it wouldn't have in real deployment).
"""

import numpy as np

from common import config, quality as quality_math, vari


def compute_quality(decoded_bgr, reference_bgr, mask=None) -> dict:
    if decoded_bgr.shape != reference_bgr.shape:
        raise ValueError(
            f"Shape mismatch: decoded {decoded_bgr.shape} vs "
            f"reference {reference_bgr.shape}"
        )

    result = {
        "psnr_db": quality_math.compute_psnr(reference_bgr, decoded_bgr),
        "ssim": quality_math.compute_ssim(reference_bgr, decoded_bgr),
    }

    if mask is None:
        result.update({"psnr_db_veg": None, "ssim_veg": None,
                        "psnr_db_bg": None, "ssim_bg": None})
        return result

    h, w = decoded_bgr.shape[:2]
    pixel_mask = vari.upsample_block_mask(mask, config.QVEG_QBG_BLOCK_SIZE, (h, w))

    result["psnr_db_veg"] = _masked_psnr(reference_bgr, decoded_bgr, pixel_mask)
    result["ssim_veg"] = _masked_ssim(reference_bgr, decoded_bgr, pixel_mask)
    result["psnr_db_bg"] = _masked_psnr(reference_bgr, decoded_bgr, ~pixel_mask)
    result["ssim_bg"] = _masked_ssim(reference_bgr, decoded_bgr, ~pixel_mask)
    return result


def _masked_psnr(reference_bgr, decoded_bgr, pixel_mask) -> float:
    if not pixel_mask.any():
        return None
    diff = (reference_bgr.astype(np.float64) - decoded_bgr.astype(np.float64)) ** 2
    mse = diff[pixel_mask].mean()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / mse))


def _masked_ssim(reference_bgr, decoded_bgr, pixel_mask) -> float:
    if not pixel_mask.any():
        return None
    # Full-image SSIM map isn't separable per-pixel from compute_ssim's mean
    # reduction, so recompute the per-channel maps and average only inside
    # the mask -- same 11x11 Gaussian-window formula as common/quality.py.
    return float(np.mean([
        _masked_ssim_channel(reference_bgr[..., c], decoded_bgr[..., c], pixel_mask)
        for c in range(3)
    ]))


def _masked_ssim_channel(img1, img2, pixel_mask) -> float:
    import cv2
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
    cropped_mask = pixel_mask[5:-5, 5:-5]
    if not cropped_mask.any():
        return float(ssim_map.mean())
    return float(ssim_map[cropped_mask].mean())
