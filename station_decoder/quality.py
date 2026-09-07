"""
station_decoder/quality.py — algo/agrijpeg-light
====================================================
No segmentation on this branch -- whole-image PSNR/SSIM only, same as
algo/jpeg-baseline. `mask` kept in the signature only so the call
matches every other branch's station_decoder/quality.py.
"""

from common import quality as quality_math


def compute_quality(decoded_bgr, reference_bgr, mask=None) -> dict:
    if decoded_bgr.shape != reference_bgr.shape:
        raise ValueError(
            f"Shape mismatch: decoded {decoded_bgr.shape} vs "
            f"reference {reference_bgr.shape}"
        )
    return {
        "psnr_db": quality_math.compute_psnr(reference_bgr, decoded_bgr),
        "ssim": quality_math.compute_ssim(reference_bgr, decoded_bgr),
    }
