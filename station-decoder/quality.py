"""
station_decoder/quality.py — algo/jpeg-baseline
==================================================
Branch-specific quality metrics, called generically by the standalone
station receiver.

This branch has no segmentation in its pipeline (Section~branches), so
only whole-image PSNR/SSIM is computed -- no vegetation/fond zoning.
Branches with segmentation active should instead recompute
common.vari.classify_blocks_composite() on the reference image here and
report both whole-image and zoned metrics.
"""

from common import quality as quality_math


def compute_quality(decoded_bgr, reference_bgr) -> dict:
    if decoded_bgr.shape != reference_bgr.shape:
        raise ValueError(
            f"Shape mismatch: decoded {decoded_bgr.shape} vs "
            f"reference {reference_bgr.shape}"
        )
    return {
        "psnr_db": quality_math.compute_psnr(reference_bgr, decoded_bgr),
        "ssim": quality_math.compute_ssim(reference_bgr, decoded_bgr),
    }
