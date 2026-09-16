"""
Exécution d'un (image, algorithme) : encode, decode, mesure toutes les
métriques, retourne une ligne de résultats (dict) prête pour le CSV.
"""

import numpy as np

import config
from core.metrics import timed_repeated, compute_psnr_ssim, check_lossless_exact, measure_baseline_rss
from core.data_loader import LoadedImage


def run_single(codec, loaded: LoadedImage) -> dict:
    ctx = {
        "W": loaded.W,
        "H": loaded.H,
        "C": loaded.C,
        "image_id": loaded.image_id,
    }

    baseline_rss = measure_baseline_rss()

    encoded_holder = {}

    def _encode():
        encoded_holder["data"] = codec.encode(loaded.array, ctx)
        return encoded_holder["data"]

    encoded, t_enc, rss_after_enc = timed_repeated(_encode)

    decoded_holder = {}

    def _decode():
        decoded_holder["data"] = codec.decode(encoded, ctx)
        return decoded_holder["data"]

    decoded, t_decode, rss_after_dec = timed_repeated(_decode)

    s_raw = loaded.s_raw
    s_encoded = len(encoded)

    cr = s_raw / s_encoded if s_encoded else float("inf")
    saving_percent = 100.0 * (1 - s_encoded / s_raw) if s_raw else 0.0
    bpp = (s_encoded * 8) / (loaded.W * loaded.H)

    peak_rss = max(rss_after_enc, rss_after_dec)
    peak_rss_additional = peak_rss - baseline_rss

    if codec.is_lossy:
        psnr, ssim = compute_psnr_ssim(loaded.array, decoded)
        lossless_exact = False
    else:
        lossless_exact = check_lossless_exact(loaded.array, decoded)
        if lossless_exact:
            # Reconstruction réellement identique bit à bit : PSNR/SSIM
            # parfaits par définition, pas besoin de les calculer.
            psnr, ssim = float("inf"), 1.0
        else:
            # Algorithme nominalement lossless mais reconstruction
            # imparfaite (cas SW/WZ non convergés) : on calcule les
            # VRAIES métriques de qualité plutôt que de coder en dur
            # inf/1.0, qui serait faux et contradictoire avec
            # lossless_exact=False. Documente la sévérité de l'échec.
            psnr, ssim = compute_psnr_ssim(loaded.array, decoded)

    return {
        "image_id": loaded.image_id,
        "algorithm": codec.name,
        "W": loaded.W,
        "H": loaded.H,
        "S_raw": s_raw,
        "S_encoded": s_encoded,
        "CR": cr,
        "saving_percent": saving_percent,
        "bpp": bpp,
        "T_enc": t_enc,
        "T_decode": t_decode,
        "T_total": t_enc + t_decode,
        "PeakRSS": peak_rss,
        "PeakRSS_additional": peak_rss_additional,
        "PSNR": psnr,
        "SSIM": ssim,
        "lossless_exact": lossless_exact,
    }
