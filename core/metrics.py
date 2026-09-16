"""
Mesures : temps (haute résolution), mémoire (Peak RSS via getrusage),
qualité (PSNR/SSIM), vérification lossless.
"""

import time
import resource
import statistics
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

import config


def _peak_rss_kb() -> int:
    """Peak RSS du processus courant, en Ko (Linux: ru_maxrss est déjà en Ko)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def measure_baseline_rss() -> int:
    return _peak_rss_kb()


def timed_repeated(fn, repeats: int = None, warmup: int = None):
    """
    Exécute fn() plusieurs fois avec un warm-up, retourne (résultat_dernier_appel,
    médiane du temps en secondes, peak_rss_kb mesuré après les répétitions).
    """
    repeats = repeats or config.TIMING_REPEATS
    warmup = warmup if warmup is not None else config.WARMUP_RUNS

    for _ in range(warmup):
        fn()

    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_rss = _peak_rss_kb()
    return result, statistics.median(times), peak_rss


def compute_psnr_ssim(original: np.ndarray, reconstructed: np.ndarray):
    """PSNR/SSIM pour méthodes lossy. Retourne (psnr, ssim), np.inf si identique."""
    if original.shape != reconstructed.shape:
        raise ValueError("Formes différentes entre original et reconstruit — bug de décodage.")

    psnr = sk_psnr(original, reconstructed, data_range=255)
    ssim = sk_ssim(original, reconstructed, channel_axis=-1, data_range=255)
    return float(psnr), float(ssim)


def check_lossless_exact(original: np.ndarray, reconstructed: np.ndarray) -> bool:
    """Vérifie l'égalité exacte pixel par pixel pour les méthodes censées être lossless."""
    if original.shape != reconstructed.shape:
        return False
    return bool(np.array_equal(original, reconstructed))
