"""
gate.py
=======
Change-detection gate (Section 4.2, step 2 of the common pipeline).

This is a direct port of the real v2 "garde_fou_energetique.py"
implementation: per-channel histograms compared with the Bhattacharyya
distance (instead of a merged histogram + L1 distance), and per-channel
MEDIAN deviation (instead of the mean, to stay robust to a localized
reflection or shadow that isn't a real scene change). Blocks are compared
the same way as the global histogram.

Inputs are HxWx3 uint8 numpy arrays. Channel order (RGB vs BGR) does not
matter here: every metric compares channel i of image A against channel i
of image B and then averages/aggregates across channels, so it is
order-invariant.

Decision: transmit if at least GATE_MIN_VOTES out of 3 criteria fire
(majority vote), exactly as in the protocol.
"""

import numpy as np

from . import config

EPS = 1e-9


def _channel_histograms(pixels: np.ndarray, bins: int) -> np.ndarray:
    """
    Normalized per-channel histograms, fully vectorized (no Python loop
    over groups).

    pixels : array (n_groups, n_pixels_per_group, 3), uint8 (0-255).
             A "group" is either the whole image (1 group) or one of
             n_by*n_bx blocks.
    bins   : number of bins PER CHANNEL.

    Returns an array (n_groups, 3, bins).
    """
    n_groups, n_pixels, _ = pixels.shape
    bin_idx = np.minimum((pixels.astype(np.int32) * bins) // 256, bins - 1)  # (n_groups, n_pixels, 3)

    hists = np.zeros((n_groups, 3, bins), dtype=np.float64)
    group_offset = (np.arange(n_groups) * bins)[:, None]

    for c in range(3):
        combined = (group_offset + bin_idx[:, :, c]).ravel()
        counts = np.bincount(combined, minlength=n_groups * bins)
        hists[:, c, :] = counts.reshape(n_groups, bins)

    hists /= hists.sum(axis=2, keepdims=True) + EPS
    return hists


def _bhattacharyya_distance(hists1: np.ndarray, hists2: np.ndarray) -> np.ndarray:
    """
    Per-channel Bhattacharyya distance, averaged over the 3 channels.
    hists1, hists2 : (n_groups, 3, bins) normalized histograms.
    Returns (n_groups,).
    """
    bc = np.sum(np.sqrt(hists1 * hists2), axis=2)  # Bhattacharyya coefficient per channel
    bc = np.clip(bc, 1e-10, 1.0)
    dist_per_channel = -np.log(bc)                 # 0 if identical, grows with divergence
    return dist_per_channel.mean(axis=1)


def balanced_difference_metrics(img1: np.ndarray, img2: np.ndarray,
                                 block_size: int = config.GATE_BLOCK_SIZE,
                                 hist_bins: int = config.GATE_HIST_BINS) -> dict:
    """
    Computes the 3 raw signals the gate votes on:
      - hist_global_diff   : Bhattacharyya distance, global per-channel histograms
      - mean_channel_diff  : mean absolute per-channel MEDIAN deviation
      - prop_blocks_changed: % of blocks whose Bhattacharyya distance exceeds
                              GATE_BLOCK_CHANGE_THRESHOLD
    """
    h, w = img1.shape[:2]
    metrics = {}

    # 1. Global histogram, per channel, Bhattacharyya distance
    g1 = img1.reshape(1, -1, 3)
    g2 = img2.reshape(1, -1, 3)
    hg1 = _channel_histograms(g1, hist_bins)
    hg2 = _channel_histograms(g2, hist_bins)
    metrics["hist_global_diff"] = float(_bhattacharyya_distance(hg1, hg2)[0])

    # 2. Per-channel shift: median (robust to a localized reflection/shadow)
    med1 = np.median(img1.reshape(-1, 3).astype(float), axis=0)
    med2 = np.median(img2.reshape(-1, 3).astype(float), axis=0)
    metrics["mean_channel_diff"] = float(np.mean(np.abs(med1 - med2)))

    # 3. Blocks — fully vectorized (reshape/transpose, no per-block Python loop)
    n_by, n_bx = h // block_size, w // block_size
    block_diffs = []

    if n_by > 0 and n_bx > 0:
        crop1 = img1[:n_by * block_size, :n_bx * block_size]
        crop2 = img2[:n_by * block_size, :n_bx * block_size]

        def to_blocks(img):
            return (img.reshape(n_by, block_size, n_bx, block_size, 3)
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(n_by * n_bx, block_size * block_size, 3))

        b1 = to_blocks(crop1)
        b2 = to_blocks(crop2)
        hb1 = _channel_histograms(b1, hist_bins)
        hb2 = _channel_histograms(b2, hist_bins)
        block_diffs.append(_bhattacharyya_distance(hb1, hb2))

    # Leftover strips if h/w are not multiples of block_size
    leftover_blocks = []
    if h % block_size != 0 or w % block_size != 0:
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if y + block_size <= n_by * block_size and x + block_size <= n_bx * block_size:
                    continue  # already covered by the vectorized part above
                bb1 = img1[y:y + block_size, x:x + block_size]
                bb2 = img2[y:y + block_size, x:x + block_size]
                if bb1.size == 0 or bb2.size == 0:
                    continue
                hbb1 = _channel_histograms(bb1.reshape(1, -1, 3), hist_bins)
                hbb2 = _channel_histograms(bb2.reshape(1, -1, 3), hist_bins)
                leftover_blocks.append(_bhattacharyya_distance(hbb1, hbb2)[0])

    all_diffs = list(block_diffs[0]) if block_diffs else []
    all_diffs.extend(leftover_blocks)

    if all_diffs:
        all_diffs = np.array(all_diffs)
        metrics["mean_block_hist_diff"] = float(all_diffs.mean())
        changed = all_diffs > config.GATE_BLOCK_CHANGE_THRESHOLD
        metrics["prop_blocks_changed"] = float(changed.mean() * 100)  # percent, 0-100
        metrics["nb_blocks_changed"] = int(changed.sum())
    else:
        metrics["mean_block_hist_diff"] = 0.0
        metrics["prop_blocks_changed"] = 0.0
        metrics["nb_blocks_changed"] = 0

    return metrics


def decide_change_balanced(metrics: dict,
                            thresh_hist: float = config.GATE_TAU_HIST,
                            thresh_mean_diff: float = config.GATE_TAU_MEAN,
                            thresh_prop_blocks: float = config.GATE_TAU_BLOCKS) -> bool:
    """
    Majority vote: at least GATE_MIN_VOTES (2) of the 3 conditions must
    fire for the decision to be "transmit".
    """
    cond1 = metrics["hist_global_diff"] > thresh_hist
    cond2 = metrics["mean_channel_diff"] > thresh_mean_diff
    cond3 = metrics["prop_blocks_changed"] > thresh_prop_blocks

    votes = int(cond1) + int(cond2) + int(cond3)
    return votes >= config.GATE_MIN_VOTES


def evaluate_gate(current_img: np.ndarray, reference_img: np.ndarray) -> dict:
    """
    Public entrypoint used by pipeline.py / pipeline_test.py. Runs the 3
    criteria, decides, and returns a dict shaped for direct use with
    common/metrics.py's NodeMetrics (gate_decision, gate_d_hist,
    gate_d_mean, gate_p_blocks), plus the two extra diagnostic fields from
    the original implementation (mean_block_hist_diff, nb_blocks_changed)
    for anyone who wants the richer detail in ad-hoc analysis.
    """
    raw = balanced_difference_metrics(current_img, reference_img)
    decision = decide_change_balanced(raw)

    return {
        "gate_decision": decision,
        "gate_d_hist": raw["hist_global_diff"],
        "gate_d_mean": raw["mean_channel_diff"],
        "gate_p_blocks": raw["prop_blocks_changed"],
        "mean_block_hist_diff": raw["mean_block_hist_diff"],
        "nb_blocks_changed": raw["nb_blocks_changed"],
    }
