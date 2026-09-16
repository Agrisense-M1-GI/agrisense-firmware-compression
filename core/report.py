"""
Écriture du CSV de résultats individuels + calcul des statistiques
agrégées (moyenne, médiane, écart-type, min, max) par algorithme.
"""

import csv
import statistics
from pathlib import Path

FIELDNAMES = [
    "image_id", "algorithm", "W", "H", "S_raw", "S_encoded", "CR",
    "saving_percent", "bpp", "T_enc", "T_decode", "T_total",
    "PeakRSS", "PeakRSS_additional", "PSNR", "SSIM", "lossless_exact",
]

NUMERIC_FIELDS = [
    "S_raw", "S_encoded", "CR", "saving_percent", "bpp",
    "T_enc", "T_decode", "T_total", "PeakRSS", "PeakRSS_additional",
    "PSNR", "SSIM",
]


def write_results_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_summary(rows: list[dict]) -> list[dict]:
    by_algo: dict[str, list[dict]] = {}
    for row in rows:
        by_algo.setdefault(row["algorithm"], []).append(row)

    summary = []
    for algo, algo_rows in by_algo.items():
        entry = {"algorithm": algo, "n_images": len(algo_rows)}
        for field in NUMERIC_FIELDS:
            values = [r[field] for r in algo_rows if r[field] not in (float("inf"), float("-inf"))]
            if not values:
                continue
            entry[f"{field}_mean"] = statistics.mean(values)
            entry[f"{field}_median"] = statistics.median(values)
            entry[f"{field}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry[f"{field}_min"] = min(values)
            entry[f"{field}_max"] = max(values)
        summary.append(entry)
    return summary


def write_summary_csv(summary: list[dict], path: Path):
    if not summary:
        return
    fieldnames = sorted({k for row in summary for k in row.keys()})
    fieldnames = ["algorithm", "n_images"] + [f for f in fieldnames if f not in ("algorithm", "n_images")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
