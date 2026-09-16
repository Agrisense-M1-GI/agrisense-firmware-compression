"""
Run complet sur les 1338 images du dataset UCID1338.
À exécuter seulement après validation du pilote (core/pilot.py).

Usage : python -m core.run_full
"""

import traceback
import csv

import config
from algorithms import ALL_CODECS
from core.data_loader import list_dataset_images, load_image
from core.runner import run_single
from core.report import write_results_csv, compute_summary, write_summary_csv, FIELDNAMES


def main():
    all_paths = list_dataset_images(config.DATASET_DIR)
    print(f"[run complet] {len(all_paths)} images à traiter x {len(ALL_CODECS)} algorithmes.")

    out_csv = config.OUTPUT_DIR / "full_results.csv"
    errors_log = config.OUTPUT_DIR / "full_run_errors.log"

    n_rows = 0
    n_errors = 0

    # Écriture incrémentale pour tenir sur 1338 images sans tout garder en RAM,
    # et pour ne pas perdre le travail en cas d'interruption.
    with open(out_csv, "w", newline="") as f_csv, open(errors_log, "w") as f_err:
        writer = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, path in enumerate(all_paths, 1):
            try:
                loaded = load_image(path)
            except Exception as e:
                f_err.write(f"{path.name} / chargement : {e}\n")
                n_errors += 1
                continue

            for algo_name, codec_cls in ALL_CODECS.items():
                try:
                    codec = codec_cls()
                    row = run_single(codec, loaded)
                    writer.writerow(row)
                    n_rows += 1
                except Exception as e:
                    f_err.write(f"{path.name} / {algo_name} : {e}\n")
                    f_err.write(traceback.format_exc() + "\n")
                    n_errors += 1

            if i % 50 == 0:
                print(f"  ... {i}/{len(all_paths)} images traitées "
                      f"({n_rows} lignes, {n_errors} erreurs)")

    print(f"\n[run complet] Terminé. {n_rows} lignes écrites dans {out_csv}")
    if n_errors:
        print(f"[run complet] {n_errors} erreurs loguées dans {errors_log}")

    # Recharge pour calculer les statistiques agrégées (lecture, pas de re-calcul lourd)
    rows = []
    with open(out_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for field in ("W", "H", "S_raw", "S_encoded"):
                r[field] = int(float(r[field]))
            for field in ("CR", "saving_percent", "bpp", "T_enc", "T_decode", "T_total",
                          "PeakRSS", "PeakRSS_additional", "PSNR", "SSIM"):
                r[field] = float(r[field])
            r["lossless_exact"] = r["lossless_exact"] == "True"
            rows.append(r)

    summary = compute_summary(rows)
    write_summary_csv(summary, config.OUTPUT_DIR / "summary_stats.csv")
    print(f"[run complet] Statistiques agrégées écrites dans "
          f"{config.OUTPUT_DIR / 'summary_stats.csv'}")


if __name__ == "__main__":
    main()
