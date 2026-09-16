"""
Benchmark pilote : vérifie encodage/décodage/reconstruction/métriques/
cohérence des tailles/absence d'erreurs sur un petit échantillon avant
de lancer les 1338 images.

Usage : python -m core.pilot
"""

import random
import traceback

import config
from algorithms import ALL_CODECS
from core.data_loader import list_dataset_images, load_image
from core.runner import run_single
from core.report import write_results_csv, compute_summary, write_summary_csv


def main():
    random.seed(config.RANDOM_SEED)

    all_paths = list_dataset_images(config.DATASET_DIR)
    sample_paths = random.sample(all_paths, min(config.PILOT_N_IMAGES, len(all_paths)))

    print(f"[pilote] {len(sample_paths)} images sélectionnées sur {len(all_paths)} disponibles.")

    rows = []
    errors = []

    for path in sample_paths:
        try:
            loaded = load_image(path)
        except Exception as e:
            errors.append((path.name, "chargement", str(e)))
            traceback.print_exc()
            continue

        for algo_name, codec_cls in ALL_CODECS.items():
            try:
                codec = codec_cls()
                row = run_single(codec, loaded)

                # Vérifications de cohérence (protocole)
                assert row["S_raw"] == loaded.W * loaded.H * loaded.C, "S_raw incohérent"
                assert row["S_encoded"] > 0, "S_encoded nul ou négatif"
                if not codec.is_lossy and not row["lossless_exact"]:
                    print(f"  [ATTENTION] {algo_name} sur {path.name} : "
                          f"reconstruction non exacte alors qu'annoncé lossless.")

                rows.append(row)
                print(f"  OK  {path.name:20s} {algo_name:15s} "
                      f"CR={row['CR']:.2f}  bpp={row['bpp']:.2f}  "
                      f"T_total={row['T_total']*1000:.1f}ms")
            except Exception as e:
                errors.append((path.name, algo_name, str(e)))
                print(f"  ERREUR {path.name} / {algo_name} : {e}")
                traceback.print_exc()

    out_csv = config.OUTPUT_DIR / "pilot_results.csv"
    write_results_csv(rows, out_csv)
    summary = compute_summary(rows)
    write_summary_csv(summary, config.OUTPUT_DIR / "pilot_summary.csv")

    print(f"\n[pilote] {len(rows)} lignes écrites dans {out_csv}")
    if errors:
        print(f"[pilote] {len(errors)} erreurs rencontrées :")
        for image, algo, msg in errors:
            print(f"   - {image} / {algo} : {msg}")
    else:
        print("[pilote] Aucune erreur. Prêt pour le run complet (core/run_full.py).")


if __name__ == "__main__":
    main()
