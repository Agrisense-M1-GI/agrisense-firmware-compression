"""
Calibration (À LANCER UNE SEULE FOIS, avant le pilote réel) : mesure le
taux d'erreur binaire réel, plan de bits par plan de bits (après code de
Gray), entre les images du dataset et leur side information générée,
sous le modèle de bruit FIXE défini dans config.py.

Objectif : décider une fois pour toutes quels plans de bits sont
suffisamment stables pour passer par le codage LDPC/syndrome, sur la
base d'une mesure réelle plutôt que d'une hypothèse de position
(cf. README, section "Conception finale SW/WZ" — l'hypothèse initiale
"les 3 plans de poids fort sont les plus stables" s'est révélée fausse
sur données synthétiques).

Ce script ne fait PAS partie du pilote ni du run complet : c'est une
étape de calibration préalable. Une fois les résultats obtenus, reporte
la liste des indices de plans à faible taux d'erreur dans
config.SW_COMPRESSED_PLANE_INDICES, puis lance core/pilot.py normalement.

Usage : python -m core.calibrate_planes
"""

import random
import numpy as np

import config
from core.data_loader import list_dataset_images, load_image
from core.side_information import generate_side_information


def _to_gray(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.uint16)
    return (v ^ (v >> 1)).astype(np.uint8)


def _uniform_quantize(arr: np.ndarray, levels: int):
    step = 256.0 / levels
    idx = np.clip((arr.astype(np.float64) // step).astype(np.int64), 0, levels - 1)
    return idx.astype(np.uint8)


def main(n_images: int = 20, seed: int = 4242):
    """
    n_images : taille de l'échantillon de calibration. Utilise une graine
    DIFFÉRENTE de celle du pilote (config.RANDOM_SEED) pour éviter de
    calibrer et d'évaluer sur exactement le même sous-échantillon.
    """
    rng = random.Random(seed)
    all_paths = list_dataset_images(config.DATASET_DIR)
    sample_paths = rng.sample(all_paths, min(n_images, len(all_paths)))

    print(f"[calibration] {len(sample_paths)} images pour mesurer le taux "
          f"d'erreur par plan de bits (code de Gray, modèle de bruit fixe).")

    # --- Slepian-Wolf : plans sur les pixels bruts (8 bits) ---
    sw_flip_rates = [[] for _ in range(8)]
    # --- Wyner-Ziv : plans sur les indices quantifiés (bits_per_symbol bits) ---
    bits_per_symbol = int(np.ceil(np.log2(config.WZ_QUANT_LEVELS)))
    wz_flip_rates = [[] for _ in range(bits_per_symbol)]

    for path in sample_paths:
        loaded = load_image(path)
        side_info = generate_side_information(loaded.array, loaded.image_id)

        # SW
        gray = _to_gray(loaded.array)
        side_gray = _to_gray(side_info)
        bits = np.unpackbits(gray.reshape(-1).astype(np.uint8)).reshape(-1, 8).T
        side_bits = np.unpackbits(side_gray.reshape(-1).astype(np.uint8)).reshape(-1, 8).T
        for plane_idx in range(8):
            sw_flip_rates[plane_idx].append((bits[plane_idx] != side_bits[plane_idx]).mean())

        # WZ
        q_idx = _uniform_quantize(loaded.array, config.WZ_QUANT_LEVELS)
        q_side = _uniform_quantize(side_info, config.WZ_QUANT_LEVELS)
        gray_q = _to_gray(q_idx)
        gray_q_side = _to_gray(q_side)
        qbits = np.unpackbits(gray_q.reshape(-1).astype(np.uint8)).reshape(-1, 8).T[8 - bits_per_symbol:]
        qbits_side = np.unpackbits(gray_q_side.reshape(-1).astype(np.uint8)).reshape(-1, 8).T[8 - bits_per_symbol:]
        for plane_idx in range(bits_per_symbol):
            wz_flip_rates[plane_idx].append((qbits[plane_idx] != qbits_side[plane_idx]).mean())

    THRESHOLD = 0.02  # relevé (était 0.01) suite au passage à un LDPC de
                       # rendement ~1/4 (LDPC_D_V=3, LDPC_D_C=4) : validé
                       # empiriquement à ~57% de blocs exacts à 2% d'erreur,
                       # ~77% à 1% — cf. historique du projet

    def _report(label, flip_rates, config_name):
        print(f"\n=== {label} ===")
        print(f"{'Plan':<6}{'Flip rate moyen':<20}{'Écart-type':<15}{'Recommandation'}")
        print("-" * 65)
        recommended = []
        for i, rates in enumerate(flip_rates):
            mean_r = np.mean(rates)
            std_r = np.std(rates)
            reco = "COMPRESSER (LDPC)" if mean_r < THRESHOLD else "transmettre en clair"
            if mean_r < THRESHOLD:
                recommended.append(i)
            print(f"{i:<6}{mean_r*100:<20.3f}{std_r*100:<15.3f}{reco}")
        print(f"\n[calibration] {config_name} = {recommended}")
        if not recommended:
            print(f"[calibration] ATTENTION : aucun plan ne passe le seuil pour {label}. "
                  "Le modèle de corrélation est probablement trop bruité pour ce "
                  "dataset avec ce rendement de code LDPC — à documenter comme "
                  "limitation, ou à revoir AVANT le pilote.")
        return recommended

    _report("Slepian-Wolf (pixels bruts, 8 bits)", sw_flip_rates, "SW_COMPRESSED_PLANE_INDICES")
    _report("Wyner-Ziv (indices quantifiés, "
            f"{bits_per_symbol} bits)", wz_flip_rates, "WZ_COMPRESSED_PLANE_INDICES")
    print("\n[calibration] Reporte ces deux listes dans config.py avant de "
          "lancer core/pilot.py.")


if __name__ == "__main__":
    main()
