"""
Configuration globale et fixe du benchmark.
Ces paramètres sont fixés A PRIORI et ne doivent JAMAIS être modifiés
en fonction des résultats obtenus (règle d'équité du protocole).
"""

from pathlib import Path

# --- Chemins ---
DATASET_DIR = Path.home() / "Downloads" / "archive" / "UCID1338"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Protocole ---
PILOT_N_IMAGES = 15          # taille de l'échantillon pilote
RANDOM_SEED = 1338           # graine globale, reproductibilité

# --- JPEG (Pillow / libjpeg-turbo) ---
JPEG_QUALITY = 75            # qualité standard, non optimisée par image
JPEG_SUBSAMPLING = 0         # 4:4:4, pas de sous-échantillonnage chroma (comparaison équitable RGB brut)

# --- Huffman ---
HUFFMAN_BLOCK_MODE = "raw_bytes"   # alternative possible: "dpcm" (non utilisée par défaut)

# --- LZW ---
LZW_MAX_BITS = 12            # taille de dictionnaire standard (comme TIFF/GIF LZW)

# --- Slepian-Wolf / Wyner-Ziv : génération de la side information ---
# Y = blur_sigma(X) + Laplace(0, LAPLACE_SCALE), cf. README section "side information"
SIDE_INFO_BLUR_SIGMA = 0.5      # réduit (était 1.2) : canal de corrélation plus
                                  # fort, recalibré AVANT le pilote suite à
                                  # core.calibrate_planes (aucun plan ne passait
                                  # le seuil de 1% avec les valeurs précédentes)
SIDE_INFO_LAPLACE_SCALE = 1.0    # réduit (était 3.0), même justification

# --- LDPC (pyldpc) : paramètres fixes du code, identiques pour SW et WZ ---
LDPC_N = 512                 # longueur du code (doit être divisible par LDPC_D_C)
LDPC_D_V = 3                 # degré des noeuds variables (>=3 nécessaire pour
                              # que la règle majoritaire du bit-flipping soit
                              # opérante : avec d_v=2, "plus de la moitié des
                              # contraintes violées" exige les 2 simultanément,
                              # trop restrictif — erreur corrigée en cours de
                              # développement, cf. historique du projet)
LDPC_D_C = 4                 # réduit (était 6) => rendement ~1/4 (était ~1/2),
                              # plus de redondance transmise pour les plans
                              # compressés, en échange d'une meilleure
                              # convergence — recalibré avant le pilote suite
                              # à core.calibrate_planes (aucun plan ne passait
                              # le seuil de 1% avec les valeurs précédentes)
LDPC_MAKE_SYSTEMATIC = True
LDPC_MAX_ITERS = 15           # borne fixe d'itérations du bit-flipping majoritaire

# --- Slepian-Wolf / Wyner-Ziv : portée de la compression par syndrome ---
# IMPORTANT : lance `python -m core.calibrate_planes` AVANT le pilote pour
# mesurer, sur un échantillon réel du dataset, quels plans de bits (après
# code de Gray) ont un taux d'erreur assez faible pour que le décodeur
# LDPC/bit-flipping converge de manière fiable. Reporte ici la liste
# d'indices recommandée par la calibration. La valeur ci-dessous
# ([1, 2] par défaut) est UN PLACEHOLDER basé sur un test synthétique,
# PAS une mesure sur UCID1338 — à remplacer impérativement par le
# résultat de la calibration avant de lancer core.pilot ou core.run_full.
#
# Justification théorique du principe (pas du choix des indices) : le
# débit atteignable par Slepian-Wolf pour un plan donné est borné par
# H(X|Y) ; un plan quasi indépendant de la side info (H(X|Y)≈1) n'est
# théoriquement pas compressible et doit être transmis tel quel.
SW_COMPRESSED_PLANE_INDICES = [0]  # calibré sur UCID1338 (core.calibrate_planes) :
                                     # flip rate moyen 1.385% pour ce plan, seul
                                     # à passer le seuil de 2%

# Idem pour Wyner-Ziv, mais sur les plans de bits des INDICES QUANTIFIÉS
# (espace différent de SW : bits_per_symbol bits au lieu de 8) -> calibrer
# séparément avec `python -m core.calibrate_planes --wz`.
WZ_COMPRESSED_PLANE_INDICES = [0]  # calibré sur UCID1338, même mesure que SW
                                     # (flip rate moyen 1.385%)

# --- Wyner-Ziv : quantification scalaire ---
WZ_QUANT_LEVELS = 16         # nombre de niveaux du quantificateur uniforme (4 bits/pixel avant binning)

# --- Mesure ---
TIMING_REPEATS = 1            # réduit (était 3) pour rester exécutable en temps
                               # raisonnable avec les méthodes LDPC ; appliqué
                               # UNIFORMÉMENT aux 5 algorithmes (fairness préservée)
WARMUP_RUNS = 0
