#!/usr/bin/env python3
"""
Calibration des seuils du garde-fou énergétique à partir de la vérité terrain
(pairs_classification.csv généré par la classification LLM).

Étape 1 : calculer les 3 métriques (hist / mean / prop) pour chaque paire du CSV,
          à partir des vraies images PNG.

    python calibrate_seuils.py extract \
        --images-dir /home/leandre/Desktop/DATASET-AGRISENSE-V1-GATE/640X480-PNG \
        --csv pairs_classification.csv \
        --out metriques_calibration.csv

Étape 2 : chercher la meilleure combinaison de seuils.

    python calibrate_seuils.py optimize \
        --metrics metriques_calibration.csv \
        --objectif f1

Notes :
- Par défaut, catégorie 0 (Différents) = CHANGEMENT, catégories 1/2/3 = similaire.
  Change avec --changement-categories "0,1" si tu veux inclure "Similaires".
- Le dataset est très déséquilibré (94.5% Différents) : privilégie f1 / balanced_acc
  / cost plutôt que accuracy brute.

--------------------------------------------------------------------------------
v2 (métriques modifiées) :
  1. Vectorisation complète (numpy) — aucun changement de logique, gain de vitesse.
  2. hist_global_diff et les histogrammes de blocs sont désormais calculés
     PAR CANAL (R, G, B séparés, hist_bins bins CHACUN) et comparés avec la
     distance de Bhattacharyya au lieu d'un histogramme fusionné + L1.
  3. mean_channel_diff utilise la MÉDIANE par canal au lieu de la moyenne
     (plus robuste à un reflet/ombre localisé).

⚠️ Ces 3 métriques n'ont plus la même échelle que dans l'ancienne version.
   Si tu avais déjà un metriques_calibration.csv généré avec l'ancienne version
   du script, il faut relancer `extract` pour le régénérer avec ces nouvelles
   métriques avant de relancer `optimize` — sinon les seuils calibrés seront
   incohérents avec ce que calcule réellement garde_fou_energetique.py.
--------------------------------------------------------------------------------
"""
import argparse
import csv
import os
import sys
from itertools import product

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("PIL manquant : pip install pillow --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Métriques — copie exacte de garde_fou_energetique.balanced_difference_metrics (v2)
# ---------------------------------------------------------------------------

EPS = 1e-9


def load_png(path):
    return np.array(Image.open(path).convert('RGB'))


def _channel_histograms(pixels, bins):
    """
    Histogrammes normalisés par canal, vectorisés.
    pixels : array (n_groupes, n_pixels_par_groupe, 3), valeurs uint8 (0-255).
    bins   : nombre de bins PAR CANAL.
    Retourne (n_groupes, 3, bins).
    """
    n_groupes, n_pixels, _ = pixels.shape
    bin_idx = np.minimum((pixels.astype(np.int32) * bins) // 256, bins - 1)

    hists = np.zeros((n_groupes, 3, bins), dtype=np.float64)
    group_offset = (np.arange(n_groupes) * bins)[:, None]

    for c in range(3):
        combined = (group_offset + bin_idx[:, :, c]).ravel()
        counts = np.bincount(combined, minlength=n_groupes * bins)
        hists[:, c, :] = counts.reshape(n_groupes, bins)

    hists /= hists.sum(axis=2, keepdims=True) + EPS
    return hists


def _bhattacharyya_distance(hists1, hists2):
    """Distance de Bhattacharyya par canal, moyennée sur les 3 canaux -> (n_groupes,)."""
    bc = np.sum(np.sqrt(hists1 * hists2), axis=2)
    bc = np.clip(bc, 1e-10, 1.0)
    dist_per_channel = -np.log(bc)
    return dist_per_channel.mean(axis=1)


def balanced_difference_metrics(img1, img2, block_size=32, hist_bins=4):
    """
    hist_bins : nombre de bins PAR CANAL (donc hist_bins*3 bins au total).
    """
    h, w = img1.shape[:2]
    metrics = {}

    # 1. Histogramme global, par canal, Bhattacharyya
    g1 = img1.reshape(1, -1, 3)
    g2 = img2.reshape(1, -1, 3)
    hg1 = _channel_histograms(g1, hist_bins)
    hg2 = _channel_histograms(g2, hist_bins)
    metrics['hist_global_diff'] = float(_bhattacharyya_distance(hg1, hg2)[0])

    # 2. Médiane par canal (au lieu de la moyenne)
    med1 = np.median(img1.reshape(-1, 3).astype(float), axis=0)
    med2 = np.median(img2.reshape(-1, 3).astype(float), axis=0)
    metrics['mean_channel_diff'] = float(np.mean(np.abs(med1 - med2)))

    # 3. Blocs, vectorisé, histogrammes par canal + Bhattacharyya
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

    leftover_blocks = []
    if h % block_size != 0 or w % block_size != 0:
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if y + block_size <= n_by * block_size and x + block_size <= n_bx * block_size:
                    continue
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
        metrics['mean_block_hist_diff'] = float(all_diffs.mean())
        seuil_bloc_change = 0.50  # placeholder, cohérent avec garde_fou_energetique.py, à recalibrer
        metrics['prop_blocks_changed'] = float((all_diffs > seuil_bloc_change).mean() * 100)
    else:
        metrics['mean_block_hist_diff'] = 0.0
        metrics['prop_blocks_changed'] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Étape 1 : extraction
# ---------------------------------------------------------------------------

def read_ground_truth(csv_path):
    gt = {}
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            gt[row['pair_id']] = {
                'f1': row['image1'],
                'f2': row['image2'],
                'categorie': int(row['categorie']),
                'confiance': float(row['confiance']),
            }
    return gt


def extract(images_dir, csv_path, out_path, block_size, hist_bins, min_confidence):
    gt = read_ground_truth(csv_path)
    rows = []
    cache = {}
    n = len(gt)
    n_missing = 0

    for i, (pid, info) in enumerate(gt.items(), 1):
        if info['confiance'] < min_confidence:
            continue
        p1 = os.path.join(images_dir, info['f1'])
        p2 = os.path.join(images_dir, info['f2'])
        if not (os.path.exists(p1) and os.path.exists(p2)):
            n_missing += 1
            continue
        try:
            if info['f1'] not in cache:
                cache[info['f1']] = load_png(p1)
            if info['f2'] not in cache:
                cache[info['f2']] = load_png(p2)
            m = balanced_difference_metrics(cache[info['f1']], cache[info['f2']],
                                             block_size, hist_bins)
            rows.append((pid, m['hist_global_diff'], m['mean_channel_diff'],
                         m['prop_blocks_changed'], info['categorie'], info['confiance']))
        except Exception as e:
            print(f"Erreur {pid}: {e}")

        if i % 100 == 0:
            print(f"{i}/{n} paires traitées...")
        if len(cache) > 60:  # limiter la RAM (images 640x480 uniquement en cache court terme)
            cache.clear()

    with open(out_path, 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'hist', 'mean', 'prop', 'categorie', 'confiance'])
        w.writerows(rows)

    print(f"\n{len(rows)} paires exportées -> {out_path}")
    if n_missing:
        print(f"({n_missing} paires ignorées : images introuvables dans {images_dir})")


# ---------------------------------------------------------------------------
# Étape 2 : optimisation des seuils
# ---------------------------------------------------------------------------

def load_metrics(path):
    data = {'hist': [], 'mean': [], 'prop': [], 'cat': []}
    with open(path, encoding='utf8') as f:
        for row in csv.DictReader(f):
            data['hist'].append(float(row['hist']))
            data['mean'].append(float(row['mean']))
            data['prop'].append(float(row['prop']))
            data['cat'].append(int(row['categorie']))
    return {k: np.array(v) for k, v in data.items()}


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def compute_scores(tp, fp, fn, tn, w_fp, w_fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = (recall + specificity) / 2
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    cost = w_fp * fp + w_fn * fn
    return dict(precision=precision, recall=recall, f1=f1, specificity=specificity,
                balanced_acc=balanced_acc, accuracy=accuracy, cost=cost)


def undersample_majority(y_true, target_frac_majority, seed=42):
    """
    Sous-échantillonne la classe majoritaire pour atteindre le ratio cible
    (ex: target_frac_majority=0.8 -> 80% classe majoritaire / 20% minoritaire),
    en gardant TOUTE la classe minoritaire intacte (elle est déjà rare, inutile
    de la rogner aussi).

    Retourne les indices à garder (tirage aléatoire sans remise, seed fixe pour
    la reproductibilité).
    """
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_true == 1)[0]  # CHANGEMENT
    idx_neg = np.where(y_true == 0)[0]  # similaire

    if len(idx_pos) >= len(idx_neg):
        majority_idx, minority_idx = idx_pos, idx_neg
    else:
        majority_idx, minority_idx = idx_neg, idx_pos

    n_minor = len(minority_idx)
    n_major_target = int(round(n_minor * target_frac_majority / (1 - target_frac_majority)))
    n_major_target = min(n_major_target, len(majority_idx))  # jamais plus que ce qui existe

    majority_sample = rng.choice(majority_idx, size=n_major_target, replace=False)
    keep = np.concatenate([majority_sample, minority_idx])
    rng.shuffle(keep)
    return keep


def optimize(metrics_path, changement_categories, objectif, grid_size, w_fp, w_fn,
             min_recall=0.0, min_specificity=0.0, balance_ratio=None, seed=42):
    d = load_metrics(metrics_path)
    y_true = np.isin(d['cat'], changement_categories).astype(int)
    print(f"{len(y_true)} paires | {y_true.sum()} CHANGEMENT | {(len(y_true) - y_true.sum())} similaire")

    if balance_ratio is not None:
        keep = undersample_majority(y_true, balance_ratio, seed=seed)
        d = {k: v[keep] for k, v in d.items()}
        y_true = y_true[keep]
        print(f"-> après sous-échantillonnage (ratio cible {balance_ratio:.0%}/"
              f"{1-balance_ratio:.0%}, seed={seed}) : {len(y_true)} paires | "
              f"{y_true.sum()} CHANGEMENT | {(len(y_true) - y_true.sum())} similaire")

    hist_grid = np.linspace(d['hist'].min(), np.percentile(d['hist'], 99), grid_size)
    mean_grid = np.linspace(d['mean'].min(), np.percentile(d['mean'], 99), grid_size)
    prop_grid = np.linspace(d['prop'].min(), np.percentile(d['prop'], 99), grid_size)

    c1 = d['hist'][:, None] > hist_grid[None, :]
    c2 = d['mean'][:, None] > mean_grid[None, :]
    c3 = d['prop'][:, None] > prop_grid[None, :]

    results = []
    for i, j, k in product(range(grid_size), repeat=3):
        score = c1[:, i].astype(int) + c2[:, j].astype(int) + c3[:, k].astype(int)
        y_pred = (score >= 2).astype(int)
        tp, fp, fn, tn = confusion(y_true, y_pred)
        s = compute_scores(tp, fp, fn, tn, w_fp, w_fn)
        s['thresh'] = (hist_grid[i], mean_grid[j], prop_grid[k])
        results.append(s)

    # on écarte les combinaisons dégénérées où un seuil est ~min ou ~max et neutralise
    # son critère (toujours vrai, ou toujours faux)
    eps_h = (d['hist'].max() - d['hist'].min()) * 0.02
    eps_m = (d['mean'].max() - d['mean'].min()) * 0.02
    eps_p = (d['prop'].max() - d['prop'].min()) * 0.02
    non_degenere = [r for r in results
                    if r['thresh'][0] > d['hist'].min() + eps_h
                    and r['thresh'][1] > d['mean'].min() + eps_m
                    and r['thresh'][2] > d['prop'].min() + eps_p]

    # contraintes explicites (ex: ne jamais descendre sous un recall minimum)
    contraint = [r for r in non_degenere
                 if r['recall'] >= min_recall and r['specificity'] >= min_specificity]
    if not contraint:
        print("Aucune combinaison ne respecte --min-recall/--min-specificity. "
              "Baisse ces contraintes.")
        return

    reverse = objectif != 'cost'
    contraint.sort(key=lambda r: r[objectif], reverse=reverse)

    print(f"\nTop 10 combinaisons (tri par {objectif}, recall>={min_recall}, "
          f"specificity>={min_specificity}), critères dégénérés exclus :")
    print(f"{'hist':>6} {'mean':>6} {'prop':>6} | {'acc':>5} {'prec':>5} {'rec':>5} "
          f"{'spec':>5} {'f1':>5} {'cost':>6}")
    for r in contraint[:10]:
        h, m, p = r['thresh']
        print(f"{h:6.3f} {m:6.2f} {p:6.1f} | {r['accuracy']:.3f} {r['precision']:.3f} "
              f"{r['recall']:.3f} {r['specificity']:.3f} {r['f1']:.3f} {r['cost']:6.0f}")

    # ⚠️ ces valeurs "actuelles" (0.38 / 6.0 / 18.0) sont celles de l'ANCIENNE échelle
    # (L1 + moyenne). Avec les nouvelles métriques (Bhattacharyya + médiane), elles ne
    # servent plus de référence valable : ne les compare pas telles quelles, regarde
    # plutôt le tableau top 10 ci-dessus pour choisir tes nouveaux seuils.
    cur_score = ((d['hist'] > 0.38).astype(int) + (d['mean'] > 6.0).astype(int)
                 + (d['prop'] > 18.0).astype(int))
    y_cur = (cur_score >= 2).astype(int)
    tp, fp, fn, tn = confusion(y_true, y_cur)
    s_cur = compute_scores(tp, fp, fn, tn, w_fp, w_fn)
    print(f"\n[repère obsolète, ancienne échelle] Seuils (0.38 / 6.0 / 18.0) : "
          f"acc={s_cur['accuracy']:.3f} prec={s_cur['precision']:.3f} "
          f"rec={s_cur['recall']:.3f} spec={s_cur['specificity']:.3f} "
          f"f1={s_cur['f1']:.3f} cost={s_cur['cost']:.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('extract', help="calcule les métriques pour chaque paire du CSV")
    p1.add_argument('--images-dir', required=True)
    p1.add_argument('--csv', required=True)
    p1.add_argument('--out', default='metriques_calibration.csv')
    p1.add_argument('--block-size', type=int, default=32)
    p1.add_argument('--hist-bins', type=int, default=4,
                     help="nombre de bins PAR CANAL (défaut 4 -> 12 bins au total, "
                          "comme l'ancien histogramme fusionné mais séparé par canal)")
    p1.add_argument('--min-confidence', type=float, default=0.0,
                     help="ignore les paires labellisées avec une confiance LLM trop faible")

    p2 = sub.add_parser('optimize', help="cherche la meilleure combinaison de seuils")
    p2.add_argument('--metrics', required=True)
    p2.add_argument('--changement-categories', default='0',
                     help="catégories considérées comme CHANGEMENT, ex: '0' ou '0,1'")
    p2.add_argument('--objectif', default='f1',
                     choices=['f1', 'accuracy', 'balanced_acc', 'recall', 'precision',
                              'specificity', 'cost'])
    p2.add_argument('--grid-size', type=int, default=25,
                     help="résolution de la grille par seuil (25 -> 25^3 combinaisons)")
    p2.add_argument('--w-fp', type=float, default=1.0,
                     help="poids d'un faux positif (transmission inutile = énergie gaspillée)")
    p2.add_argument('--w-fn', type=float, default=1.0,
                     help="poids d'un faux négatif (vrai changement raté = perte d'info)")
    p2.add_argument('--min-recall', type=float, default=0.0,
                     help="contrainte : ne garder que les seuils qui détectent au moins "
                          "cette fraction des vrais changements")
    p2.add_argument('--min-specificity', type=float, default=0.0,
                     help="contrainte : ne garder que les seuils qui filtrent au moins "
                          "cette fraction des vraies paires similaires")
    p2.add_argument('--balance-ratio', type=float, default=None,
                     help="rééquilibre les classes en sous-échantillonnant la classe "
                          "majoritaire (garde 100%% de la classe minoritaire). "
                          "Ex: --balance-ratio 0.8 -> ~80%% CHANGEMENT / 20%% similaire "
                          "au lieu du déséquilibre naturel du dataset. Défaut : pas de "
                          "rééquilibrage (toutes les données utilisées telles quelles).")
    p2.add_argument('--seed', type=int, default=42,
                     help="graine aléatoire pour le sous-échantillonnage (--balance-ratio), "
                          "reproductibilité du tirage")

    args = parser.parse_args()
    if args.cmd == 'extract':
        extract(args.images_dir, args.csv, args.out, args.block_size,
                 args.hist_bins, args.min_confidence)
    else:
        cats = [int(x) for x in args.changement_categories.split(',')]
        optimize(args.metrics, cats, args.objectif, args.grid_size, args.w_fp, args.w_fn,
                 args.min_recall, args.min_specificity, args.balance_ratio, args.seed)


if __name__ == '__main__':
    main()
