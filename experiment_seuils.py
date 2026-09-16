#!/usr/bin/env python3
"""
Expérimentation systématique de calibration des seuils du garde-fou énergétique.

Teste automatiquement plusieurs stratégies de gestion du déséquilibre de classes
(94.5% CHANGEMENT / 5.5% similaire) sans avoir à choisir une seule approche
à l'avance :

  - baseline        : aucun rééquilibrage, toutes les données telles quelles
  - undersample      : sous-échantillonnage de la classe majoritaire (garde
                        100% de la classe minoritaire), plusieurs ratios cibles,
                        plusieurs seeds pour vérifier la stabilité du tirage
  - weight (Option B): pondération des classes SANS jeter de données — chaque
                        exemple CHANGEMENT compte pour un poids <1 dans le calcul
                        agrégé (precision/F1/accuracy/cost), recall et specificity
                        restent inchangés par construction (ce sont des taux
                        intra-classe, indépendants des poids appliqués)

Pour chaque combinaison (méthode, ratio, seed, objectif), le script :
  1. cherche la meilleure combinaison de seuils SUR les données de la méthode
     (sous-échantillonnées ou pondérées)
  2. réapplique ensuite ces seuils trouvés sur la totalité des données brutes
     (2500 paires, comptage réel, sans pondération ni sous-échantillonnage)
     pour voir la performance "réelle" de ce jeu de seuils — c'est cette colonne
     (préfixe full_) qui permet de comparer les méthodes entre elles sur un pied
     d'égalité, indépendamment de l'artifice utilisé pour les calibrer.

Usage :
    python experiment_seuils.py --metrics metriques_calibration.csv

    # personnaliser les ratios/seeds/objectifs testés :
    python experiment_seuils.py --metrics metriques_calibration.csv \
        --ratios 0.5,0.6,0.7,0.8,0.9 --seeds 1,2,3,4,5 --objectifs f1,balanced_acc

Sortie :
    experiment_results.csv  (une ligne par combinaison testée, envoie-moi ce fichier)
    + un résumé imprimé dans le terminal (copie-le moi aussi si tu ne peux pas
      envoyer le fichier directement).
"""
import argparse
import csv
import itertools

import numpy as np

from calibrate_seuils import load_metrics, confusion, compute_scores, undersample_majority


# ---------------------------------------------------------------------------
# Option B : pondération sans perte de données
# ---------------------------------------------------------------------------

def compute_class_weights(y_true, target_frac_pos):
    """
    Calcule (w_pos, w_neg) tels que la somme des poids de la classe CHANGEMENT
    représente `target_frac_pos` du poids total, SANS retirer aucune ligne.

    w_neg = 1.0 (référence)
    w_pos choisi pour que : w_pos*n_pos / (w_pos*n_pos + w_neg*n_neg) = target_frac_pos
    """
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    w_neg = 1.0
    w_pos = target_frac_pos * n_neg / ((1 - target_frac_pos) * n_pos)
    return w_pos, w_neg


def weighted_confusion(y_true, y_pred, w_pos, w_neg):
    weights = np.where(y_true == 1, w_pos, w_neg)
    tp = float(weights[(y_pred == 1) & (y_true == 1)].sum())
    fp = float(weights[(y_pred == 1) & (y_true == 0)].sum())
    fn = float(weights[(y_pred == 0) & (y_true == 1)].sum())
    tn = float(weights[(y_pred == 0) & (y_true == 0)].sum())
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Recherche de seuils (identique en logique à calibrate_seuils.optimize,
# mais retourne juste le meilleur résultat au lieu d'imprimer un top 10)
# ---------------------------------------------------------------------------

def grid_search_best(d, y_true, grid_size, objectif, min_recall, min_specificity,
                      weights=None):
    """
    weights : None -> comptage brut (confusion classique)
              (w_pos, w_neg) -> confusion pondérée (Option B)
    """
    hist_grid = np.linspace(d['hist'].min(), np.percentile(d['hist'], 99), grid_size)
    mean_grid = np.linspace(d['mean'].min(), np.percentile(d['mean'], 99), grid_size)
    prop_grid = np.linspace(d['prop'].min(), np.percentile(d['prop'], 99), grid_size)

    c1 = d['hist'][:, None] > hist_grid[None, :]
    c2 = d['mean'][:, None] > mean_grid[None, :]
    c3 = d['prop'][:, None] > prop_grid[None, :]

    eps_h = (d['hist'].max() - d['hist'].min()) * 0.02
    eps_m = (d['mean'].max() - d['mean'].min()) * 0.02
    eps_p = (d['prop'].max() - d['prop'].min()) * 0.02

    best = None
    for i, j, k in itertools.product(range(grid_size), repeat=3):
        th = (hist_grid[i], mean_grid[j], prop_grid[k])
        if (th[0] <= d['hist'].min() + eps_h or th[1] <= d['mean'].min() + eps_m
                or th[2] <= d['prop'].min() + eps_p):
            continue  # seuil dégénéré (neutralise un des 3 critères)

        score = c1[:, i].astype(int) + c2[:, j].astype(int) + c3[:, k].astype(int)
        y_pred = (score >= 2).astype(int)

        if weights is None:
            tp, fp, fn, tn = confusion(y_true, y_pred)
        else:
            tp, fp, fn, tn = weighted_confusion(y_true, y_pred, *weights)

        s = compute_scores(tp, fp, fn, tn, w_fp=1.0, w_fn=1.0)
        if s['recall'] < min_recall or s['specificity'] < min_specificity:
            continue

        s['thresh'] = th
        if best is None:
            best = s
        elif objectif == 'cost':
            if s[objectif] < best[objectif]:
                best = s
        else:
            if s[objectif] > best[objectif]:
                best = s

    return best


def evaluate_on_full(d_full, y_full, thresh):
    """Réapplique un triplet de seuils (hist, mean, prop) sur les données brutes
    complètes (comptage réel, non pondéré, non sous-échantillonné)."""
    h_th, m_th, p_th = thresh
    score = ((d_full['hist'] > h_th).astype(int)
             + (d_full['mean'] > m_th).astype(int)
             + (d_full['prop'] > p_th).astype(int))
    y_pred = (score >= 2).astype(int)
    tp, fp, fn, tn = confusion(y_full, y_pred)
    return compute_scores(tp, fp, fn, tn, w_fp=1.0, w_fn=1.0)


# ---------------------------------------------------------------------------
# Boucle d'expériences
# ---------------------------------------------------------------------------

def run_experiments(metrics_path, changement_categories, grid_size, ratios, seeds,
                     objectifs, min_recall, min_specificity, out_path):
    d_full = load_metrics(metrics_path)
    y_full = np.isin(d_full['cat'], changement_categories).astype(int)
    n_pos, n_neg = int(y_full.sum()), int(len(y_full) - y_full.sum())
    print(f"Dataset complet : {len(y_full)} paires | {n_pos} CHANGEMENT | {n_neg} similaire\n")

    rows = []

    def add_row(method, ratio, seed, objectif, best, full_scores):
        rows.append({
            'method': method,
            'ratio': ratio if ratio is not None else '',
            'seed': seed if seed is not None else '',
            'objectif': objectif,
            'thresh_hist': round(best['thresh'][0], 4),
            'thresh_mean': round(best['thresh'][1], 4),
            'thresh_prop': round(best['thresh'][2], 4),
            'opt_precision': round(best['precision'], 4),
            'opt_recall': round(best['recall'], 4),
            'opt_specificity': round(best['specificity'], 4),
            'opt_f1': round(best['f1'], 4),
            'full_accuracy': round(full_scores['accuracy'], 4),
            'full_precision': round(full_scores['precision'], 4),
            'full_recall': round(full_scores['recall'], 4),
            'full_specificity': round(full_scores['specificity'], 4),
            'full_f1': round(full_scores['f1'], 4),
            'full_balanced_acc': round(full_scores['balanced_acc'], 4),
            'full_cost': round(full_scores['cost'], 1),
        })

    total_runs = len(objectifs) * (1 + len(ratios) * len(seeds) + len(ratios))
    done = 0

    for objectif in objectifs:
        # --- baseline : pas de rééquilibrage ---
        best = grid_search_best(d_full, y_full, grid_size, objectif, min_recall, min_specificity)
        if best is not None:
            full_scores = evaluate_on_full(d_full, y_full, best['thresh'])
            add_row('baseline', None, None, objectif, best, full_scores)
        done += 1
        print(f"[{done}/{total_runs}] baseline | objectif={objectif} -> "
              f"{'OK' if best else 'aucune combinaison valide'}")

        # --- undersample : plusieurs ratios x plusieurs seeds ---
        for ratio in ratios:
            for seed in seeds:
                keep = undersample_majority(y_full, ratio, seed=seed)
                d_sub = {k: v[keep] for k, v in d_full.items()}
                y_sub = y_full[keep]

                best = grid_search_best(d_sub, y_sub, grid_size, objectif, min_recall, min_specificity)
                done += 1
                if best is None:
                    print(f"[{done}/{total_runs}] undersample ratio={ratio} seed={seed} "
                          f"objectif={objectif} -> aucune combinaison valide")
                    continue
                full_scores = evaluate_on_full(d_full, y_full, best['thresh'])
                add_row('undersample', ratio, seed, objectif, best, full_scores)
                print(f"[{done}/{total_runs}] undersample ratio={ratio} seed={seed} "
                      f"objectif={objectif} -> full_f1={full_scores['f1']:.3f} "
                      f"full_spec={full_scores['specificity']:.3f}")

        # --- weight (Option B) : plusieurs ratios cibles, pas de seed (déterministe) ---
        for ratio in ratios:
            w_pos, w_neg = compute_class_weights(y_full, ratio)
            best = grid_search_best(d_full, y_full, grid_size, objectif, min_recall,
                                     min_specificity, weights=(w_pos, w_neg))
            done += 1
            if best is None:
                print(f"[{done}/{total_runs}] weight ratio={ratio} objectif={objectif} "
                      f"-> aucune combinaison valide")
                continue
            full_scores = evaluate_on_full(d_full, y_full, best['thresh'])
            add_row('weight', ratio, None, objectif, best, full_scores)
            print(f"[{done}/{total_runs}] weight ratio={ratio} objectif={objectif} -> "
                  f"full_f1={full_scores['f1']:.3f} full_spec={full_scores['specificity']:.3f}")

    # --- sauvegarde CSV ---
    fieldnames = ['method', 'ratio', 'seed', 'objectif',
                  'thresh_hist', 'thresh_mean', 'thresh_prop',
                  'opt_precision', 'opt_recall', 'opt_specificity', 'opt_f1',
                  'full_accuracy', 'full_precision', 'full_recall', 'full_specificity',
                  'full_f1', 'full_balanced_acc', 'full_cost']
    with open(out_path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} résultats -> {out_path}")

    # --- résumé lisible dans le terminal ---
    print("\n=== Top 15 par full_f1 (perf réelle sur les 2500 paires) ===")
    rows_sorted = sorted(rows, key=lambda r: r['full_f1'], reverse=True)
    _print_table(rows_sorted[:15])

    print("\n=== Top 15 par full_specificity (perf réelle, filtrage des similaires) ===")
    rows_sorted = sorted(rows, key=lambda r: r['full_specificity'], reverse=True)
    _print_table(rows_sorted[:15])

    print("\n=== Top 15 par full_balanced_acc (compromis recall/specificity réel) ===")
    rows_sorted = sorted(rows, key=lambda r: r['full_balanced_acc'], reverse=True)
    _print_table(rows_sorted[:15])


def _print_table(rows):
    cols = ['method', 'ratio', 'seed', 'objectif', 'thresh_hist', 'thresh_mean', 'thresh_prop',
            'full_precision', 'full_recall', 'full_specificity', 'full_f1', 'full_balanced_acc', 'full_cost']
    header = " | ".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r[c]):>10}" for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--changement-categories', default='0',
                         help="catégories considérées comme CHANGEMENT, ex: '0' ou '0,1'")
    parser.add_argument('--grid-size', type=int, default=15,
                         help="résolution de la grille par seuil (attention : cubique)")
    parser.add_argument('--ratios', default='0.5,0.6,0.7,0.8,0.9',
                         help="ratios cibles (fraction de la classe majoritaire) testés "
                              "pour undersample ET weight, séparés par des virgules")
    parser.add_argument('--seeds', default='1,2,3,4,5',
                         help="seeds testées pour le sous-échantillonnage (vérifier la "
                              "stabilité du tirage), séparées par des virgules")
    parser.add_argument('--objectifs', default='f1,balanced_acc',
                         help="objectifs d'optimisation testés, séparés par des virgules "
                              "(parmi f1, accuracy, balanced_acc, recall, precision, "
                              "specificity, cost)")
    parser.add_argument('--min-recall', type=float, default=0.0)
    parser.add_argument('--min-specificity', type=float, default=0.0)
    parser.add_argument('--out', default='experiment_results.csv')

    args = parser.parse_args()
    cats = [int(x) for x in args.changement_categories.split(',')]
    ratios = [float(x) for x in args.ratios.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    objectifs = args.objectifs.split(',')

    run_experiments(args.metrics, cats, args.grid_size, ratios, seeds, objectifs,
                     args.min_recall, args.min_specificity, args.out)


if __name__ == '__main__':
    main()
