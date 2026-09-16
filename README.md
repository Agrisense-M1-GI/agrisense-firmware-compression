# README — Calibration des seuils du garde-fou énergétique

Ce dossier contient 3 scripts qui travaillent ensemble :

- `garde_fou_energetique.py` — l'algorithme déployé (calcule les 3 métriques,
  applique le vote "≥2 sur 3").
- `calibrate_seuils.py` — extrait les métriques depuis les images + vérité
  terrain, puis cherche les meilleurs seuils par grid search.
- `experiment_seuils.py` — teste automatiquement plusieurs stratégies de
  rééquilibrage des classes (sous-échantillonnage, pondération) et plusieurs
  objectifs, pour comparer les résultats sur un pied d'égalité.

## Prérequis

```bash
pip install numpy pillow --break-system-packages
```

Il te faut :
- un dossier d'images PNG (les vraies photos, ex: `640X480-PNG/`)
- un CSV de vérité terrain avec les colonnes `pair_id, image1, image2, categorie, confiance`

## Étape 1 — Extraire les métriques

Calcule `hist_global_diff`, `mean_channel_diff`, `prop_blocks_changed` pour
chaque paire du CSV, à partir des vraies images (utilise exactement les
mêmes fonctions que le garde-fou déployé) :

```bash
python calibrate_seuils.py extract \
    --images-dir . \
    --csv pairs_classification.csv \
    --out metriques_calibration.csv
```

- `--images-dir` : dossier contenant les PNG (`.` si tu es déjà dedans)
- Si tu vois `0 paires exportées` avec plein d'images "introuvables", vérifie
  que les noms de fichiers dans le CSV correspondent bien à ce qu'il y a
  dans `--images-dir` (`ls *.png` pour comparer).

## Étape 2 — Chercher les meilleurs seuils (une seule stratégie)

```bash
python calibrate_seuils.py optimize \
    --metrics metriques_calibration.csv \
    --objectif f1
```

Options utiles :
- `--objectif` : `f1`, `balanced_acc`, `recall`, `specificity`, `cost`, ...
- `--min-recall 0.9` / `--min-specificity 0.4` : contraintes à respecter
- `--balance-ratio 0.8` : sous-échantillonne la classe majoritaire pour viser
  un ratio 80/20 au lieu du déséquilibre naturel (garde 100% de la classe
  minoritaire) — utile pour éviter que l'optimiseur ignore complètement les
  paires "similaire"
- **`--objectif cost --w-fp <N> --w-fn 1`** : c'est l'option à utiliser une
  fois que tu as une estimation du coût relatif d'une transmission gaspillée
  (FP) vs un changement raté (FN). `N` = combien de fois plus cher est un FP
  qu'un FN. Voir `threshold_calibration_notes.md` pour le contexte complet.

## Étape 3 — Comparer plusieurs stratégies d'un coup (recommandé)

Plutôt que de deviner quelle stratégie utiliser, ce script teste tout et
te laisse comparer :

```bash
python experiment_seuils.py --metrics metriques_calibration.csv
```

Par défaut, teste : aucun rééquilibrage, sous-échantillonnage (5 ratios × 5
seeds), pondération (5 ratios), sur les objectifs `f1` et `balanced_acc`.

Options utiles :
- `--grid-size 10` : réduit la résolution de la grille (plus rapide, moins précis)
- `--ratios 0.5,0.6,0.7,0.8,0.9` : ratios testés
- `--seeds 1,2,3,4,5` : graines testées (vérifie que les seuils restent stables)
- `--objectifs f1,balanced_acc` : objectifs testés
- `--out mes_resultats.csv` : nom du fichier de sortie

**Sortie** : `experiment_results.csv` (une ligne par combinaison testée) +
un résumé dans le terminal (top 15 par F1, par specificity, par balanced
accuracy — colonnes `full_*` = performance réelle sur toutes les données,
pas sur le sous-échantillon).

Envoie-moi ce CSV (ou colle la sortie terminal) et je t'aide à choisir.

## Étape 4 — Appliquer les seuils choisis

Une fois les 3 seuils choisis, mets-les à jour dans `garde_fou_energetique.py` :

```python
thresh_hist      = ...   # hist_global_diff
thresh_mean_diff = ...   # mean_channel_diff
thresh_prop      = ...   # prop_blocks_changed
```

## Pour relancer tout le test après un changement du dataset ou des métriques

1. Si tu changes `garde_fou_energetique.py` (nouvelle métrique, autre
   `block_size`, autre `hist_bins`...), **répercute le même changement dans
   `calibrate_seuils.py`** (les deux fichiers doivent utiliser la fonction
   `balanced_difference_metrics` strictement identique — sinon les seuils
   calibrés ne correspondent plus à ce que le garde-fou calcule réellement).
2. Relance l'étape 1 (extract) — obligatoire si les images ou le CSV de
   vérité terrain ont changé.
3. Relance l'étape 2 ou 3 pour retrouver les seuils.

## Fichiers de référence

- `threshold_calibration_notes.md` — historique de la méthode, limites
  connues (déséquilibre de classes, biais de sélection, ratio coût FP/FN
  encore à quantifier) et pistes pour la suite.
