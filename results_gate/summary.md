# Calibration robuste du vote majoritaire

Règle : transmettre si **au moins 2 critères sur 3** dépassent leur seuil (comparaison stricte `>`). PROP = % de blocs 32×32 dont la distance de Bhattacharyya dépasse le « seuil de bloc ».

## Seuils recommandés — scénario `hard` (méthode robuste)

| objectif | HIST > | MEAN > | PROP > (%) | seuil de bloc | IC95 HIST | IC95 MEAN | IC95 PROP* |
|---|---|---|---|---|---|---|---|
| spec@0.95 | 0.00208 | 20.253 | 4.67 | 0.3 | [0.00104, 0.01665] | [6.7, 20.25] | [4.667, 8.133] |
| spec@0.98 | 0.02023 | 1.333 | 16.00 | 0.1 | [0.000187, 0.02844] | [1.333, 20.25] | [16, 25.51] |
| bacc | 0.00146 | 20.253 | 64.00 | 0.05 | [0.0003322, 0.02414] | [1.333, 20.25] | [58, 64] |

\* IC à seuil de bloc fixé (bootstrap par blocs temporels, refait toute la sélection). Un IC étroit = seuil stable.

## Seuils recommandés — scénario `all` (méthode robuste)

| objectif | HIST > | MEAN > | PROP > (%) | seuil de bloc | IC95 HIST | IC95 MEAN | IC95 PROP* |
|---|---|---|---|---|---|---|---|
| spec@0.95 | 0.02292 | 1.000 | 60.67 | 0.05 | [7.948e-06, 0.02974] | [1, 18.33] | [60.67, 60.67] |
| spec@0.98 | 0.00067 | 18.333 | 18.41 | 0.1 | [0.0001436, 0.02292] | [0.3333, 18.33] | [18.41, 18.41] |
| bacc | 0.00108 | 15.333 | 60.67 | 0.05 | [0.000356, 0.02292] | [1, 18.33] | [60.67, 60.67] |

\* IC à seuil de bloc fixé (bootstrap par blocs temporels, refait toute la sélection). Un IC étroit = seuil stable.

## Scénario `hard` — labels non triviaux (auto-paires exclues) — scénario principal ; n=1663, positifs=832

**AUC des critères bruts (IC95 % bootstrap par blocs)**

| critère | AUC |
|---|---|
| hist | 0.958 [0.947, 0.970] |
| mean | 0.876 [0.856, 0.897] |
| mblock | 0.980 [0.971, 0.988] |
| prop@0.05 | 0.987 [0.981, 0.992] |
| prop@0.1 | 0.987 [0.980, 0.992] |
| prop@0.2 | 0.984 [0.977, 0.990] |
| prop@0.3 | 0.982 [0.975, 0.989] |
| prop@0.5 | 0.977 [0.967, 0.986] |

### Objectif `spec@0.95` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.939 [0.900, 0.970] | 0.929 [0.892, 0.959] | 0.934 [0.912, 0.952] | 0.868 [0.825, 0.904] | 0.743 [0.711, 0.771] | 0.245 [0.217, 0.277] | 3/5 | 0.742 |
| robuste | 0.936 [0.897, 0.969] | 0.948 [0.921, 0.971] | 0.942 [0.922, 0.959] | 0.885 [0.845, 0.918] | 0.759 [0.734, 0.782] | 0.229 [0.206, 0.254] | 3/5 | 0.742 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : -0.002 [-0.009, +0.002] 
- robuste / specificity : +0.019 [+0.003, +0.042] ✱

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.001517 [0.001231, 0.03602] | 19.99 [1, 20.11] | 22.69 [7, 55.5] | 0.1×2, 0.2×1, 0.05×1, 0.3×1 |
| robuste | 0.0265 [0.001328, 0.03625] | 1.333 [1, 20.11] | 55.5 [11, 59] | 0.05×4, 0.3×1 |

### Objectif `spec@0.98` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.987 [0.976, 0.995] | 0.798 [0.742, 0.848] | 0.892 [0.866, 0.916] | 0.799 [0.757, 0.839] | 0.629 [0.583, 0.669] | 0.359 [0.319, 0.405] | 4/5 | 0.944 |
| robuste | 0.982 [0.970, 0.992] | 0.845 [0.786, 0.896] | 0.913 [0.886, 0.937] | 0.835 [0.790, 0.876] | 0.667 [0.620, 0.709] | 0.321 [0.279, 0.368] | 4/5 | 0.935 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : -0.005 [-0.011, +0.000] 
- robuste / specificity : +0.047 [+0.028, +0.068] ✱

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.00029 [0.0001864, 0.001003] | 9 [8.333, 9.667] | 0 [0, 10.8] | 0.5×3, 0.2×2 |
| robuste | 0.0003855 [0.0001864, 0.001003] | 19.99 [9, 20.33] | 8.06 [0.6333, 19.67] | 0.1×2, 0.2×2, 0.5×1 |

### Objectif `bacc` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.944 [0.917, 0.966] | 0.961 [0.940, 0.980] | 0.953 [0.938, 0.966] | 0.905 [0.876, 0.932] | 0.768 [0.749, 0.785] | 0.220 [0.203, 0.239] |  | 0.871 |
| robuste | 0.944 [0.915, 0.969] | 0.954 [0.930, 0.975] | 0.949 [0.933, 0.963] | 0.898 [0.867, 0.925] | 0.763 [0.740, 0.782] | 0.225 [0.206, 0.248] |  | 0.806 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : +0.000 [-0.018, +0.017] 
- robuste / specificity : -0.007 [-0.018, +0.001] 

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.002392 [0.001073, 0.002585] | 16.33 [16.33, 20] | 61.33 [55.5, 69.93] | 0.05×5 |
| robuste | 0.002393 [0.002355, 0.02036] | 16.33 [1.333, 20.11] | 55.5 [40, 59] | 0.05×4, 0.1×1 |

### Robustesse du point de fonctionnement final — `hard`

Sensibilité = pire résultat quand chaque seuil bouge de ±1 pas de grille (sur les données de calibration). Décisif = part des paires où le critère change la décision.

| objectif | méthode | recall / spéc. (calibration) | pire recall ±1 pas | pire spéc. ±1 pas | décisif HIST | décisif MEAN | décisif PROP |
|---|---|---|---|---|---|---|---|
| spec@0.95 | classique | 0.969 / 0.940 | 0.934 | 0.893 | 0.52 | 0.13 | 0.60 |
| spec@0.95 | robuste | 0.963 / 0.940 | 0.930 | 0.898 | 0.52 | 0.14 | 0.60 |
| spec@0.98 | classique | 0.989 / 0.832 | 0.984 | 0.817 | 0.58 | 0.22 | 0.76 |
| spec@0.98 | robuste | 0.989 / 0.853 | 0.975 | 0.771 | 0.28 | 0.40 | 0.66 |
| bacc | classique | 0.944 / 0.981 | 0.879 | 0.929 | 0.45 | 0.12 | 0.53 |
| bacc | robuste | 0.954 / 0.974 | 0.898 | 0.919 | 0.48 | 0.17 | 0.64 |

### Sensibilité au prior de déploiement — `hard` (part réelle de paires « différentes » ; énergie économisée / changements ratés)

| objectif | méthode | prior 0.05 | prior 0.1 | prior 0.2 | prior 0.3 | prior 0.5 |
|---|---|---|---|---|---|---|
| spec@0.95 | classique | 0.874 / 0.0031 | 0.830 / 0.0061 | 0.743 / 0.0123 | 0.657 / 0.0184 | 0.483 / 0.0306 |
| spec@0.95 | robuste | 0.892 / 0.0032 | 0.848 / 0.0064 | 0.759 / 0.0127 | 0.671 / 0.0191 | 0.494 / 0.0319 |
| spec@0.98 | classique | 0.747 / 0.0007 | 0.707 / 0.0013 | 0.629 / 0.0026 | 0.550 / 0.0040 | 0.394 / 0.0066 |
| spec@0.98 | robuste | 0.791 / 0.0009 | 0.750 / 0.0018 | 0.667 / 0.0036 | 0.585 / 0.0054 | 0.419 / 0.0090 |
| bacc | classique | 0.904 / 0.0028 | 0.859 / 0.0056 | 0.768 / 0.0113 | 0.678 / 0.0169 | 0.497 / 0.0282 |
| bacc | robuste | 0.897 / 0.0028 | 0.852 / 0.0056 | 0.763 / 0.0113 | 0.673 / 0.0169 | 0.493 / 0.0282 |

## Scénario `all` — tous labels (optimiste : inclut les auto-paires triviales) ; n=2503, positifs=832

**AUC des critères bruts (IC95 % bootstrap par blocs)**

| critère | AUC |
|---|---|
| hist | 0.979 [0.974, 0.986] |
| mean | 0.938 [0.927, 0.950] |
| mblock | 0.990 [0.985, 0.994] |
| prop@0.05 | 0.992 [0.988, 0.996] |
| prop@0.1 | 0.992 [0.987, 0.996] |
| prop@0.2 | 0.990 [0.985, 0.995] |
| prop@0.3 | 0.989 [0.984, 0.994] |
| prop@0.5 | 0.987 [0.980, 0.992] |

### Objectif `spec@0.95` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.959 [0.933, 0.981] | 0.963 [0.945, 0.977] | 0.961 [0.947, 0.973] | 0.915 [0.888, 0.940] | 0.766 [0.751, 0.780] | 0.222 [0.208, 0.237] | 3/5 | 0.848 |
| robuste | 0.952 [0.923, 0.976] | 0.963 [0.946, 0.978] | 0.958 [0.943, 0.971] | 0.910 [0.881, 0.936] | 0.768 [0.752, 0.782] | 0.220 [0.206, 0.236] | 3/5 | 0.816 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : -0.007 [-0.015, -0.001] ✱
- robuste / specificity : +0.001 [-0.003, +0.005] 

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.00152 [0.001268, 0.002281] | 18.33 [18, 18.33] | 22.33 [6, 44.77] | 0.05×2, 0.2×1, 0.1×1, 0.3×1 |
| robuste | 0.00152 [0.001268, 0.002281] | 18.33 [18, 18.33] | 5.173 [2, 23] | 0.3×2, 0.1×1, 0.2×1, 0.5×1 |

### Objectif `spec@0.98` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.989 [0.980, 0.997] | 0.901 [0.870, 0.928] | 0.945 [0.930, 0.958] | 0.858 [0.823, 0.888] | 0.711 [0.686, 0.733] | 0.277 [0.255, 0.302] | 4/5 | 0.952 |
| robuste | 0.968 [0.943, 0.986] | 0.920 [0.887, 0.947] | 0.944 [0.926, 0.959] | 0.863 [0.824, 0.900] | 0.730 [0.703, 0.753] | 0.258 [0.235, 0.285] | 3/5 | 0.864 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : -0.022 [-0.043, -0.005] ✱
- robuste / specificity : +0.019 [+0.007, +0.031] ✱

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.0003066 [0.0001988, 0.02209] | 9.273 [0.6667, 18.33] | 0.3333 [0, 13.61] | 0.5×3, 0.1×2 |
| robuste | 0.0002057 [0.000186, 0.0009579] | 18 [9.333, 18.33] | 7 [0, 44.33] | 0.5×2, 0.1×1, 0.2×1, 0.05×1 |

### Objectif `bacc` (performance hors-échantillon, CV imbriquée)

| méthode | recall | specificity | balanced_acc | mcc | energy_saving_deploy | tx_rate_deploy | folds recall≥cible | recall pire fold |
|---|---|---|---|---|---|---|---|---|
| classique | 0.950 [0.919, 0.975] | 0.971 [0.959, 0.982] | 0.960 [0.945, 0.973] | 0.919 [0.891, 0.943] | 0.775 [0.763, 0.786] | 0.213 [0.202, 0.225] |  | 0.816 |
| robuste | 0.972 [0.955, 0.987] | 0.959 [0.941, 0.975] | 0.966 [0.955, 0.976] | 0.920 [0.895, 0.944] | 0.761 [0.745, 0.774] | 0.227 [0.214, 0.243] |  | 0.936 |

Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :

- robuste / recall : +0.023 [+0.007, +0.044] ✱
- robuste / specificity : -0.012 [-0.024, -0.001] ✱

Stabilité des seuils entre folds (médiane [min, max]) :

| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |
|---|---|---|---|---|
| classique | 0.002143 [0.0008421, 0.002391] | 18 [15, 18.33] | 47.51 [2, 64.67] | 0.05×4, 0.5×1 |
| robuste | 0.001383 [0.001268, 0.01785] | 18 [1, 18.33] | 12.33 [3.333, 23] | 0.1×2, 0.2×2, 0.3×1 |

### Robustesse du point de fonctionnement final — `all`

Sensibilité = pire résultat quand chaque seuil bouge de ±1 pas de grille (sur les données de calibration). Décisif = part des paires où le critère change la décision.

| objectif | méthode | recall / spéc. (calibration) | pire recall ±1 pas | pire spéc. ±1 pas | décisif HIST | décisif MEAN | décisif PROP |
|---|---|---|---|---|---|---|---|
| spec@0.95 | classique | 0.964 / 0.974 | 0.875 | 0.922 | 0.24 | 0.24 | 0.47 |
| spec@0.95 | robuste | 0.969 / 0.975 | 0.880 | 0.922 | 0.24 | 0.24 | 0.47 |
| spec@0.98 | classique | 0.990 / 0.910 | 0.970 | 0.901 | 0.19 | 0.30 | 0.47 |
| spec@0.98 | robuste | 0.987 / 0.935 | 0.963 | 0.883 | 0.36 | 0.14 | 0.48 |
| bacc | classique | 0.969 / 0.972 | 0.883 | 0.928 | 0.29 | 0.13 | 0.41 |
| bacc | robuste | 0.969 / 0.972 | 0.883 | 0.928 | 0.29 | 0.13 | 0.41 |

### Sensibilité au prior de déploiement — `all` (part réelle de paires « différentes » ; énergie économisée / changements ratés)

| objectif | méthode | prior 0.05 | prior 0.1 | prior 0.2 | prior 0.3 | prior 0.5 |
|---|---|---|---|---|---|---|
| spec@0.95 | classique | 0.905 / 0.0020 | 0.859 / 0.0041 | 0.766 / 0.0082 | 0.674 / 0.0123 | 0.490 / 0.0204 |
| spec@0.95 | robuste | 0.906 / 0.0024 | 0.860 / 0.0048 | 0.768 / 0.0096 | 0.677 / 0.0144 | 0.494 / 0.0240 |
| spec@0.98 | classique | 0.845 / 0.0005 | 0.800 / 0.0011 | 0.711 / 0.0022 | 0.622 / 0.0032 | 0.444 / 0.0054 |
| spec@0.98 | robuste | 0.863 / 0.0016 | 0.819 / 0.0032 | 0.730 / 0.0065 | 0.642 / 0.0097 | 0.464 / 0.0162 |
| bacc | classique | 0.913 / 0.0025 | 0.867 / 0.0050 | 0.775 / 0.0101 | 0.683 / 0.0151 | 0.499 / 0.0252 |
| bacc | robuste | 0.901 / 0.0014 | 0.854 / 0.0028 | 0.761 / 0.0055 | 0.668 / 0.0083 | 0.481 / 0.0138 |

## Paramètres

```
{
 "cmd": "calibrate",
 "metrics": "metriques_gate.csv",
 "out": "results_gate",
 "positive_labels": "2",
 "scenarios": "hard,all",
 "objectives": "spec@0.95,spec@0.98,bacc",
 "k": 2,
 "grid_size": 25,
 "outer_folds": 5,
 "inner_folds": 3,
 "wilson_z": 1.645,
 "tol": 0.002,
 "robust_alpha": 0.1,
 "cal_blocks": 20,
 "inner_boot": 200,
 "boot": 2000,
 "boot_final": 200,
 "n_blocks": 40,
 "deploy_prior": 0.2,
 "deploy_priors": "0.05,0.1,0.2,0.3,0.5",
 "e_tx": 250.0,
 "e_gate": 3.0,
 "article_thresholds": null,
 "article_block": 0.5,
 "seed": 0,
 "article": null
}
```