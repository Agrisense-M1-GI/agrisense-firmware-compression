#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_gate.py — calibration ROBUSTE du vote majoritaire du garde-fou (AgriSense)

Mécanisme calibré (inchangé) : on transmet l'image si au moins K critères sur 3 dépassent leur
seuil (K=2 : vote majoritaire). Critères : HIST (histogramme global), MEAN (écart des médianes
RGB), PROP (% de blocs 32x32 dont la distance dépasse un « seuil de bloc »). Comparaison stricte (>).

Ce script ne compare PLUS d'autres règles (single_*, logreg) : il cherche seulement à améliorer
et fiabiliser le choix des seuils de CE vote. Trois méthodes de calibration sont comparées :
  classique  la méthode d'origine : meilleur point de grille sur tout l'entraînement
             (contrainte de recall via borne de Wilson, seuil de bloc choisi par CV interne)
  robuste    NOUVELLE méthode : chaque point de grille est évalué sur de nombreux rééchantillons
             par blocs temporels de l'entraînement ; on retient le point dont la performance
             est bonne dans (1-alpha) des rééchantillons (pas seulement en moyenne). Le seuil de
             bloc de PROP est choisi dans la même procédure. Contrairement à Wilson, cela tient
             compte de la corrélation temporelle entre paires.
  article    (optionnel) seuils fixes de l'article, évalués sur les mêmes folds pour mesurer
             le gain réel (--article-thresholds HIST,MEAN,PROP --article-block B)

Protocole d'évaluation (inchangé, défendable en relecture) :
  - labels : 0 = auto-paire (triviale), 1 = similaire, 2 = différent ; POSITIF = 2 (transmettre)
  - scénarios : hard (labels 1,2 : principal) ; all (0,1,2 : optimiste)
  - validation croisée imbriquée par BLOCS TEMPORELS avec PURGE des paires partageant une image
  - IC95 % par bootstrap par blocs, différences appariées vs `classique`
Diagnostics de robustesse ajoutés : recall du pire fold, stabilité des seuils entre folds, IC
bootstrap des seuils finaux, sensibilité à ±1 pas de grille, part de décisions où chaque critère
est décisif, énergie économisée pour plusieurs priors de déploiement.

Usage
  python calibrate_gate.py extract   --dataset dataset-gate-calibration.csv \\
        --images-dir /chemin/640X480-PNG --out metriques_gate.csv
  python calibrate_gate.py calibrate --metrics metriques_gate.csv --out results_gate \\
        --objectives "spec@0.95,spec@0.98,bacc" --deploy-prior 0.2 --e-tx 250 --e-gate 3
  # avec comparaison aux seuils actuels de l'article :
        --article-thresholds 0.05,10,30 --article-block 0.1

Dépendances : numpy, pandas, pillow (extract), matplotlib (figures, optionnel).
"""
import argparse
import itertools
import json
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

EPS = 1e-9
FEATS = ['hist', 'mean']          # + une variante prop@t (colonne 3)
CRIT = ['hist', 'mean', 'prop']
METHODS_BASE = ['classique', 'robuste']


# =============================================================================
# 1. Métriques (copie fidèle de garde_fou_energetique v2) + prop à plusieurs seuils de bloc
# =============================================================================

def _channel_histograms(pixels, bins):
    n_groupes, _, _ = pixels.shape
    bin_idx = np.minimum((pixels.astype(np.int32) * bins) // 256, bins - 1)
    hists = np.zeros((n_groupes, 3, bins), dtype=np.float64)
    group_offset = (np.arange(n_groupes) * bins)[:, None]
    for c in range(3):
        combined = (group_offset + bin_idx[:, :, c]).ravel()
        counts = np.bincount(combined, minlength=n_groupes * bins)
        hists[:, c, :] = counts.reshape(n_groupes, bins)
    hists /= hists.sum(axis=2, keepdims=True) + EPS
    return hists


def _bhattacharyya_distance(h1, h2):
    bc = np.clip(np.sum(np.sqrt(h1 * h2), axis=2), 1e-10, 1.0)
    return (-np.log(bc)).mean(axis=1)


def pair_metrics(img1, img2, block_size=32, hist_bins=4, block_thresholds=(0.5,)):
    h, w = img1.shape[:2]
    out = {}
    hg1 = _channel_histograms(img1.reshape(1, -1, 3), hist_bins)
    hg2 = _channel_histograms(img2.reshape(1, -1, 3), hist_bins)
    out['hist'] = float(_bhattacharyya_distance(hg1, hg2)[0])

    med1 = np.median(img1.reshape(-1, 3).astype(float), axis=0)
    med2 = np.median(img2.reshape(-1, 3).astype(float), axis=0)
    out['mean'] = float(np.mean(np.abs(med1 - med2)))

    n_by, n_bx = h // block_size, w // block_size
    diffs = []
    if n_by > 0 and n_bx > 0:
        def to_blocks(img):
            c = img[:n_by * block_size, :n_bx * block_size]
            return (c.reshape(n_by, block_size, n_bx, block_size, 3)
                     .transpose(0, 2, 1, 3, 4)
                     .reshape(n_by * n_bx, block_size * block_size, 3))
        diffs.append(_bhattacharyya_distance(_channel_histograms(to_blocks(img1), hist_bins),
                                             _channel_histograms(to_blocks(img2), hist_bins)))
    if h % block_size or w % block_size:      # blocs de bord (absents en 640x480 / 32)
        left = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if y + block_size <= n_by * block_size and x + block_size <= n_bx * block_size:
                    continue
                b1, b2 = img1[y:y + block_size, x:x + block_size], img2[y:y + block_size, x:x + block_size]
                if b1.size == 0 or b2.size == 0:
                    continue
                left.append(_bhattacharyya_distance(_channel_histograms(b1.reshape(1, -1, 3), hist_bins),
                                                    _channel_histograms(b2.reshape(1, -1, 3), hist_bins))[0])
        diffs.append(np.array(left))
    d = np.concatenate(diffs) if diffs else np.zeros(1)
    out['mblock'] = float(d.mean())
    for t in block_thresholds:
        out[f'prop@{t:g}'] = float((d > t).mean() * 100)
    return out


def _num(name):
    return int(re.findall(r'\d+', name)[-1])


def cmd_extract(a):
    try:
        from PIL import Image
    except ImportError:
        sys.exit("pillow manquant : pip install pillow")
    df = pd.read_csv(a.dataset, encoding='utf-8-sig')
    if not {'image_1', 'image_2', 'label'} <= set(df.columns):
        sys.exit("Colonnes attendues : image_1, image_2, label")
    bts = [float(x) for x in a.block_thresholds.split(',')]

    @lru_cache(maxsize=96)
    def load(name):
        return np.array(Image.open(os.path.join(a.images_dir, name)).convert('RGB'))

    rows, missing = [], 0
    for n, r in enumerate(df.itertuples(index=False), 1):
        try:
            m = pair_metrics(load(r.image_1), load(r.image_2), a.block_size, a.hist_bins, bts)
        except FileNotFoundError:
            missing += 1
            continue
        rows.append({'image_1': r.image_1, 'image_2': r.image_2, 'label': int(r.label),
                     'idx1': _num(r.image_1), 'idx2': _num(r.image_2), **m})
        if n % 250 == 0:
            print(f"{n}/{len(df)} paires...")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} paires -> {a.out}" + (f" ({missing} ignorées : image introuvable)" if missing else ""))


# =============================================================================
# 2. Outils statistiques
# =============================================================================

def wilson_lb(k, n, z):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        lb = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return np.where(n > 0, lb, 0.0)


def confusion(y, p):
    return np.array([np.sum((p == 1) & (y == 1)), np.sum((p == 1) & (y == 0)),
                     np.sum((p == 0) & (y == 1)), np.sum((p == 0) & (y == 0))], float)


def cm_metrics(c, prior, e_tx, e_gate):
    """c[..., 4] = tp, fp, fn, tn. Renvoie un dict de métriques (vectorisé pour le bootstrap)."""
    tp, fp, fn, tn = [np.asarray(c)[..., i].astype(float) for i in range(4)]
    rec = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mcc = (tp * tn - fp * fn) / np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    tx = prior * rec + (1 - prior) * (1 - spec)
    out = dict(recall=rec, specificity=spec, precision=prec,
               f1=2 * prec * rec / np.maximum(prec + rec, 1e-12),
               balanced_acc=(rec + spec) / 2, mcc=mcc,
               tx_rate_deploy=tx, missed_change_rate_deploy=prior * (1 - rec),
               precision_deploy=prior * rec / np.maximum(tx, 1e-12))
    if e_tx:
        out['energy_saving_deploy'] = 1 - e_gate / e_tx - tx
    return out


def auc_score(score, y):
    npos, nneg = int(y.sum()), int((1 - y).sum())
    r = pd.Series(score).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def roc_curve(score, y):
    o = np.argsort(-score, kind='stable')
    s, yy = score[o], y[o]
    tps, fps = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    return np.r_[0, fps[last] / max(fps[-1], 1)], np.r_[0, tps[last] / max(tps[-1], 1)]


# =============================================================================
# 3. Objectifs
# =============================================================================

class Obj:
    def __init__(self, s):
        self.name = s
        if s == 'bacc':
            self.kind, self.scale = 'bacc', 1.0
        elif s.startswith('spec@'):
            self.kind, self.r, self.scale = 'spec', float(s[5:]), 1.0
        elif s.startswith('cost:'):
            _, cfp, cfn = s.split(':')
            self.kind, self.cfp, self.cfn = 'cost', float(cfp), float(cfn)
            self.scale = max(self.cfp, self.cfn)
        else:
            raise ValueError(f"objectif inconnu : {s}")

    def value(self, tp, fp, fn, tn, z):
        """Score à maximiser ; -inf = infaisable (contrainte de recall non tenue)."""
        rec = tp / np.maximum(tp + fn, 1)
        spec = tn / np.maximum(tn + fp, 1)
        if self.kind == 'bacc':
            return (rec + spec) / 2
        if self.kind == 'spec':
            return np.where(wilson_lb(tp, tp + fn, z) >= self.r, spec, -np.inf)
        return -(self.cfp * fp + self.cfn * fn) / (tp + fp + fn + tn)

    def utility(self, tp, fp, fn, tn):
        """Utilité scalaire poolée (sélection du seuil de bloc en boucle interne, méthode classique)."""
        rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if self.kind == 'bacc':
            return (rec + spec) / 2
        if self.kind == 'spec':
            return spec - 5 * max(0.0, self.r - rec)
        return -(self.cfp * fp + self.cfn * fn) / (tp + fp + fn + tn)


def choose(tp, fp, fn, tn, obj, z, tol):
    """Méthode classique : centre du plateau quasi-optimal de la grille. Si la contrainte est
    infaisable -> recall maximal."""
    shape = tp.shape
    tp, fp, fn, tn = [a.ravel().astype(float) for a in (tp, fp, fn, tn)]
    v = obj.value(tp, fp, fn, tn, z)
    feasible = bool(np.isfinite(v).any())
    if not feasible:
        v = tp / np.maximum(tp + fn, 1) + 1e-3 * tn / np.maximum(tn + fp, 1)
    near = np.where(v >= v.max() - tol * obj.scale)[0]
    coords = np.stack(np.unravel_index(near, shape), 1).astype(float)
    j = near[np.argmin(((coords - coords.mean(0)) ** 2).sum(1))]
    return int(j), feasible


# =============================================================================
# 4. Le vote : grille, ajustement classique, prédiction
# =============================================================================

def _grid(x, n, qlo=2.0, qhi=98.0):
    return np.unique(np.percentile(x, np.linspace(qlo, qhi, n)))


def table_vote(X, y, G, k):
    grids = [_grid(X[:, m], G) for m in range(3)]
    G1, G2, G3 = map(len, grids)
    cnt = {}
    for cls in (1, 0):
        Xc = X[y == cls]
        a, b, c = [(Xc[:, m][:, None] > grids[m][None, :]).astype(np.uint8) for m in range(3)]
        bc = b[:, :, None] + c[:, None, :]
        res = np.zeros((G1, G2, G3), np.int64)
        for i in range(G1):
            res[i] = ((a[:, i][:, None, None] + bc) >= k).sum(0)
        cnt[cls] = res
    return grids, cnt, int(y.sum()), int((1 - y).sum())


def fit_vote(k, X, y, objs, cfg):
    """Méthode classique : ajuste les 3 seuils sur (X, y) pour CHAQUE objectif."""
    if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
        raise ValueError("une classe est absente du jeu d'entraînement")
    grids, cnt, npos, nneg = table_vote(X, y, cfg.grid_size, k)
    tp, fp = cnt[1], cnt[0]
    fn, tn = npos - tp, nneg - fp
    models = {}
    for o in objs:
        j, feas = choose(tp, fp, fn, tn, o, cfg.wilson_z, cfg.tol)
        ijk = np.unravel_index(j, tp.shape)
        models[o.name] = dict(rule=f'vote{k}', k=k, th=[float(grids[m][ijk[m]]) for m in range(3)],
                              feasible=feas)
    return models


def predict(model, X):
    s = sum((X[:, m] > model['th'][m]).astype(int) for m in range(3))
    return (s >= model['k']).astype(int)


def thresholds_of(model):
    return {'thr_hist': model['th'][0], 'thr_mean': model['th'][1], 'thr_prop': model['th'][2]}


def pivot_rates(th, X, k):
    """Part des paires où chaque critère est DÉCISIF (le retourner changerait la décision)."""
    f = np.stack([X[:, m] > th[m] for m in range(3)], 1).astype(int)
    s = f.sum(1)
    return [float(np.mean((s - f[:, m]) == k - 1)) for m in range(3)]


def sensitivity(th, X, y, cfg):
    """Pire recall / pire spécificité quand chaque seuil bouge de ±1 pas de grille (27 combinaisons)."""
    grids = [_grid(X[:, m], cfg.grid_size) for m in range(3)]
    idx = [int(np.argmin(np.abs(grids[m] - th[m]))) for m in range(3)]
    wr, ws = 1.0, 1.0
    for d in itertools.product((-1, 0, 1), repeat=3):
        t = [grids[m][int(np.clip(idx[m] + d[m], 0, len(grids[m]) - 1))] for m in range(3)]
        tp, fp, fn, tn = confusion(y, predict(dict(k=cfg.k, th=t), X))
        wr, ws = min(wr, tp / max(tp + fn, 1)), min(ws, tn / max(tn + fp, 1))
    return dict(worst_recall=float(wr), worst_specificity=float(ws))


# =============================================================================
# 5. Découpage par blocs temporels + purge ; calibration classique
# =============================================================================

def blocked_folds(anchor, K):
    order = np.argsort(anchor, kind='stable')
    fold = np.empty(len(anchor), int)
    fold[order] = (np.arange(len(anchor)) * K) // len(anchor)
    return fold


def purged_train_mask(test_mask, idx1, idx2):
    """Entraînement = paires sans AUCUNE image commune avec le fold de test."""
    imgs = np.union1d(idx1[test_mask], idx2[test_mask])
    return ~(np.isin(idx1, imgs) | np.isin(idx2, imgs))


def fit_classic(feats, y, idx1, idx2, anchor, objs, cfg, pvs):
    """Méthode classique : seuil de bloc (prop@t) choisi par CV interne (blocs + purge)."""
    fold = blocked_folds(anchor, cfg.inner_folds)
    full, util = {}, {o.name: {} for o in objs}
    for pv in pvs:
        X = feats[pv]
        full[pv] = fit_vote(cfg.k, X, y, objs, cfg)
        cm = {o.name: np.zeros(4) for o in objs}
        for f in range(cfg.inner_folds):
            te = fold == f
            trm = purged_train_mask(te, idx1, idx2)
            if te.sum() < 20 or trm.sum() < 50 or len(np.unique(y[trm])) < 2:
                continue
            ms = fit_vote(cfg.k, X[trm], y[trm], objs, cfg)
            for o in objs:
                cm[o.name] += confusion(y[te], predict(ms[o.name], X[te]))
        for o in objs:
            util[o.name][pv] = o.utility(*cm[o.name]) if cm[o.name].sum() else -np.inf
    out = {}
    for o in objs:
        best = max(pvs, key=lambda p: util[o.name][p])
        m = dict(full[best][o.name])
        m['pv'] = best
        out[o.name] = m
    return out


# =============================================================================
# 6. Calibration ROBUSTE : choix du point de grille sur des rééchantillons par blocs
# =============================================================================

def vote_block_tables(X, y, blk, nb, grids, k):
    """Pour chaque point (HIST, MEAN, PROP) de la grille et chaque BLOC temporel : nombre de
    positifs détectés (tp) et de négatifs transmis (fp). Un rééchantillonnage de blocs se réduit
    alors à une somme pondérée de ces tableaux (très rapide)."""
    G = [len(g) for g in grids]
    shape, tot = tuple(G), G[0] * G[1] * G[2]
    tp = np.zeros((nb, tot), np.float32)
    fp = np.zeros((nb, tot), np.float32)
    npos = np.bincount(blk[y == 1], minlength=nb).astype(np.float32)
    nneg = np.bincount(blk[y == 0], minlength=nb).astype(np.float32)
    for cls, out in ((1, tp), (0, fp)):
        for b in range(nb):
            m = (blk == b) & (y == cls)
            if not m.any():
                continue
            Xc = X[m]
            a, bb, c = [(Xc[:, j][:, None] > grids[j][None, :]).astype(np.uint8) for j in range(3)]
            bc = bb[:, :, None] + c[:, None, :]
            res = np.zeros(shape, np.int64)
            for i in range(G[0]):
                res[i] = ((a[:, i][:, None, None] + bc) >= k).sum(0)
            out[b] = res.ravel()
    return dict(grids=grids, shape=shape, tp=tp, fp=fp, npos=npos, nneg=nneg)


def build_tables(feats, y, anchor, pvs, cfg):
    nb = int(min(cfg.cal_blocks, max(len(y) // 25, 5)))
    blk = blocked_folds(anchor, nb)
    tabs = {}
    for pv in pvs:
        X = feats[pv]
        grids = [_grid(X[:, m], cfg.grid_size) for m in range(3)]
        tabs[pv] = vote_block_tables(X, y, blk, nb, grids, cfg.k)
    return tabs


def _lowq(v, alpha):
    """Quantile alpha (par rapport aux rééchantillons, axe 0) — valeur atteinte dans (1-alpha) des cas."""
    kth = int(np.floor(alpha * (v.shape[0] - 1)))
    return np.partition(v, kth, axis=0)[kth]


def _robust_score(obj, rec, spec, NP, NN, alpha):
    """Score robuste (à maximiser) de chaque point de la grille + faisabilité."""
    if obj.kind == 'bacc':
        return _lowq((rec + spec) / 2, alpha), np.ones(rec.shape[1], bool)
    if obj.kind == 'spec':
        rq = _lowq(rec, alpha)                    # recall garanti dans (1-alpha) des rééchantillons
        feas = rq >= obj.r
        if feas.any():
            return np.where(feas, spec.mean(0), -np.inf), feas
        return rq + 1e-3 * spec.mean(0), feas     # infaisable -> recall garanti maximal
    cost = obj.cfp * (1 - spec) * NN / (NP + NN) + obj.cfn * (1 - rec) * NP / (NP + NN)
    return _lowq(-cost, alpha), np.ones(rec.shape[1], bool)


def _pick(score, shape, tol):
    """Centre du plateau quasi-optimal (stabilité) plutôt que le premier maximum."""
    m = score[np.isfinite(score)].max()
    near = np.where(score >= m - tol)[0]
    coords = np.stack(np.unravel_index(near, shape), 1).astype(float)
    j = near[np.argmin(((coords - coords.mean(0)) ** 2).sum(1))]
    return int(j), float(m)


def robust_select_all(tabs, objs, cfg, rng, rows=None, B=200):
    """Choisit (seuil de bloc, HIST, MEAN, PROP) pour chaque objectif. `rows` = rééchantillon
    externe de blocs (pour l'IC des seuils). Mêmes tirages pour tous les seuils de bloc."""
    first = next(iter(tabs.values()))
    npos0 = first['npos'] if rows is None else first['npos'][rows]
    nb = len(npos0)
    draws = rng.integers(0, nb, (B, nb))
    W = (draws[:, :, None] == np.arange(nb)[None, None, :]).sum(1).astype(np.float32)
    best = {}
    for pv, t in tabs.items():
        tp, fp, npos, nneg = t['tp'], t['fp'], t['npos'], t['nneg']
        if rows is not None:
            tp, fp, npos, nneg = tp[rows], fp[rows], npos[rows], nneg[rows]
        NP = np.maximum(W @ npos, 1)[:, None]
        NN = np.maximum(W @ nneg, 1)[:, None]
        rec = (W @ tp) / NP
        spec = 1 - (W @ fp) / NN
        for o in objs:
            sc, feas = _robust_score(o, rec, spec, NP, NN, cfg.robust_alpha)
            j, m = _pick(sc, t['shape'], cfg.tol * o.scale)
            key = (bool(feas.any()), m)
            if o.name not in best or key > best[o.name][0]:
                idx = np.unravel_index(j, t['shape'])
                best[o.name] = (key, dict(rule=f'vote{cfg.k}', k=cfg.k, pv=pv, feasible=bool(feas.any()),
                                          th=[float(t['grids'][q][idx[q]]) for q in range(3)],
                                          robust_score=m))
    return {n: v[1] for n, v in best.items()}


def robust_ci(tabs, objs, cfg, rng, n_boot, finals):
    """IC95 % des seuils finaux : on rééchantillonne les blocs, on refait TOUTE la sélection robuste.
    Les IC sont donnés à seuil de bloc fixé (celui du modèle final), sinon PROP n'est pas comparable."""
    nb = len(next(iter(tabs.values()))['npos'])
    acc = {o.name: [] for o in objs}
    for _ in range(n_boot):
        rows = rng.integers(0, nb, nb)
        ms = robust_select_all(tabs, objs, cfg, rng, rows, min(cfg.inner_boot, 100))
        for o in objs:
            acc[o.name].append(dict(pv=ms[o.name]['pv'], **thresholds_of(ms[o.name])))
    out = {}
    for o in objs:
        d = pd.DataFrame(acc[o.name])
        pv_final = finals[o.name]['pv']
        sub = d[d.pv == pv_final]
        n_cond = len(sub)
        if n_cond < 10:
            sub = d
        out[o.name] = dict(
            ci95={c: [float(np.percentile(sub[c], 2.5)), float(np.percentile(sub[c], 97.5))]
                  for c in ('thr_hist', 'thr_mean', 'thr_prop')},
            pv_freq={k: float(v) for k, v in d.pv.value_counts(normalize=True).items()},
            n_cond=int(n_cond), n_boot=int(len(d)))
    return out


# =============================================================================
# 7. Validation croisée imbriquée + tables bootstrap
# =============================================================================

def run_cv(df, objs, pvs, cfg, tag, methods):
    y = df['y'].values
    idx1, idx2 = df['idx1'].values, df['idx2'].values
    anchor = np.minimum(idx1, idx2)
    feats = {pv: df[FEATS + [pv]].values for pv in pvs}
    fold = blocked_folds(anchor, cfg.outer_folds)
    preds = {(m, o.name): np.full(len(df), -1) for m in methods for o in objs}
    rng = np.random.default_rng(cfg.seed + 10)
    frows = []
    for f in range(cfg.outer_folds):
        te = fold == f
        trm = purged_train_mask(te, idx1, idx2)
        print(f"[{tag}] fold {f + 1}/{cfg.outer_folds} : test={te.sum()} train(purgé)={trm.sum()} "
              f"(retirées par purge : {(~te).sum() - trm.sum()})", flush=True)
        ftr = {pv: feats[pv][trm] for pv in pvs}
        fitted = {'classique': fit_classic(ftr, y[trm], idx1[trm], idx2[trm], anchor[trm], objs, cfg, pvs),
                  'robuste': robust_select_all(build_tables(ftr, y[trm], anchor[trm], pvs, cfg),
                                               objs, cfg, rng, None, cfg.inner_boot)}
        if 'article' in methods:
            fitted['article'] = {o.name: cfg.article for o in objs}
        for meth in methods:
            for o in objs:
                m = fitted[meth][o.name]
                p = predict(m, feats[m['pv']][te])
                preds[(meth, o.name)][te] = p
                tp, fp, fn, tn = confusion(y[te], p)
                frows.append(dict(scenario=tag, fold=f, method=meth, objective=o.name, pv=m['pv'],
                                  feasible=m.get('feasible', True), n_train=int(trm.sum()),
                                  n_test=int(te.sum()), test_recall=tp / max(tp + fn, 1),
                                  test_spec=tn / max(tn + fp, 1), **thresholds_of(m)))
    return preds, pd.DataFrame(frows), anchor


def bootstrap_tables(df, preds, objs, methods, anchor, cfg, tag, ref='classique'):
    y = df['y'].values
    nb = min(cfg.n_blocks, max(len(df) // 25, 5))
    bid = blocked_folds(anchor, nb)
    rng = np.random.default_rng(cfg.seed)
    draws = rng.integers(0, nb, (cfg.boot, nb))
    rows = []
    for o in objs:
        counts = {}
        for r in methods:
            p = preds[(r, o.name)]
            counts[r] = np.stack([confusion(y[bid == b], p[bid == b]) for b in range(nb)])
        est, boot = {}, {}
        for r in methods:
            est[r] = cm_metrics(counts[r].sum(0), cfg.deploy_prior, cfg.e_tx, cfg.e_gate)
            boot[r] = cm_metrics(counts[r][draws].sum(1), cfg.deploy_prior, cfg.e_tx, cfg.e_gate)
        for r in methods:
            for m in est[r]:
                lo, hi = np.percentile(boot[r][m], [2.5, 97.5])
                row = dict(scenario=tag, objective=o.name, method=r, metric=m, est=float(est[r][m]),
                           lo=lo, hi=hi, n=len(y))
                if ref in methods and r != ref:
                    dd = boot[r][m] - boot[ref][m]
                    row.update(diff_vs_ref=float(est[r][m] - est[ref][m]),
                               diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5))
                rows.append(row)
    return pd.DataFrame(rows)


def prior_table(df, preds, objs, methods, priors, cfg, tag):
    """Projection énergie / changements ratés pour plusieurs priors de déploiement."""
    y = df['y'].values
    rows = []
    for o in objs:
        for m in methods:
            c = confusion(y, preds[(m, o.name)])
            for pr in priors:
                cm = cm_metrics(c, pr, cfg.e_tx, cfg.e_gate)
                rows.append(dict(scenario=tag, objective=o.name, method=m, prior=pr,
                                 **{k: float(v) for k, v in cm.items()}))
    return pd.DataFrame(rows)


def auc_table(df, pvs, anchor, cfg, tag):
    y = df['y'].values
    nb = min(cfg.n_blocks, max(len(df) // 25, 5))
    bid = blocked_folds(anchor, nb)
    blocks = [np.where(bid == b)[0] for b in range(nb)]
    rng = np.random.default_rng(cfg.seed + 1)
    B = min(cfg.boot, 500)
    samples = [np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)]) for _ in range(B)]
    rows = []
    for col in ['hist', 'mean', 'mblock'] + list(pvs):
        if col not in df:
            continue
        s = df[col].values
        bs = [auc_score(s[i], y[i]) for i in samples if 0 < y[i].sum() < len(i)]
        rows.append(dict(scenario=tag, feature=col, auc=auc_score(s, y),
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows)


# =============================================================================
# 8. Ajustement final (toutes les données) + diagnostics de robustesse
# =============================================================================

def final_fit(df, objs, pvs, cfg):
    y = df['y'].values
    idx1, idx2 = df['idx1'].values, df['idx2'].values
    anchor = np.minimum(idx1, idx2)
    feats = {pv: df[FEATS + [pv]].values for pv in pvs}
    rng = np.random.default_rng(cfg.seed + 2)
    cl = fit_classic(feats, y, idx1, idx2, anchor, objs, cfg, pvs)
    tabs = build_tables(feats, y, anchor, pvs, cfg)
    ro = robust_select_all(tabs, objs, cfg, rng, None, cfg.inner_boot)
    ci = robust_ci(tabs, objs, cfg, rng, cfg.boot_final, ro)
    out = {}
    for o in objs:
        rec = {}
        for name, m in (('classique', cl[o.name]), ('robuste', ro[o.name])):
            X = feats[m['pv']]
            tp, fp, fn, tn = confusion(y, predict(m, X))
            rec[name] = dict(model=m,
                             resub=dict(recall=float(tp / (tp + fn)), specificity=float(tn / (tn + fp))),
                             pivot=dict(zip(CRIT, pivot_rates(m['th'], X, cfg.k))),
                             sensitivity=sensitivity(m['th'], X, y, cfg))
        rec['robuste'].update(ci[o.name])
        out[o.name] = rec
    return out


# =============================================================================
# 9. Sorties : résumé Markdown, seuils, figures
# =============================================================================

KEY = ['recall', 'specificity', 'balanced_acc', 'mcc']


def _fmt(row):
    return f"{row.est:.3f} [{row.lo:.3f}, {row.hi:.3f}]"


def _ci(ci, c):
    return f"[{ci[c][0]:.4g}, {ci[c][1]:.4g}]"


def export_thresholds(finals, res):
    out = {}
    for tag, objs in finals.items():
        for obj, rec in objs.items():
            m, r = rec['robuste']['model'], rec['robuste']
            perf = {}
            sub = res[(res.scenario == tag) & (res.objective == obj) & (res.method == 'robuste')]
            for met in ('recall', 'specificity', 'energy_saving_deploy'):
                s = sub[sub.metric == met]
                if len(s):
                    perf[f'cv_{met}'] = float(s.iloc[0].est)
            out.setdefault(tag, {})[obj] = dict(
                rule=f"vote {m['k']} sur 3 (comparaison stricte >)",
                HIST=m['th'][0], MEAN=m['th'][1], PROP_pct_blocs=m['th'][2],
                seuil_de_bloc=float(m['pv'][5:]), ci95=r['ci95'], pv_freq=r['pv_freq'],
                feasible=m['feasible'], **perf)
    return out


def print_thresholds(finals, cfg):
    print("\n== SEUILS RECOMMANDÉS (méthode robuste ; vote "
          f"{cfg.k} sur 3 ; transmettre si le critère est > seuil) ==")
    for tag, objs in finals.items():
        for obj, rec in objs.items():
            m = rec['robuste']['model']
            print(f"[{tag}] {obj:10s} HIST>{m['th'][0]:.5f}  MEAN>{m['th'][1]:.3f}  "
                  f"PROP>{m['th'][2]:.2f}%  (seuil de bloc {m['pv'][5:]})"
                  + ("" if m['feasible'] else "  [contrainte de recall NON tenable]"))


def summary_md(res, folds, auc, finals, prior_df, scen_info, cfg, methods):
    L = ["# Calibration robuste du vote majoritaire", "",
         f"Règle : transmettre si **au moins {cfg.k} critères sur 3** dépassent leur seuil "
         "(comparaison stricte `>`). PROP = % de blocs 32×32 dont la distance de Bhattacharyya "
         "dépasse le « seuil de bloc ».", ""]
    for tag in scen_info:
        L += [f"## Seuils recommandés — scénario `{tag}` (méthode robuste)", "",
              "| objectif | HIST > | MEAN > | PROP > (%) | seuil de bloc | IC95 HIST | IC95 MEAN | IC95 PROP* |",
              "|---|---|---|---|---|---|---|---|"]
        for obj, rec in finals[tag].items():
            m, r = rec['robuste']['model'], rec['robuste']
            L.append(f"| {obj} | {m['th'][0]:.5f} | {m['th'][1]:.3f} | {m['th'][2]:.2f} | {m['pv'][5:]} | "
                     f"{_ci(r['ci95'], 'thr_hist')} | {_ci(r['ci95'], 'thr_mean')} | {_ci(r['ci95'], 'thr_prop')} |")
        L += ["", "\\* IC à seuil de bloc fixé (bootstrap par blocs temporels, refait toute la sélection). "
              "Un IC étroit = seuil stable.", ""]
    for tag, info in scen_info.items():
        L += [f"## Scénario `{tag}` — {info}", ""]
        a = auc[auc.scenario == tag]
        L += ["**AUC des critères bruts (IC95 % bootstrap par blocs)**", "", "| critère | AUC |", "|---|---|"]
        L += [f"| {r.feature} | {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}] |" for r in a.itertuples()] + [""]
        for obj in res[res.scenario == tag].objective.unique():
            sub = res[(res.scenario == tag) & (res.objective == obj)]
            metrics = KEY + [m for m in ('energy_saving_deploy', 'tx_rate_deploy') if m in set(sub.metric)]
            L += [f"### Objectif `{obj}` (performance hors-échantillon, CV imbriquée)", "",
                  "| méthode | " + " | ".join(metrics) + " | folds recall≥cible | recall pire fold |",
                  "|---|" + "---|" * (len(metrics) + 2)]
            for meth in methods:
                cells = [_fmt(sub[(sub.method == meth) & (sub.metric == m)].iloc[0]) for m in metrics]
                ff = folds[(folds.scenario == tag) & (folds.objective == obj) & (folds.method == meth)]
                ok = f"{int((ff.test_recall >= float(obj[5:])).sum())}/{len(ff)}" if obj.startswith('spec@') else ""
                L.append(f"| {meth} | " + " | ".join(cells) + f" | {ok} | {ff.test_recall.min():.3f} |")
            L += [""]
            d = sub[(sub.metric.isin(['specificity', 'recall'])) & sub.diff_vs_ref.notna()]
            if len(d):
                L += ["Différences appariées vs `classique` (IC95 % ; ✱ = IC exclut 0) :", ""]
                for r in d.itertuples():
                    star = "✱" if (r.diff_lo > 0 or r.diff_hi < 0) else ""
                    L.append(f"- {r.method} / {r.metric} : {r.diff_vs_ref:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}] {star}")
                L += [""]
            L += ["Stabilité des seuils entre folds (médiane [min, max]) :", "",
                  "| méthode | HIST | MEAN | PROP | seuil de bloc (folds) |", "|---|---|---|---|---|"]
            for meth in [m for m in methods if m != 'article']:
                ff = folds[(folds.scenario == tag) & (folds.objective == obj) & (folds.method == meth)]
                cells = [f"{ff[c].median():.4g} [{ff[c].min():.4g}, {ff[c].max():.4g}]"
                         for c in ('thr_hist', 'thr_mean', 'thr_prop')]
                pvc = ", ".join(f"{k[5:]}×{v}" for k, v in ff.pv.value_counts().items())
                L.append(f"| {meth} | " + " | ".join(cells) + f" | {pvc} |")
            L += [""]
        L += [f"### Robustesse du point de fonctionnement final — `{tag}`", "",
              "Sensibilité = pire résultat quand chaque seuil bouge de ±1 pas de grille (sur les données de calibration). "
              "Décisif = part des paires où le critère change la décision.", "",
              "| objectif | méthode | recall / spéc. (calibration) | pire recall ±1 pas | pire spéc. ±1 pas | "
              "décisif HIST | décisif MEAN | décisif PROP |", "|---|---|---|---|---|---|---|---|"]
        for obj, rec in finals[tag].items():
            for meth in ('classique', 'robuste'):
                r = rec[meth]
                L.append(f"| {obj} | {meth} | {r['resub']['recall']:.3f} / {r['resub']['specificity']:.3f} | "
                         f"{r['sensitivity']['worst_recall']:.3f} | {r['sensitivity']['worst_specificity']:.3f} | "
                         f"{r['pivot']['hist']:.2f} | {r['pivot']['mean']:.2f} | {r['pivot']['prop']:.2f} |")
        L += [""]
        pd_t = prior_df[prior_df.scenario == tag]
        priors = sorted(pd_t.prior.unique())
        has_e = 'energy_saving_deploy' in pd_t
        L += [f"### Sensibilité au prior de déploiement — `{tag}` "
              "(part réelle de paires « différentes » ; " +
              ("énergie économisée / changements ratés)" if has_e else "taux de transmission / changements ratés)"), "",
              "| objectif | méthode | " + " | ".join(f"prior {p:g}" for p in priors) + " |",
              "|---|---|" + "---|" * len(priors)]
        for (obj, meth), g in pd_t.groupby(['objective', 'method'], sort=False):
            cells = []
            for p in priors:
                r = g[g.prior == p].iloc[0]
                first = r.energy_saving_deploy if has_e else r.tx_rate_deploy
                cells.append(f"{first:.3f} / {r.missed_change_rate_deploy:.4f}")
            L.append(f"| {obj} | {meth} | " + " | ".join(cells) + " |")
        L += [""]
    L += ["## Paramètres", "", "```",
          json.dumps({k: v for k, v in vars(cfg).items() if k != 'func'}, indent=1, default=str), "```"]
    return "\n".join(L)


def _boxplot(ax, groups, names):
    try:
        ax.boxplot(groups, tick_labels=names, showfliers=False)     # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(groups, labels=names, showfliers=False)          # anciennes versions


def make_figures(out, data, res, folds, auc_df, pvs, objs, methods):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : figures ignorées")
        return
    df = data.get('all', next(iter(data.values())))
    ref_tag = 'hard' if 'hard' in data else next(iter(data))
    best_pv = auc_df[(auc_df.scenario == ref_tag) & auc_df.feature.isin(pvs)].sort_values('auc').iloc[-1].feature
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a_, col in zip(ax, ['hist', 'mean', best_pv]):
        labs = sorted(df.label.unique())
        _boxplot(a_, [df[df.label == l][col] for l in labs], [f"label {l}" for l in labs])
        a_.set_title(col)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_distributions.png'), dpi=200)
    plt.close(fig)
    mk = dict(zip(methods, 'os^'))
    for tag, d in data.items():
        y = d['y'].values
        fig, a_ = plt.subplots(figsize=(5.4, 5))
        for col in ['hist', 'mean', best_pv]:
            fpr, tpr = roc_curve(d[col].values, y)
            a_.plot(fpr, tpr, lw=1.0, alpha=.6, label=f"{col} seul (AUC {auc_score(d[col].values, y):.3f})")
        a_.plot([0, 1], [0, 1], 'k:', lw=.6)
        sub = res[res.scenario == tag]
        for (meth, obj), g in sub.groupby(['method', 'objective']):
            r, s = g[g.metric == 'recall'].iloc[0], g[g.metric == 'specificity'].iloc[0]
            a_.errorbar(1 - s.est, r.est, xerr=[[s.est - s.lo], [s.hi - s.est]],
                        yerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk[meth], ms=5, lw=.6, alpha=.85,
                        label=f"vote · {meth} · {obj}")
        a_.set_xlabel("1 − specificity (fausses transmissions)")
        a_.set_ylabel("recall (changements détectés)")
        a_.set_title(f"Scénario {tag} — points de fonctionnement CV")
        a_.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'fig_roc_{tag}.png'), dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(2, len(objs), figsize=(3.8 * len(objs), 5.6), squeeze=False)
        for j, o in enumerate(objs):
            for meth in methods:
                g = folds[(folds.scenario == tag) & (folds.objective == o.name) & (folds.method == meth)]
                g = g.sort_values('fold')
                ax[0][j].plot(g.fold + 1, g.test_recall, marker=mk[meth], label=meth)
                ax[1][j].plot(g.fold + 1, g.test_spec, marker=mk[meth], label=meth)
            if o.kind == 'spec':
                ax[0][j].axhline(o.r, color='k', ls=':', lw=.8)
            ax[0][j].set_title(o.name)
            ax[0][j].set_ylabel("recall (test)")
            ax[1][j].set_ylabel("specificity (test)")
            ax[1][j].set_xlabel("fold")
        ax[0][0].legend(fontsize=7)
        fig.suptitle(f"Scénario {tag} — performance par fold (robustesse)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'fig_folds_{tag}.png'), dpi=200)
        plt.close(fig)


# =============================================================================
# 10. Commande calibrate
# =============================================================================

def parse_article(a, pvs):
    if not a.article_thresholds:
        return None
    try:
        h, m, p = [float(x) for x in a.article_thresholds.split(',')]
    except ValueError:
        sys.exit("--article-thresholds attend HIST,MEAN,PROP (ex. 0.05,10,30)")
    pv = f"prop@{float(a.article_block):g}"
    if pv not in pvs:
        sys.exit(f"--article-block : {pv} absent des métriques (disponibles : {', '.join(pvs)})")
    return dict(rule=f'vote{a.k}', k=a.k, th=[h, m, p], pv=pv, feasible=True)


def cmd_calibrate(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.metrics)
    pvs = [c for c in df.columns if c.startswith('prop@')]
    need = {'image_1', 'image_2', 'label', 'idx1', 'idx2', 'hist', 'mean'}
    if not need <= set(df.columns) or not pvs:
        sys.exit("CSV de métriques invalide : relance `extract` (colonnes prop@t requises).")
    if a.k not in (1, 2, 3):
        sys.exit("--k doit valoir 1, 2 ou 3 (2 = vote majoritaire)")
    pvs = sorted(pvs, key=lambda c: float(c[5:]))
    a.article = parse_article(a, pvs)
    methods = METHODS_BASE + (['article'] if a.article else [])
    objs = [Obj(s.strip()) for s in a.objectives.split(',')]
    priors = [float(x) for x in a.deploy_priors.split(',')]
    pos = [int(x) for x in a.positive_labels.split(',')]

    self_pair = df.idx1 == df.idx2
    print("\n== Dataset ==")
    print(pd.crosstab(df.label, self_pair.map({True: 'auto-paire', False: 'paire distincte'}).rename('type')).to_string())
    if self_pair.any():
        print(f"Auto-paires : métriques max = hist {df[self_pair]['hist'].max():.2g}, mean {df[self_pair]['mean'].max():.2g}"
              " -> triviales (== 0) ; scénario 'all' = borne optimiste.\n")
    df['y'] = df.label.isin(pos).astype(int)

    nonself = sorted(l for l in df.label.unique()
                     if not (df[df.label == l].idx1 == df[df.label == l].idx2).all())
    scen = {'hard': (nonself, "labels non triviaux (auto-paires exclues) — scénario principal"),
            'all': (sorted(df.label.unique()), "tous labels (optimiste : inclut les auto-paires triviales)")}
    wanted = [s.strip() for s in a.scenarios.split(',')]

    data, res_all, fold_all, auc_all, prior_all, finals, info = {}, [], [], [], [], {}, {}
    for tag in wanted:
        labs, desc = scen[tag]
        d = df[df.label.isin(labs)].reset_index(drop=True)
        if d.y.nunique() < 2:
            print(f"[{tag}] une seule classe -> ignoré")
            continue
        data[tag], info[tag] = d, f"{desc} ; n={len(d)}, positifs={int(d.y.sum())}"
        print(f"\n== Scénario {tag} : {info[tag]} ==")
        preds, folds, anchor = run_cv(d, objs, pvs, a, tag, methods)
        pd.DataFrame({'image_1': d.image_1, 'image_2': d.image_2, 'label': d.label, 'y': d.y,
                      **{f"pred|{m}|{o}": p for (m, o), p in preds.items()}}
                     ).to_csv(os.path.join(a.out, f'cv_predictions_{tag}.csv'), index=False)
        res_all.append(bootstrap_tables(d, preds, objs, methods, anchor, a, tag))
        prior_all.append(prior_table(d, preds, objs, methods, priors, a, tag))
        fold_all.append(folds)
        auc_all.append(auc_table(d, pvs, anchor, a, tag))
        print(f"[{tag}] ajustement final + bootstrap des seuils ({a.boot_final} rééchantillons)...", flush=True)
        finals[tag] = final_fit(d, objs, pvs, a)

    if not res_all:
        sys.exit("Aucun scénario exploitable.")
    res, folds, auc, prior_df = pd.concat(res_all), pd.concat(fold_all), pd.concat(auc_all), pd.concat(prior_all)
    res.to_csv(os.path.join(a.out, 'results_cv.csv'), index=False)
    folds.to_csv(os.path.join(a.out, 'fold_thresholds.csv'), index=False)
    auc.to_csv(os.path.join(a.out, 'auc_features.csv'), index=False)
    prior_df.to_csv(os.path.join(a.out, 'prior_sensitivity.csv'), index=False)
    with open(os.path.join(a.out, 'final_models.json'), 'w') as f:
        json.dump(finals, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    with open(os.path.join(a.out, 'thresholds.json'), 'w', encoding='utf8') as f:
        json.dump(export_thresholds(finals, res), f, indent=1, ensure_ascii=False)
    md = summary_md(res, folds, auc, finals, prior_df, info, a, methods)
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf8') as f:
        f.write(md)
    print("\n" + md)
    print_thresholds(finals, a)          # AVANT les figures : un souci matplotlib ne masque plus les seuils
    print(f"\nFichiers écrits dans {a.out}/  (seuils : thresholds.json et summary.md)")
    try:
        make_figures(a.out, data, res, folds, auc, pvs, objs, methods)
    except Exception as e:               # les figures sont accessoires
        print(f"[avertissement] figures non générées : {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--dataset', required=True)
    e.add_argument('--images-dir', required=True)
    e.add_argument('--out', default='metriques_gate.csv')
    e.add_argument('--block-size', type=int, default=32)
    e.add_argument('--hist-bins', type=int, default=4)
    e.add_argument('--block-thresholds', default='0.05,0.1,0.2,0.3,0.5',
                   help="seuils de distance de bloc testés pour prop (calibrés avec les autres seuils)")
    c = sub.add_parser('calibrate')
    c.add_argument('--metrics', required=True)
    c.add_argument('--out', default='results_gate')
    c.add_argument('--positive-labels', default='2')
    c.add_argument('--scenarios', default='hard,all')
    c.add_argument('--objectives', default='spec@0.95,spec@0.98,bacc')
    c.add_argument('--k', type=int, default=2, help="nombre de critères requis sur 3 (2 = vote majoritaire)")
    c.add_argument('--grid-size', type=int, default=25)
    c.add_argument('--outer-folds', type=int, default=5)
    c.add_argument('--inner-folds', type=int, default=3, help="CV interne de la méthode classique")
    c.add_argument('--wilson-z', type=float, default=1.645, help="méthode classique : marge Wilson sur le recall")
    c.add_argument('--tol', type=float, default=0.002, help="tolérance du plateau quasi-optimal")
    c.add_argument('--robust-alpha', type=float, default=0.10,
                   help="méthode robuste : la performance visée doit être tenue dans (1-alpha) des rééchantillons")
    c.add_argument('--cal-blocks', type=int, default=20, help="méthode robuste : nb de blocs temporels rééchantillonnés")
    c.add_argument('--inner-boot', type=int, default=200, help="méthode robuste : nb de rééchantillons par calibration")
    c.add_argument('--boot', type=int, default=2000, help="bootstrap des IC de performance")
    c.add_argument('--boot-final', type=int, default=200, help="bootstrap des IC des seuils finaux")
    c.add_argument('--n-blocks', type=int, default=40, help="blocs temporels fins pour le bootstrap des IC")
    c.add_argument('--deploy-prior', type=float, default=0.2,
                   help="fraction réelle de paires 'différentes' en déploiement")
    c.add_argument('--deploy-priors', default='0.05,0.1,0.2,0.3,0.5',
                   help="priors testés pour la table de sensibilité au déploiement")
    c.add_argument('--e-tx', type=float, default=None, help="énergie d'une transmission (mJ)")
    c.add_argument('--e-gate', type=float, default=0.0, help="énergie du calcul du garde-fou par paire (mJ)")
    c.add_argument('--article-thresholds', default=None,
                   help="seuils actuels de l'article HIST,MEAN,PROP à évaluer comme référence (ex. 0.05,10,30)")
    c.add_argument('--article-block', type=float, default=0.5,
                   help="seuil de bloc utilisé par PROP dans l'article (doit figurer dans les métriques extraites)")
    c.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cmd_extract(a) if a.cmd == 'extract' else cmd_calibrate(a)


if __name__ == '__main__':
    main()
