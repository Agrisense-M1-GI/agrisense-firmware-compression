#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_gate.py — calibration et validation du garde-fou énergétique (AgriSense)

Protocole (pensé pour être défendable en relecture) :
  1. Données : CSV (image_1, image_2, label) ; label 0 = identique (paire image/elle-même),
     1 = similaire, 2 = différent. Cible du garde-fou : POSITIF = 2 (transmettre).
  2. Deux scénarios rapportés séparément :
       hard : labels {1,2}   -> scénario principal (le seul qui teste vraiment le garde-fou)
       all  : labels {0,1,2} -> optimiste : les paires 0 sont des auto-paires, métriques == 0
  3. Validation croisée IMBRIQUÉE, par BLOCS TEMPORELS (ordre de img_XXXX) avec PURGE :
     toute paire d'entraînement partageant une image avec le fold de test est retirée
     (les paires forment une composante connexe géante : un split aléatoire fuirait).
       - boucle externe : estimation non biaisée de la performance (jamais vue à la calibration)
       - boucle interne : choix de l'hyperparamètre "seuil de bloc" (prop@t)
  4. Objectifs opérationnels (--objectives) :
       spec@0.95  maximiser la specificity (énergie économisée) sous recall >= 0.95, la
                  contrainte étant testée sur la borne basse de Wilson du recall d'entraînement
       bacc       balanced accuracy
       cost:1:5   coût = c_fp*FP + c_fn*FN
  5. Familles de règles comparées : vote 1/3, 2/3, 3/3 sur (hist, mean, prop), seuil sur une
     seule métrique (3 baselines), régression logistique (plafond de référence).
  6. Incertitudes : bootstrap par blocs temporels (IC 95 %), différences appariées vs vote2,
     stabilité des seuils par fold, IC bootstrap des seuils finaux.
  7. Énergie : projection à un prior de déploiement (--deploy-prior), car le dataset est
     équilibré par construction et ne reflète pas la fréquence réelle des changements.

Usage
  python calibrate_gate.py extract   --dataset dataset-gate-calibration.csv \
        --images-dir /chemin/640X480-PNG --out metriques_gate.csv
  python calibrate_gate.py calibrate --metrics metriques_gate.csv --out results_gate \
        --objectives "spec@0.95,spec@0.98,bacc" --deploy-prior 0.2 --e-tx 250 --e-gate 3

Dépendances : numpy, pandas, pillow (extract), matplotlib (figures, optionnel).
"""
import argparse
import json
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

EPS = 1e-9
FEATS = ['hist', 'mean']          # + une variante prop@t
PROP_RULES_EXCLUDED = ('single_hist', 'single_mean')
ALL_RULES = ['vote1', 'vote2', 'vote3', 'single_hist', 'single_mean', 'single_prop', 'logreg']


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
        """Utilité scalaire poolée (sélection de l'hyperparamètre en boucle interne)."""
        rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if self.kind == 'bacc':
            return (rec + spec) / 2
        if self.kind == 'spec':
            return spec - 5 * max(0.0, self.r - rec)
        return -(self.cfp * fp + self.cfn * fn) / (tp + fp + fn + tn)


def choose(tp, fp, fn, tn, obj, z, tol):
    """Choisit un point de la grille : centre du plateau quasi-optimal (stabilité),
    plutôt que le premier maximum. Si la contrainte est infaisable -> recall maximal."""
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
# 4. Règles de décision
# =============================================================================

def _grid(x, n, qlo=2.0, qhi=98.0):
    return np.unique(np.percentile(x, np.linspace(qlo, qhi, n)))


def table_vote(X, y, G):
    grids = [_grid(X[:, m], G) for m in range(3)]
    G1, G2, G3 = map(len, grids)
    cnt = {}
    for cls in (1, 0):
        Xc = X[y == cls]
        a, b, c = [(Xc[:, m][:, None] > grids[m][None, :]).astype(np.uint8) for m in range(3)]
        bc = b[:, :, None] + c[:, None, :]
        res = {k: np.zeros((G1, G2, G3), np.int64) for k in (1, 2, 3)}
        for i in range(G1):
            s = a[:, i][:, None, None] + bc
            for k in (1, 2, 3):
                res[k][i] = (s >= k).sum(0)
        cnt[cls] = res
    return grids, cnt, int(y.sum()), int((1 - y).sum())


def fit_logreg(X, y, l2=1e-2, iters=60):
    s = np.maximum(np.median(X, axis=0), 1e-9)
    T = np.log1p(X / s)
    mu, sd = T.mean(0), T.std(0) + 1e-12
    Z1 = np.c_[np.ones(len(T)), (T - mu) / sd]
    w = np.zeros(Z1.shape[1])
    R = l2 * np.eye(len(w))
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z1 @ w, -30, 30)))
        g = Z1.T @ (p - y) + R @ w
        H = (Z1 * (p * (1 - p) + 1e-9)[:, None]).T @ Z1 + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(s=s.tolist(), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def lr_score(lr, X):
    T = np.log1p(X / np.asarray(lr['s']))
    Z = (T - np.asarray(lr['mu'])) / np.asarray(lr['sd'])
    w = np.asarray(lr['w'])
    return w[0] + Z @ w[1:]


def fit_rule(rule, X, y, objs, cfg):
    """Ajuste les seuils de `rule` sur (X, y) pour CHAQUE objectif. Renvoie {obj.name: modèle}."""
    z, tol = cfg.wilson_z, cfg.tol
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("une classe est absente du jeu d'entraînement")
    models = {}
    if rule.startswith('vote'):
        k = int(rule[4])
        grids, cnt, npos, nneg = table_vote(X, y, cfg.grid_size)
        tp, fp = cnt[1][k], cnt[0][k]
        fn, tn = npos - tp, nneg - fp
        for o in objs:
            j, feas = choose(tp, fp, fn, tn, o, z, tol)
            ijk = np.unravel_index(j, tp.shape)
            models[o.name] = dict(rule=rule, k=k, th=[float(grids[m][ijk[m]]) for m in range(3)],
                                  feasible=feas)
        return models
    if rule.startswith('single_'):
        m = ['hist', 'mean', 'prop'].index(rule[7:])
        score, aux = X[:, m], dict(m=m)
    else:
        lr = fit_logreg(X, y)
        score, aux = lr_score(lr, X), dict(lr=lr)
    grid = _grid(score, 200, 1, 99)
    c = score[:, None] > grid[None, :]
    tp = (c & (y == 1)[:, None]).sum(0)
    fp = (c & (y == 0)[:, None]).sum(0)
    fn, tn = npos - tp, nneg - fp
    for o in objs:
        j, feas = choose(tp, fp, fn, tn, o, z, tol)
        models[o.name] = dict(rule=rule, t=float(grid[j]), feasible=feas, **aux)
    return models


def predict(model, X):
    r = model['rule']
    if r.startswith('vote'):
        s = sum((X[:, m] > model['th'][m]).astype(int) for m in range(3))
        return (s >= model['k']).astype(int)
    if r.startswith('single_'):
        return (X[:, model['m']] > model['t']).astype(int)
    return (lr_score(model['lr'], X) > model['t']).astype(int)


def thresholds_of(model):
    if model['rule'].startswith('vote'):
        return {'thr_hist': model['th'][0], 'thr_mean': model['th'][1], 'thr_prop': model['th'][2]}
    if model['rule'].startswith('single_'):
        return {'thr_' + model['rule'][7:]: model['t']}
    return {'thr_logit': model['t']}


# =============================================================================
# 5. Découpage par blocs temporels + purge
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


def uses_prop(rule):
    return rule not in PROP_RULES_EXCLUDED


def fit_with_pv(rule, feats, y, idx1, idx2, anchor, objs, cfg, pvs):
    """Choisit la variante prop@t par CV interne (blocs + purge), puis ajuste sur tout l'entraînement."""
    cand = pvs if uses_prop(rule) else pvs[:1]
    fold = blocked_folds(anchor, cfg.inner_folds)
    full, util = {}, {o.name: {} for o in objs}
    for pv in cand:
        X = feats[pv]
        full[pv] = fit_rule(rule, X, y, objs, cfg)
        if len(cand) == 1:
            continue
        cm = {o.name: np.zeros(4) for o in objs}
        for f in range(cfg.inner_folds):
            te = fold == f
            trm = purged_train_mask(te, idx1, idx2)
            if te.sum() < 20 or trm.sum() < 50 or len(np.unique(y[trm])) < 2:
                continue
            ms = fit_rule(rule, X[trm], y[trm], objs, cfg)
            for o in objs:
                cm[o.name] += confusion(y[te], predict(ms[o.name], X[te]))
        for o in objs:
            util[o.name][pv] = o.utility(*cm[o.name]) if cm[o.name].sum() else -np.inf
    out = {}
    for o in objs:
        best = cand[0] if len(cand) == 1 else max(cand, key=lambda p: util[o.name][p])
        m = dict(full[best][o.name])
        m['pv'] = best
        out[o.name] = m
    return out


# =============================================================================
# 6. Validation croisée imbriquée
# =============================================================================

def run_cv(df, rules, objs, pvs, cfg, tag):
    y = df['y'].values
    idx1, idx2 = df['idx1'].values, df['idx2'].values
    anchor = np.minimum(idx1, idx2)
    feats = {pv: df[FEATS + [pv]].values for pv in pvs}
    fold = blocked_folds(anchor, cfg.outer_folds)
    preds = {(r, o.name): np.full(len(df), -1) for r in rules for o in objs}
    frows = []
    for f in range(cfg.outer_folds):
        te = fold == f
        trm = purged_train_mask(te, idx1, idx2)
        print(f"[{tag}] fold {f + 1}/{cfg.outer_folds} : test={te.sum()} train(purgé)={trm.sum()} "
              f"(retirées par purge : {(~te).sum() - trm.sum()})", flush=True)
        ftr = {pv: feats[pv][trm] for pv in pvs}
        for r in rules:
            ms = fit_with_pv(r, ftr, y[trm], idx1[trm], idx2[trm], anchor[trm], objs, cfg, pvs)
            for o in objs:
                m = ms[o.name]
                p = predict(m, feats[m['pv']][te])
                preds[(r, o.name)][te] = p
                tp, fp, fn, tn = confusion(y[te], p)
                frows.append(dict(scenario=tag, fold=f, rule=r, objective=o.name, pv=m['pv'],
                                  feasible=m['feasible'], n_train=int(trm.sum()), n_test=int(te.sum()),
                                  test_recall=tp / max(tp + fn, 1), test_spec=tn / max(tn + fp, 1),
                                  **thresholds_of(m)))
    return preds, pd.DataFrame(frows), anchor


def bootstrap_tables(df, preds, objs, rules, anchor, cfg, tag, ref='vote2'):
    y = df['y'].values
    nb = min(cfg.n_blocks, max(len(df) // 25, 5))
    bid = blocked_folds(anchor, nb)
    rng = np.random.default_rng(cfg.seed)
    draws = rng.integers(0, nb, (cfg.boot, nb))
    rows = []
    for o in objs:
        counts = {}
        for r in rules:
            p = preds[(r, o.name)]
            counts[r] = np.stack([confusion(y[bid == b], p[bid == b]) for b in range(nb)])
        est, boot = {}, {}
        for r in rules:
            est[r] = cm_metrics(counts[r].sum(0), cfg.deploy_prior, cfg.e_tx, cfg.e_gate)
            boot[r] = cm_metrics(counts[r][draws].sum(1), cfg.deploy_prior, cfg.e_tx, cfg.e_gate)
        for r in rules:
            for m in est[r]:
                lo, hi = np.percentile(boot[r][m], [2.5, 97.5])
                row = dict(scenario=tag, objective=o.name, rule=r, metric=m, est=float(est[r][m]),
                           lo=lo, hi=hi, n=len(y))
                if ref in rules and r != ref:
                    dd = boot[r][m] - boot[ref][m]
                    row.update(diff_vs_ref=float(est[r][m] - est[ref][m]),
                               diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5))
                rows.append(row)
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
# 7. Ajustement final + IC bootstrap des seuils
# =============================================================================

def final_fit(df, rules, objs, pvs, cfg, tag, ci_rules):
    y = df['y'].values
    idx1, idx2 = df['idx1'].values, df['idx2'].values
    anchor = np.minimum(idx1, idx2)
    feats = {pv: df[FEATS + [pv]].values for pv in pvs}
    out = {}
    nb = min(cfg.n_blocks, max(len(df) // 25, 5))
    bid = blocked_folds(anchor, nb)
    blocks = [np.where(bid == b)[0] for b in range(nb)]
    rng = np.random.default_rng(cfg.seed + 2)
    for r in rules:
        ms = fit_with_pv(r, feats, y, idx1, idx2, anchor, objs, cfg, pvs)
        for o in objs:
            m = ms[o.name]
            X = feats[m['pv']]
            tp, fp, fn, tn = confusion(y, predict(m, X))
            rec = dict(model=m, resub=dict(recall=tp / (tp + fn), specificity=tn / (tn + fp)))
            out.setdefault(o.name, {})[r] = rec
        if r in ci_rules:
            for pv in sorted({ms[o.name]['pv'] for o in objs}):
                sub = [o for o in objs if ms[o.name]['pv'] == pv]
                acc = {o.name: [] for o in sub}
                for b in range(cfg.boot_final):
                    ids = np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)])
                    try:
                        mb = fit_rule(r, feats[pv][ids], y[ids], sub, cfg)
                    except ValueError:
                        continue
                    for o in sub:
                        acc[o.name].append(thresholds_of(mb[o.name]))
                for o in sub:
                    d = pd.DataFrame(acc[o.name])
                    out[o.name][r]['ci95'] = {c: [float(np.percentile(d[c], 2.5)),
                                                  float(np.percentile(d[c], 97.5))] for c in d}
    return out


# =============================================================================
# 8. Sorties : résumé Markdown + figures
# =============================================================================

KEY = ['recall', 'specificity', 'balanced_acc', 'mcc']


def _fmt(row):
    return f"{row.est:.3f} [{row.lo:.3f}, {row.hi:.3f}]"


def summary_md(res, folds, auc, scen_info, cfg):
    L = ["# Résumé de calibration", ""]
    for tag, info in scen_info.items():
        L += [f"## Scénario `{tag}` — {info}", ""]
        a = auc[auc.scenario == tag]
        L += ["**AUC des métriques brutes (IC95 % bootstrap par blocs)**", "", "| métrique | AUC |", "|---|---|"]
        L += [f"| {r.feature} | {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}] |" for r in a.itertuples()] + [""]
        for obj in res[res.scenario == tag].objective.unique():
            sub = res[(res.scenario == tag) & (res.objective == obj)]
            metrics = KEY + [m for m in ('energy_saving_deploy', 'tx_rate_deploy') if m in set(sub.metric)]
            L += [f"### Objectif `{obj}` (performance hors-échantillon, CV imbriquée)", "",
                  "| règle | " + " | ".join(metrics) + " | folds recall≥cible |",
                  "|---|" + "---|" * (len(metrics) + 1)]
            for rule in sub.rule.unique():
                cells = []
                for m in metrics:
                    rr = sub[(sub.rule == rule) & (sub.metric == m)].iloc[0]
                    cells.append(_fmt(rr))
                ff = folds[(folds.scenario == tag) & (folds.objective == obj) & (folds.rule == rule)]
                ok = ""
                if obj.startswith('spec@'):
                    ok = f"{int((ff.test_recall >= float(obj[5:])).sum())}/{len(ff)}"
                L.append(f"| {rule} | " + " | ".join(cells) + f" | {ok} |")
            L += [""]
            d = sub[(sub.metric.isin(['specificity', 'recall'])) & sub.diff_vs_ref.notna()]
            if len(d):
                L += ["Différences appariées vs `vote2` (IC95 % ; ✱ = IC exclut 0) :", ""]
                for r in d.itertuples():
                    star = "✱" if (r.diff_lo > 0 or r.diff_hi < 0) else ""
                    L.append(f"- {r.rule} / {r.metric} : {r.diff_vs_ref:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}] {star}")
                L += [""]
    L += ["## Paramètres", "", "```", json.dumps({k: v for k, v in vars(cfg).items() if k != 'func'}, indent=1,
                                                default=str), "```"]
    return "\n".join(L)


def make_figures(out, data, res, auc_df, pvs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : figures ignorées")
        return
    # Distributions par label (jeu complet)
    df = data.get('all', next(iter(data.values())))
    ref_tag = 'hard' if 'hard' in data else next(iter(data))
    best_pv = auc_df[(auc_df.scenario == ref_tag) & auc_df.feature.isin(pvs)].sort_values('auc').iloc[-1].feature
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a_, col in zip(ax, ['hist', 'mean', best_pv]):
        a_.boxplot([df[df.label == l][col] for l in sorted(df.label.unique())],
           tick_labels=[f"label {l}" for l in sorted(df.label.unique())], showfliers=False)
        a_.set_title(col)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_distributions.png'), dpi=200)
    plt.close(fig)
    # ROC brutes + points de fonctionnement CV
    for tag, d in data.items():
        y = d['y'].values
        fig, a_ = plt.subplots(figsize=(5.2, 5))
        for col in ['hist', 'mean', best_pv]:
            fpr, tpr = roc_curve(d[col].values, y)
            a_.plot(fpr, tpr, lw=1.2, label=f"{col} (AUC {auc_score(d[col].values, y):.3f})")
        a_.plot([0, 1], [0, 1], 'k:', lw=.6)
        sub = res[res.scenario == tag]
        mk = dict(zip(sub.rule.unique(), 'osD^v<>'))
        for (rule, obj), g in sub.groupby(['rule', 'objective']):
            r = g[g.metric == 'recall'].iloc[0]
            s = g[g.metric == 'specificity'].iloc[0]
            a_.errorbar(1 - s.est, r.est, xerr=[[s.est - s.lo], [s.hi - s.est]],
                        yerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk[rule], ms=4, lw=.6, alpha=.8,
                        label=f"{rule} · {obj}")
        a_.set_xlabel("1 − specificity (fausses transmissions)")
        a_.set_ylabel("recall (changements détectés)")
        a_.set_title(f"Scénario {tag}")
        a_.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'fig_roc_{tag}.png'), dpi=200)
        plt.close(fig)


# =============================================================================
# 9. Commande calibrate
# =============================================================================

def cmd_calibrate(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.metrics)
    pvs = [c for c in df.columns if c.startswith('prop@')]
    need = {'image_1', 'image_2', 'label', 'idx1', 'idx2', 'hist', 'mean'}
    if not need <= set(df.columns) or not pvs:
        sys.exit("CSV de métriques invalide : relance `extract` (colonnes prop@t requises).")
    pvs = sorted(pvs, key=lambda c: float(c[5:]))
    objs = [Obj(s.strip()) for s in a.objectives.split(',')]
    rules = [r.strip() for r in a.rules.split(',')]
    ci_rules = [r.strip() for r in a.final_rules.split(',') if r.strip()]
    pos = [int(x) for x in a.positive_labels.split(',')]

    # --- contrôles de cohérence du dataset ---
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

    data, res_all, fold_all, auc_all, finals, info = {}, [], [], [], {}, {}
    for tag in wanted:
        labs, desc = scen[tag]
        d = df[df.label.isin(labs)].reset_index(drop=True)
        if d.y.nunique() < 2:
            print(f"[{tag}] une seule classe -> ignoré")
            continue
        data[tag], info[tag] = d, f"{desc} ; n={len(d)}, positifs={int(d.y.sum())}"
        print(f"\n== Scénario {tag} : {info[tag]} ==")
        preds, folds, anchor = run_cv(d, rules, objs, pvs, a, tag)
        pd.DataFrame({'image_1': d.image_1, 'image_2': d.image_2, 'label': d.label, 'y': d.y,
                      **{f"pred|{r}|{o}": p for (r, o), p in preds.items()}}
                     ).to_csv(os.path.join(a.out, f'cv_predictions_{tag}.csv'), index=False)
        res_all.append(bootstrap_tables(d, preds, objs, rules, anchor, a, tag))
        fold_all.append(folds)
        auc_all.append(auc_table(d, pvs, anchor, a, tag))
        print(f"[{tag}] ajustement final + bootstrap des seuils ({a.boot_final} rééchantillons)...", flush=True)
        finals[tag] = final_fit(d, rules, objs, pvs, a, tag, ci_rules)

    res, folds, auc = pd.concat(res_all), pd.concat(fold_all), pd.concat(auc_all)
    res.to_csv(os.path.join(a.out, 'results_cv.csv'), index=False)
    folds.to_csv(os.path.join(a.out, 'fold_thresholds.csv'), index=False)
    auc.to_csv(os.path.join(a.out, 'auc_features.csv'), index=False)
    with open(os.path.join(a.out, 'final_models.json'), 'w') as f:
        json.dump(finals, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    md = summary_md(res, folds, auc, info, a)
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf8') as f:
        f.write(md)
    make_figures(a.out, data, res, auc, pvs)
    print("\n" + md)
    print("\n== Seuils prêts à coller (règle vote, ajustement final) ==")
    for tag in finals:
        for obj, rr in finals[tag].items():
            for r, rec in rr.items():
                if r.startswith('vote'):
                    m = rec['model']
                    print(f"[{tag}] {obj} {r} pv={m['pv']} : HIST>{m['th'][0]:.4f}  MEAN>{m['th'][1]:.3f}  "
                          f"PROP>{m['th'][2]:.2f}  (seuil de bloc {m['pv'][5:]})  "
                          f"IC95={rec.get('ci95', '-')}")
    print(f"\nFichiers écrits dans {a.out}/")


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
                   help="seuils de distance de bloc testés pour prop (hyperparamètre calibré)")
    c = sub.add_parser('calibrate')
    c.add_argument('--metrics', required=True)
    c.add_argument('--out', default='results_gate')
    c.add_argument('--positive-labels', default='2')
    c.add_argument('--scenarios', default='hard,all')
    c.add_argument('--objectives', default='spec@0.95,spec@0.98,bacc')
    c.add_argument('--rules', default=','.join(ALL_RULES))
    c.add_argument('--final-rules', default='vote2', help="règles dont on calcule l'IC bootstrap des seuils")
    c.add_argument('--grid-size', type=int, default=25)
    c.add_argument('--outer-folds', type=int, default=5)
    c.add_argument('--inner-folds', type=int, default=3)
    c.add_argument('--wilson-z', type=float, default=1.645, help="0 = pas de marge Wilson sur le recall")
    c.add_argument('--tol', type=float, default=0.002, help="tolérance du plateau quasi-optimal")
    c.add_argument('--boot', type=int, default=2000)
    c.add_argument('--boot-final', type=int, default=200)
    c.add_argument('--n-blocks', type=int, default=40, help="blocs temporels fins pour le bootstrap")
    c.add_argument('--deploy-prior', type=float, default=0.2,
                   help="fraction réelle de paires 'différentes' en déploiement")
    c.add_argument('--e-tx', type=float, default=None, help="énergie d'une transmission (mJ)")
    c.add_argument('--e-gate', type=float, default=0.0, help="énergie du calcul du garde-fou par paire (mJ)")
    c.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cmd_extract(a) if a.cmd == 'extract' else cmd_calibrate(a)


if __name__ == '__main__':
    main()
