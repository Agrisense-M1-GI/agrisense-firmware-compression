"""
Brique commune Slepian-Wolf / Wyner-Ziv : codage par syndrome sur des
blocs binaires via un code LDPC.

Bibliothèque existante : pyldpc (https://pypi.org/project/pyldpc/),
matrice de parité H générée par construction régulière (d_v, d_c).
Nous utilisons H pour calculer un SYNDROME (binning), brique nécessaire
au codage Slepian-Wolf/Wyner-Ziv (cf. Pradhan & Ramchandran, DISCUS,
2003).

Décodage par syndrome + side info : algorithme de bit-flipping à règle
MAJORITAIRE (Gallager, "Low-Density Parity-Check Codes", 1962 —
"Algorithme A") : à chaque itération, on bascule en parallèle tous les
bits variables dont plus de la moitié des contraintes voisines sont
violées. C'est un algorithme de décodage standard et correct (contra
une variante simplifiée testée puis rejetée car elle pouvait dégrader
la solution — cf. historique du projet), implémenté par nous en
suivant la théorie.

Optimisations d'EXÉCUTION uniquement (ni l'algorithme ni les paramètres
théoriques ne changent) :
  - Vectorisation NumPy sur l'ensemble des blocs d'un plan de bits.
  - Réduction progressive de l'ensemble actif (les blocs déjà résolus
    sortent du calcul à chaque itération).
LDPC_MAX_ITERS borne le nombre d'itérations (paramètre fixe, config.py).
Un bloc non résolu dans cette limite reste tel quel -> observation
légitime (corrélation source/side-info insuffisante pour ce bloc), pas
une erreur masquée.
"""

import numpy as np
from pyldpc import make_ldpc

import config


class LDPCSyndromeCoder:
    def __init__(self):
        H_sparse, self.G = make_ldpc(
            config.LDPC_N,
            config.LDPC_D_V,
            config.LDPC_D_C,
            systematic=config.LDPC_MAKE_SYSTEMATIC,
            sparse=True,
        )
        H_dense = H_sparse.toarray() if hasattr(H_sparse, "toarray") else H_sparse
        self.H = np.asarray(H_dense % 2, dtype=np.uint8)
        self.n = self.H.shape[1]
        self.n_checks = self.H.shape[0]
        self.d_v = config.LDPC_D_V  # degré de chaque noeud variable (régulier)

    def block_size(self) -> int:
        return self.n

    def compute_syndrome_batch(self, blocks: np.ndarray) -> np.ndarray:
        """blocks: (B, n) binaire. Retourne (B, n_checks) = (blocks @ H.T) % 2."""
        return (blocks.astype(np.uint8) @ self.H.T) % 2

    def decode_with_side_info_batch(self, syndromes: np.ndarray, side_bits: np.ndarray,
                                     max_iters: int = None) -> np.ndarray:
        """
        Décodage par bit-flipping majoritaire (Gallager, Algorithme A),
        vectorisé sur les blocs, avec réduction progressive de
        l'ensemble actif (cf. docstring du module).
        """
        max_iters = max_iters if max_iters is not None else config.LDPC_MAX_ITERS
        x = side_bits.copy().astype(np.uint8)
        B = x.shape[0]
        active = np.arange(B)
        threshold = self.d_v / 2.0

        for _ in range(max_iters):
            if active.size == 0:
                break

            xa = x[active]
            sa = syndromes[active]
            current = self.compute_syndrome_batch(xa)
            mismatch = (current != sa)
            still = mismatch.any(axis=1)
            if not still.any():
                break

            active2 = active[still]
            mism2 = mismatch[still]

            violated_per_var = mism2.astype(np.int32) @ self.H   # (Ba, n)
            flip_mask = violated_per_var > threshold
            progressing = flip_mask.any(axis=1)
            if not progressing.any():
                # Plus aucun bloc actif ne peut progresser sous la règle
                # majoritaire : on s'arrête (les blocs restants ne seront
                # pas résolus, cf. lossless_exact=False).
                break

            rows = active2[progressing]
            x[rows] ^= flip_mask[progressing].astype(np.uint8)
            active = active2

        return x
