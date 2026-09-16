"""
LZW — implémentation existante : package `ncompress` (PyPI), portage
Python/C++ minimal de l'outil CLI historique (N)compress
(https://github.com/vapier/ncompress), lui-même l'implémentation de
référence de l'algorithme de Welch (1984) utilisée par Unix `compress`.
Source : https://pypi.org/project/ncompress/
Wheels précompilées disponibles pour Python 3.8-3.12 (Linux/macOS/Windows),
pas de compilation nécessaire.

API : ncompress.compress(bytes) -> bytes ; ncompress.decompress(bytes) -> bytes
Aucune modification de l'algorithme ; seul le wrapper d'E/S (reshape vers/
depuis un tableau numpy RGB) est écrit par nous.
"""

import numpy as np
import ncompress

from .base import BaseCodec


class LZWCodec(BaseCodec):
    name = "LZW"
    is_lossy = False
    requires_side_information = False

    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        raw = image_array.tobytes()
        return ncompress.compress(raw)

    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        raw = ncompress.decompress(encoded)
        W, H, C = ctx["W"], ctx["H"], ctx["C"]
        return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, C)
