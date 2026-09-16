"""
JPEG — implémentation existante : Pillow (bindings libjpeg-turbo).
Source : https://python-pillow.org
Aucune modification de l'algorithme ; seuls les wrappers d'E/S sont écrits par nous.
"""

import io
import numpy as np
from PIL import Image

import config
from .base import BaseCodec


class JPEGCodec(BaseCodec):
    name = "JPEG"
    is_lossy = True
    requires_side_information = False

    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        img = Image.fromarray(image_array, mode="RGB")
        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=config.JPEG_QUALITY,
            subsampling=config.JPEG_SUBSAMPLING,
        )
        return buf.getvalue()

    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        buf = io.BytesIO(encoded)
        img = Image.open(buf)
        img.load()
        return np.array(img.convert("RGB"))
