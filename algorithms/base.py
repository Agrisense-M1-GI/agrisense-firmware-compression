"""
Interface commune à tous les codecs du benchmark.
Chaque codec expose encode(image_array, ctx) -> bytes  et decode(bytes, ctx) -> np.ndarray
`ctx` est un dict libre pour passer W,H,C, side info, etc. selon l'algo.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseCodec(ABC):
    name: str = "base"
    is_lossy: bool = False
    requires_side_information: bool = False

    @abstractmethod
    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        ...

    @abstractmethod
    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        ...
