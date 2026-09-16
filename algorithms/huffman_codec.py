"""
Huffman — implémentation maison, fidèle à l'algorithme canonique de
Huffman (D.A. Huffman, "A Method for the Construction of Minimum-Redundancy
Codes", Proc. IRE, 1952).

Justification : il n'existe pas de bibliothèque Python mature et générique
dédiée au codage de Huffman "nu" (hors formats conteneurs comme JPEG/DEFLATE
qui l'utilisent en interne mais ne l'exposent pas isolément). Implémentation
complète ci-dessous, sans optimisation ad hoc.

Codage appliqué directement sur les octets bruts RGB (pas de DPCM ni de
transformation préalable), pour rester un point de comparaison "pur"
entropique face à JPEG (avec DCT) et LZW (dictionnaire).
"""

import heapq
import struct
from collections import Counter
import numpy as np

from .base import BaseCodec


class _Node:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_tree(freqs: Counter) -> _Node:
    heap = [_Node(f, s) for s, f in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # cas dégénéré : un seul symbole distinct
        only = heap[0]
        return _Node(only.freq, left=only)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(a.freq + b.freq, left=a, right=b))
    return heap[0]


def _build_codes(node: _Node, prefix="", codes=None) -> dict:
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
        return codes
    if node.left:
        _build_codes(node.left, prefix + "0", codes)
    if node.right:
        _build_codes(node.right, prefix + "1", codes)
    return codes


class HuffmanCodec(BaseCodec):
    name = "Huffman"
    is_lossy = False
    requires_side_information = False

    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        raw = image_array.tobytes()
        freqs = Counter(raw)
        tree = _build_tree(freqs)
        codes = _build_codes(tree)

        bitstring = "".join(codes[b] for b in raw)
        padding = (8 - len(bitstring) % 8) % 8
        bitstring += "0" * padding

        body = int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder="big") if bitstring else b""

        # En-tête : table de fréquences (nécessaire pour reconstruire l'arbre au décodage,
        # comptée dans S_encoded car indispensable à la reconstruction) + padding + longueur brute.
        header_entries = list(freqs.items())
        header = struct.pack(">I", len(header_entries))
        for symbol, freq in header_entries:
            header += struct.pack(">BQ", symbol, freq)
        header += struct.pack(">BQ", padding, len(raw))

        return header + body

    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        offset = 0
        (n_entries,) = struct.unpack_from(">I", encoded, offset)
        offset += 4
        freqs = Counter()
        for _ in range(n_entries):
            symbol, freq = struct.unpack_from(">BQ", encoded, offset)
            offset += 9
            freqs[symbol] = freq
        padding, raw_len = struct.unpack_from(">BQ", encoded, offset)
        offset += 9

        tree = _build_tree(freqs)
        body = encoded[offset:]
        bitstring = bin(int.from_bytes(body, byteorder="big"))[2:].zfill(len(body) * 8)
        if padding:
            bitstring = bitstring[:-padding]

        out = bytearray()
        node = tree
        for bit in bitstring:
            node = node.left if bit == "0" else node.right
            if node.symbol is not None:
                out.append(node.symbol)
                node = tree
                if len(out) == raw_len:
                    break

        W, H, C = ctx["W"], ctx["H"], ctx["C"]
        return np.frombuffer(bytes(out), dtype=np.uint8).reshape(H, W, C)
