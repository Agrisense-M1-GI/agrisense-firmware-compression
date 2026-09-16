"""
Wyner-Ziv — codage de source distribué AVEC PERTE, side information
disponible uniquement au décodeur (Wyner & Ziv, 1976).

Différence avec Slepian-Wolf : une quantification scalaire (lossy) est
appliquée AVANT le binning par syndrome LDPC (cf. théorie classique WZ :
quantification + codage Slepian-Wolf sur les indices de quantification).

Brique existante : pyldpc (syndrome), réutilisée directement (voir
ldpc_helper.py). Même stratégie que Slepian-Wolf : code de Gray sur les
indices quantifiés, et seuls les plans listés dans
config.WZ_COMPRESSED_PLANE_INDICES (déterminés par calibration empirique,
cf. core/calibrate_planes.py — PAS une hypothèse de position) passent par
le LDPC ; le reste est transmis tel quel.
Quantificateur : uniforme scalaire, maison (théorie standard).
Reconstruction au décodeur : centre de cellule du niveau quantifié reçu.
"""

import struct
import numpy as np

from .base import BaseCodec
from .ldpc_helper import LDPCSyndromeCoder
from core.side_information import generate_side_information
import config


def _to_gray(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.uint16)
    return (v ^ (v >> 1)).astype(np.uint8)


def _from_gray(gray: np.ndarray) -> np.ndarray:
    v = gray.astype(np.uint16)
    mask = v >> 1
    while mask.any():
        v = v ^ mask
        mask = mask >> 1
    return v.astype(np.uint8)


def _uniform_quantize(arr: np.ndarray, levels: int):
    step = 256.0 / levels
    idx = np.clip((arr.astype(np.float64) // step).astype(np.int64), 0, levels - 1)
    return idx.astype(np.uint8), step


def _dequantize_bin_center(idx: np.ndarray, step: float):
    return (idx.astype(np.float64) + 0.5) * step


class WynerZivCodec(BaseCodec):
    name = "Wyner-Ziv"
    is_lossy = True
    requires_side_information = True

    def __init__(self):
        self.ldpc = LDPCSyndromeCoder()
        self.levels = config.WZ_QUANT_LEVELS
        self.bits_per_symbol = int(np.ceil(np.log2(self.levels)))
        # Ne garder que les indices de plan valides pour bits_per_symbol.
        self.compressed_planes = sorted(
            i for i in config.WZ_COMPRESSED_PLANE_INDICES if i < self.bits_per_symbol
        )
        self.raw_planes = [i for i in range(self.bits_per_symbol)
                            if i not in self.compressed_planes]

    def _to_blocks(self, plane: np.ndarray):
        n = self.ldpc.block_size()
        pad = (-len(plane)) % n
        padded = np.concatenate([plane, np.zeros(pad, dtype=np.uint8)])
        return padded.reshape(-1, n), pad

    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        idx, step = _uniform_quantize(image_array, self.levels)
        gray_idx = _to_gray(idx)

        flat = gray_idx.reshape(-1).astype(np.uint8)
        bits = np.unpackbits(flat).reshape(-1, 8).T
        used_planes = bits[8 - self.bits_per_symbol:]  # (bits_per_symbol, N), indice 0 = MSB

        n = self.ldpc.block_size()
        out = bytearray()
        out += struct.pack(">IBB", n, self.bits_per_symbol, len(self.compressed_planes))
        for idx_p in self.compressed_planes:
            out += struct.pack(">B", idx_p)
        out += struct.pack(">f", step)

        for idx_p in self.compressed_planes:
            blocks, pad = self._to_blocks(used_planes[idx_p])
            syndromes = self.ldpc.compute_syndrome_batch(blocks)
            packed = np.packbits(syndromes.reshape(-1))
            out += struct.pack(">BII", pad, syndromes.shape[0], syndromes.shape[1])
            out += packed.tobytes()

        remaining_bits = used_planes[self.raw_planes].reshape(-1)
        packed_raw = np.packbits(remaining_bits)
        out += struct.pack(">I", len(remaining_bits))
        out += packed_raw.tobytes()

        ctx["_side_info"] = generate_side_information(image_array, ctx["image_id"])
        return bytes(out)

    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        offset = 0
        n, bits_per_symbol, n_compressed = struct.unpack_from(">IBB", encoded, offset)
        offset += 6
        compressed_planes = list(struct.unpack_from(f">{n_compressed}B", encoded, offset))
        offset += n_compressed
        (step,) = struct.unpack_from(">f", encoded, offset)
        offset += 4
        raw_planes = [i for i in range(bits_per_symbol) if i not in compressed_planes]

        side_info = ctx["_side_info"]
        side_idx, _ = _uniform_quantize(side_info, self.levels)
        side_gray = _to_gray(side_idx)
        side_flat = side_gray.reshape(-1).astype(np.uint8)
        side_bits = np.unpackbits(side_flat).reshape(-1, 8).T
        side_used_planes = side_bits[8 - bits_per_symbol:]

        n_pixels = ctx["H"] * ctx["W"] * ctx["C"]
        used_planes_rec = np.zeros((bits_per_symbol, n_pixels), dtype=np.uint8)

        for idx_p in compressed_planes:
            pad, n_blocks, syn_len = struct.unpack_from(">BII", encoded, offset)
            offset += 9
            n_syn_bytes = (n_blocks * syn_len + 7) // 8
            packed = np.frombuffer(encoded, dtype=np.uint8, count=n_syn_bytes, offset=offset)
            offset += n_syn_bytes
            syndromes = np.unpackbits(packed)[: n_blocks * syn_len].reshape(n_blocks, syn_len)

            side_plane = side_used_planes[idx_p]
            side_blocks, _ = self._to_blocks(side_plane)
            side_blocks = side_blocks[:n_blocks]

            rec_blocks = self.ldpc.decode_with_side_info_batch(syndromes, side_blocks)
            rec_plane = rec_blocks.reshape(-1)
            if pad:
                rec_plane = rec_plane[:-pad]
            used_planes_rec[idx_p] = rec_plane[:n_pixels]

        (n_raw_bits,) = struct.unpack_from(">I", encoded, offset)
        offset += 4
        n_raw_bytes = (n_raw_bits + 7) // 8
        raw_packed = np.frombuffer(encoded, dtype=np.uint8, count=n_raw_bytes, offset=offset)
        raw_bits = np.unpackbits(raw_packed)[:n_raw_bits]
        raw_data = raw_bits.reshape(len(raw_planes), n_pixels) if raw_planes else \
            np.zeros((0, n_pixels), dtype=np.uint8)
        for j, idx_p in enumerate(raw_planes):
            used_planes_rec[idx_p] = raw_data[j]

        full_planes = np.zeros((8, n_pixels), dtype=np.uint8)
        full_planes[8 - bits_per_symbol:] = used_planes_rec

        H, W, C = ctx["H"], ctx["W"], ctx["C"]
        gray_idx_flat = np.packbits(full_planes.T.reshape(-1))
        gray_idx = gray_idx_flat.reshape(H, W, C)
        idx = _from_gray(gray_idx)

        recon = _dequantize_bin_center(idx, step)
        return np.clip(recon, 0, 255).round().astype(np.uint8)
