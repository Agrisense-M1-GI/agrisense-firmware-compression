"""
Slepian-Wolf — codage de source distribué SANS perte, avec side information
disponible uniquement au décodeur (Slepian & Wolf, 1973).

Brique existante : pyldpc (matrice LDPC + syndrome). Voir ldpc_helper.py.
Side information : générée par transformation contrôlée de l'image
(core/side_information.py), jamais incluse dans S_encoded.

Protocole :
  1. X (pixels bruts, 8 bits/canal) est converti en CODE DE GRAY avant
     décomposition en plans de bits (deux valeurs de pixel proches ne
     diffèrent alors que d'1 bit, au lieu de plusieurs à cause des
     retenues binaires -> réduit le taux d'erreur par bloc, pratique
     standard en codage source corrélé).
  2. Seuls les SW_MSB_PLANES plans de bits de poids fort passent par le
     codage LDPC/syndrome (cf. justification théorique dans config.py) ;
     les plans restants sont transmis tels quels (comptés intégralement
     dans S_encoded).
  3. Pour chaque bloc compressé, seul le SYNDROME (n-k bits) est transmis.
  4. Le décodeur reconstruit chaque bloc compressé à partir du syndrome +
     du bloc correspondant de la side info Y (elle-même Gray-codée) ;
     reconstruction exacte vérifiée par check_lossless_exact.
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


def _to_bitplanes(arr: np.ndarray) -> np.ndarray:
    """(H,W,C) uint8 -> (8, H*W*C) bits, plan de poids fort en premier."""
    flat = arr.reshape(-1).astype(np.uint8)
    bits = np.unpackbits(flat)
    return bits.reshape(-1, 8).T  # (8, H*W*C)


def _from_bitplanes(bitplanes: np.ndarray, shape) -> np.ndarray:
    bits = bitplanes.T.reshape(-1)
    flat = np.packbits(bits)
    return flat.reshape(shape)


class SlepianWolfCodec(BaseCodec):
    name = "Slepian-Wolf"
    is_lossy = False
    requires_side_information = True

    def __init__(self):
        self.ldpc = LDPCSyndromeCoder()
        self.compressed_planes = sorted(config.SW_COMPRESSED_PLANE_INDICES)
        self.raw_planes = [i for i in range(8) if i not in self.compressed_planes]

    def _to_blocks(self, plane: np.ndarray):
        n = self.ldpc.block_size()
        pad = (-len(plane)) % n
        padded = np.concatenate([plane, np.zeros(pad, dtype=np.uint8)])
        return padded.reshape(-1, n), pad

    def encode(self, image_array: np.ndarray, ctx: dict) -> bytes:
        gray = _to_gray(image_array)
        bitplanes = _to_bitplanes(gray)  # (8, N)
        n = self.ldpc.block_size()

        out = bytearray()
        out += struct.pack(">IB", n, len(self.compressed_planes))
        for idx in self.compressed_planes:
            out += struct.pack(">B", idx)

        # Plans sélectionnés par calibration : codage LDPC/syndrome
        for idx in self.compressed_planes:
            blocks, pad = self._to_blocks(bitplanes[idx])
            syndromes = self.ldpc.compute_syndrome_batch(blocks)
            packed = np.packbits(syndromes.reshape(-1))
            out += struct.pack(">BII", pad, syndromes.shape[0], syndromes.shape[1])
            out += packed.tobytes()

        # Plans restants : transmis tels quels (empaquetés en octets),
        # dans l'ordre croissant de leur indice (même ordre au décodage).
        remaining_bits = bitplanes[self.raw_planes].reshape(-1)
        packed_raw = np.packbits(remaining_bits)
        out += struct.pack(">I", len(remaining_bits))
        out += packed_raw.tobytes()

        ctx["_side_info"] = generate_side_information(image_array, ctx["image_id"])
        return bytes(out)

    def decode(self, encoded: bytes, ctx: dict) -> np.ndarray:
        offset = 0
        n, n_compressed = struct.unpack_from(">IB", encoded, offset)
        offset += 5
        compressed_planes = list(struct.unpack_from(f">{n_compressed}B", encoded, offset))
        offset += n_compressed
        raw_planes = [i for i in range(8) if i not in compressed_planes]

        side_info = ctx.get("_side_info")
        if side_info is None:
            side_info = generate_side_information(
                np.zeros((ctx["H"], ctx["W"], ctx["C"]), dtype=np.uint8), ctx["image_id"]
            )
        side_gray = _to_gray(side_info)
        side_bitplanes = _to_bitplanes(side_gray)

        n_pixels = ctx["H"] * ctx["W"] * ctx["C"]
        all_planes = np.zeros((8, n_pixels), dtype=np.uint8)

        for plane_idx in compressed_planes:
            pad, n_blocks, syn_len = struct.unpack_from(">BII", encoded, offset)
            offset += 9
            n_syn_bytes = (n_blocks * syn_len + 7) // 8
            packed = np.frombuffer(encoded, dtype=np.uint8, count=n_syn_bytes, offset=offset)
            offset += n_syn_bytes
            syndromes = np.unpackbits(packed)[: n_blocks * syn_len].reshape(n_blocks, syn_len)

            side_plane = side_bitplanes[plane_idx]
            side_blocks, _ = self._to_blocks(side_plane)
            side_blocks = side_blocks[:n_blocks]

            rec_blocks = self.ldpc.decode_with_side_info_batch(syndromes, side_blocks)
            rec_plane = rec_blocks.reshape(-1)
            if pad:
                rec_plane = rec_plane[:-pad]
            all_planes[plane_idx] = rec_plane[:n_pixels]

        (n_raw_bits,) = struct.unpack_from(">I", encoded, offset)
        offset += 4
        n_raw_bytes = (n_raw_bits + 7) // 8
        raw_packed = np.frombuffer(encoded, dtype=np.uint8, count=n_raw_bytes, offset=offset)
        raw_bits = np.unpackbits(raw_packed)[:n_raw_bits]
        raw_planes_data = raw_bits.reshape(len(raw_planes), n_pixels)
        for j, plane_idx in enumerate(raw_planes):
            all_planes[plane_idx] = raw_planes_data[j]

        shape = (ctx["H"], ctx["W"], ctx["C"])
        gray_rec = _from_bitplanes(all_planes, shape)
        return _from_gray(gray_rec)
