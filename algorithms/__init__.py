from .jpeg_codec import JPEGCodec
from .huffman_codec import HuffmanCodec
from .lzw_codec import LZWCodec
from .slepian_wolf_codec import SlepianWolfCodec
from .wyner_ziv_codec import WynerZivCodec

ALL_CODECS = {
    "JPEG": JPEGCodec,
    "Huffman": HuffmanCodec,
    "LZW": LZWCodec,
    "Slepian-Wolf": SlepianWolfCodec,
    "Wyner-Ziv": WynerZivCodec,
}
