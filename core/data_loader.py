"""
Chargement des images TIFF et calcul de la référence S_raw.
Les images originales ne sont jamais modifiées (lecture seule).
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import tifffile


@dataclass
class LoadedImage:
    image_id: str
    path: Path
    array: np.ndarray   # (H, W, 3) uint8 RGB
    W: int
    H: int
    C: int
    s_raw: int           # W * H * C en octets (référence unique, PAS la taille du fichier TIFF)


def list_dataset_images(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("*.tif")) + sorted(dataset_dir.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier .tif/.tiff trouvé dans {dataset_dir}. "
            "Vérifie le chemin défini dans config.DATASET_DIR."
        )
    return files


def load_image(path: Path) -> LoadedImage:
    """Charge un TIFF en RGB 8 bits, sans jamais écrire ni modifier le fichier source."""
    arr = tifffile.imread(str(path))

    # Normalisation défensive du format en (H, W, 3) uint8
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # Ramène à 8 bits si le TIFF est stocké sur une autre profondeur,
        # documenté explicitement (pas de perte silencieuse).
        arr = (arr.astype(np.float64) / arr.max() * 255.0).round().astype(np.uint8)

    H, W, C = arr.shape
    s_raw = W * H * C  # référence unique, cf. spécification (jamais la taille fichier)

    return LoadedImage(
        image_id=path.stem,
        path=path,
        array=arr,
        W=W,
        H=H,
        C=C,
        s_raw=s_raw,
    )
