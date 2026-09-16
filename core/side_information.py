"""
Génération de la side information Y pour Slepian-Wolf et Wyner-Ziv.

Protocole (documenté dans README.md) :
    Y = blur_sigma(X) + Laplace(0, scale)

Ceci simule un canal de corrélation virtuel entre X (source) et Y (side info
disponible uniquement au décodeur), conformément au protocole standard des
papiers fondateurs de Distributed Source Coding (Pradhan & Ramchandran 2003 ;
Aaron & Girod 2002), en l'absence de paires d'images réellement corrélées
dans UCID1338.

La graine aléatoire est dérivée déterministiquement de l'identifiant de
l'image, pour la reproductibilité totale du benchmark.
"""

import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter

import config


def _seed_from_image_id(image_id: str) -> int:
    h = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def generate_side_information(image_array: np.ndarray, image_id: str) -> np.ndarray:
    """
    Génère Y à partir de X par transformation contrôlée + bruit additif.
    N'est JAMAIS incluse dans S_encoded (elle est supposée déjà disponible
    au décodeur, comme dans le modèle Slepian-Wolf/Wyner-Ziv classique).
    """
    rng = np.random.default_rng(_seed_from_image_id(image_id))

    x = image_array.astype(np.float64)
    blurred = gaussian_filter(x, sigma=(config.SIDE_INFO_BLUR_SIGMA,
                                         config.SIDE_INFO_BLUR_SIGMA, 0))
    noise = rng.laplace(loc=0.0, scale=config.SIDE_INFO_LAPLACE_SCALE, size=x.shape)

    y = np.clip(blurred + noise, 0, 255).round().astype(np.uint8)
    return y
