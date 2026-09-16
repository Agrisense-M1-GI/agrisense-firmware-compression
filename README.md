**Benchmark de compression d'images — JPEG / Huffman / LZW / Slepian-Wolf / Wyner-Ziv**  
Dataset : UCID1338 (1338 TIFF RGB non compressés), attendu dans ~/Downloads/archive/UCID1338. ( à modifier si necessaire dans le fichier config.py)  
**Sources des implémentations (traçabilité)**  
| | | | | |  
|-|-|-|-|-|  
| **Algo** | **Implémentation** | **Source** | **Version** | **Ce qui est écrit par nous** |   
| JPEG | Pillow (bindings vers libjpeg-turbo) | [https://python-pillow.org](https://python-pillow.org "https://python-pillow.org") | Pillow >= 10.0 | wrapper d'encodage/décodage + mesures |   
| Huffman | Maison, algorithme canonique de Huffman (1952) sur octets bruts (DPCM optionnel désactivé par défaut) | — | — | implémentation complète, car aucune lib mature de Huffman "nu" pour images n'existe |   
| LZW | ncompress (portage Python/C++ de l'outil CLI historique (N)compress, implémentation de Welch 1984) | [https://pypi.org/project/ncompress/](https://pypi.org/project/ncompress/ "https://pypi.org/project/ncompress/") | 1.0.2 | wrapper only (reshape numpy) |   
| Slepian-Wolf | pyldpc (codes LDPC, syndrome-based) | [https://pypi.org/project/pyldpc/](https://pypi.org/project/pyldpc/ "https://pypi.org/project/pyldpc/") | dernière version PyPI | protocole de binning + génération side info (voir ci-dessous), wrapper |   
| Wyner-Ziv | pyldpc pour le binning + quantificateur scalaire maison (uniforme, théorie Wyner-Ziv classique) | [https://pypi.org/project/pyldpc/](https://pypi.org/project/pyldpc/ "https://pypi.org/project/pyldpc/") | idem | quantificateur + reconstruction MMSE + wrapper |   
   
**Règle stricte respectée** : aucun paramètre n'est ajusté pour améliorer artificiellement les résultats. Les paramètres par défaut (qualité JPEG, taux du code LDPC, pas de quantification WZ) sont fixés a priori dans config.py et documentés, pas optimisés a posteriori par algorithme.  
**Génération de la side information (Slepian-Wolf / Wyner-Ziv)**  
UCID1338 ne contient pas de paires d'images corrélées (pas de stéréo, pas de séquence vidéo). Pour rester scientifiquement défendable (protocole standard en Distributed Source Coding, cf. Pradhan & Ramchandran, *"Distributed Source Coding Using Syndromes (DISCUS)"*, IEEE Trans. Info Theory 2003 ; Aaron & Girod,  *"Wyner-Ziv video coding"*, 2002 — qui simulent tous deux la corrélation source/side-info via un **canal virtuel**), la side information Y est générée à partir de l'image source X elle-même par une  **transformation contrôlée simulant un canal de corrélation** :  
Y = Q(X) + N,  N ~ Laplace(0, b)   (canal additif, modèle standard DSC)  
   
Paramètres fixes (config.py, jamais réajustés par image) : flou gaussien σ=1.2, bruit de Laplace d'échelle 3.0.  
**Impl** **émentation de ** ** SW / WZ**  
Deux problèmes ont été identifiés et corrigés pendant le développement, documentés ici pour la traçabilité :  
1. **Décomposition en plans de bits binaires classique → sensibilité excessive au bruit** (une différence de pixel de ±1 peut faire basculer plusieurs bits à la fois à cause des retenues binaires, ex. 127→128).  **Correction : code de Gray** appliqué avant la décomposition en plans de bits — deux valeurs proches ne diffèrent alors que d'1 bit. Pratique standard en codage source/vidéo corrélé.  
2. **Un rendement LDPC fixe (1/2) appliqué uniformément à tous les plans de bits ne peut pas converger sur les plans trop bruités.** C'est cohérent avec la théorie de Slepian-Wolf elle-même : le débit atteignable pour un plan donné est borné par H(X|Y) ; pour un plan proche de l'indépendance, H(X|Y)≈1, donc  **aucune compression n'est théoriquement possible**. Seuls certains plans passent donc par le LDPC ; les autres sont transmis tels quels, comptés intégralement dans S_encoded.  
3. **Point méthodologique important, corrigé en cours de développement** : la première version choisissait les plans à compresser par POSITION (les K premiers plans de poids fort après code de Gray), en supposant que position = stabilité. Cette hypothèse s'est révélée  **fausse** sur données de test — le plan de poids le plus fort avait en pratique un taux d'erreur de 18,7%, bien supérieur à des plans de poids plus faible. La stabilité réelle d'un plan dépend de la distribution des valeurs, pas de sa position binaire.  
4. **Correction : sélection des plans par CALIBRATION EMPIRIQUE**, à faire une seule fois avant le pilote :  
5. python -m core.calibrate_planes  
   
6. Ce script mesure, sur un échantillon du dataset (graine différente de celle du pilote), le taux d'erreur binaire réel de chaque plan sous le modèle de corrélation fixe, et recommande les indices à compresser (seuil fixe : <1% d'erreur, validé empiriquement pour la convergence du décodeur bit-flipping majoritaire). **Reporte les deux listes recommandées dans ** **config.SW_COMPRESSED_PLANE_INDICES** ** et ** **config.WZ_COMPRESSED_PLANE_INDICES** ** avant de lancer ** **core.pilot** **.** Les valeurs actuelles dans config.py sont des PLACEHOLDERS non vérifiés sur UCID1338.  
7. Ce protocole reste conforme à la règle "aucun paramètre ajusté pour améliorer les résultats" : la calibration est une mesure des propriétés du canal (fixée une fois, avant tout résultat de benchmark), pas un réglage a posteriori sur les résultats de compression eux-mêmes.  
8. Le décodeur par syndrome utilise l'algorithme de **bit-flipping à règle majoritaire** (Gallager, "Algorithme A", 1962) : à chaque itération, tous les bits dont plus de la moitié des contraintes voisines sont violées sont basculés en parallèle. C'est l'algorithme de décodage standard le plus simple pour les codes LDPC, correct par construction (contrairement à une première variante simplifiée testée et rejetée en cours de développement, qui pouvait dégrader la solution).  
**Limite connue ** : même restreinte aux plans MSB, la reconstruction exacte (lossless_exact) ne sera pas garantie à 100% sur toutes les images — elle dépend de la corrélation locale réelle entre image et side info simulée, qui varie d'une image à l'autre. C'est une observation légitime du protocole (la corrélation source/side-info y est simulée, pas mesurée sur des paires réelles), pas un bug caché.  
**Protocole d'exécution**  
1. **python -m core.calibrate_planes** → calibration à lancer UNE FOIS avant tout, reporter les indices recommandés dans config.py (cf. section précédente). Sans cette étape, SW/WZ utiliseront des placeholders non vérifiés.  
2. python -m core.pilot → lance le pilote sur N images (par défaut 15) tirées aléatoirement (graine fixe) de UCID1338, vérifie encodage/décodage/reconstruction/cohérence de tailles/erreurs, écrit output/pilot_results.csv.  
3. Après validation manuelle du pilote, python -m core.run_full → lance les 1338 images, écrit output/full_results.csv + output/summary_stats.csv.  
Aucune image originale n'est modifiée (tout se fait en mémoire / fichiers temporaires).  
**Installation**  
pip install pillow numpy scipy scikit-image tifffile ncompress pyldpc psutil  
   
