#!/usr/bin/env python3
"""
pipeline_test.py -- ADRES branch (algo/adres) — version 2.1
------------------------------------------------------------------
Modifications par rapport à la v2.0 :
  - Les masques SAM sont lus depuis ~/DATASET/mask/ et uploadés
    avec les artefacts pour servir de ground truth côté serveur
    (remplace le Otsu recalculé côté serveur, qui n'était pas
    un vrai ground truth).
  - Les images source sont en PNG (inchangé depuis v2.0).
  - Suppression complète de la mesure énergie UART/ESP8266 :
    le module INA219 saturait à 3.2 A au boot du Raspberry Pi
    et n'a jamais démarré correctement. Les métriques retenues
    sont cpu_time_ms, memory_kb, compressed_bytes, ratio.
  - Skip automatique des images déjà traitées (inchangé).

Layout attendu sur le Pi :
    ~/firmware/
        pipeline_test.py
        adres/encode          <- gcc -O2 -o encode encode.cpp -lpng -lz -lm
    ~/DATASET/
        images/img_xxxx.png
        mask/img_xxxx_mask.png   <- masques SAM (PNG, même stem + _mask)

Usage :
    python3 pipeline_test.py \
        --dataset ~/DATASET \
        --server http://<laptop-ip>:8000 \
        --node-id pi-test-01

Dépendances :
    pip3 install --break-system-packages pillow requests
"""
import argparse
import sys
import time
from pathlib import Path

import requests
from PIL import Image

TARGET_WIDTH  = 640
TARGET_HEIGHT = 480
SUPPORTED_EXTENSIONS = {".png"}

import subprocess


# ---------------------------------------------------------------------------
# Chargement des image_id déjà traités depuis le serveur
# ---------------------------------------------------------------------------
def fetch_done_ids(server: str) -> set:
    try:
        resp = requests.get(f"{server}/test/done_ids", timeout=10)
        if resp.status_code == 200:
            ids = set(resp.json().get("done_ids", []))
            print(f"[Skip] {len(ids)} images déjà traitées récupérées du serveur")
            return ids
    except requests.exceptions.RequestException as exc:
        print(f"[Skip] Impossible de récupérer done_ids : {exc}")
    return set()


# ---------------------------------------------------------------------------
# Recherche du masque SAM correspondant à une image
# ---------------------------------------------------------------------------
def find_sam_mask(mask_dir: Path, image_stem: str) -> Path | None:
    """
    Cherche le masque SAM pour une image donnée.
    Conventions testées dans l'ordre :
      1. img_xxxx_mask.png  (convention principale du dataset)
      2. img_xxxx.png       (même nom que l'image)
    Retourne None si aucun masque trouvé.
    """
    candidates = [
        mask_dir / f"{image_stem}_mask.png",
        mask_dir / f"{image_stem}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Encodage
# ---------------------------------------------------------------------------
def run_encoder(binary: Path, args: list, label: str) -> bool:
    try:
        result = subprocess.run(
            [str(binary)] + args,
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  [!] {label} a échoué (code {result.returncode}): "
                  f"{result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [!] {label} a dépassé le timeout")
        return False
    except FileNotFoundError:
        print(f"  [!] Binaire introuvable: {binary}")
        return False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload_result(
    server: str, node_id: str, image_id: str,
    img_path: Path,
    sam_mask_path: Path | None,
    adres_q_dir: Path, adres_e_dir: Path,
) -> bool:
    opened = []
    try:
        files = {
            "original":           (img_path.name,  open(img_path, "rb")),
            "adres_q_compressed": ("compressed.p2", open(adres_q_dir / "compressed.p2",    "rb")),
            "adres_q_roi_mask":   ("roi_mask.png",  open(adres_q_dir / "roi_mask.png",     "rb")),
            "adres_q_metrics":    ("node_metrics.txt", open(adres_q_dir / "node_metrics.txt", "rb")),
            "adres_e_compressed": ("compressed.p2", open(adres_e_dir / "compressed.p2",    "rb")),
            "adres_e_roi_mask":   ("roi_mask.png",  open(adres_e_dir / "roi_mask.png",     "rb")),
            "adres_e_metrics":    ("node_metrics.txt", open(adres_e_dir / "node_metrics.txt", "rb")),
        }

        # Ajout du masque SAM si disponible — le serveur l'utilisera
        # comme ground truth à la place du Otsu recalculé
        if sam_mask_path is not None:
            files["sam_mask"] = (sam_mask_path.name, open(sam_mask_path, "rb"))
        else:
            print("  [!] Masque SAM absent — le serveur utilisera Otsu comme référence")

        opened = [f for _, f in files.values()]

        url  = f"{server}/test/submit/adres/{node_id}/{image_id}"
        resp = requests.post(url, files=files, timeout=60)

        if resp.status_code != 200:
            print(f"  [!] Upload refusé ({resp.status_code}): {resp.text[:300]}")
            return False

        data = resp.json()
        if data.get("status") == "skipped":
            print("  -> Déjà traité (skip serveur)")
        return True

    except requests.exceptions.RequestException as exc:
        print(f"  [!] Erreur réseau : {exc}")
        return False
    finally:
        for f in opened:
            f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline test ADRES — PNG, masques SAM, sans mesure énergie.")
    parser.add_argument("--dataset",  default=str(Path.home() / "DATASET"),
                        help="Chemin vers DATASET/ (contient images/ et mask/). "
                             "Défaut: ~/DATASET")
    parser.add_argument("--server",   required=True,
                        help="URL du serveur ex: http://192.168.1.10:8000")
    parser.add_argument("--node-id",  required=True)
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--adres-bin", default=str(Path.home() / "firmware" / "adres" / "encode"),
                        help="Chemin vers le binaire encode. "
                             "Défaut: ~/firmware/adres/encode")
    args = parser.parse_args()

    dataset_dir  = Path(args.dataset)
    images_dir   = dataset_dir / "images"
    mask_dir     = dataset_dir / "mask"
    work_dir     = Path(args.work_dir)
    adres_bin    = Path(args.adres_bin)

    # Vérifications
    if not images_dir.is_dir():
        print(f"Dossier images introuvable : {images_dir}")
        sys.exit(1)
    if not adres_bin.exists():
        print(f"Binaire ADRES introuvable : {adres_bin}")
        sys.exit(1)
    if not mask_dir.is_dir():
        print(f"[Avertissement] Dossier mask introuvable : {mask_dir}")
        print("  Les masques SAM ne seront pas uploadés.")

    # Liste des images PNG triées
    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"Aucune image PNG trouvée dans {images_dir}")
        sys.exit(1)

    # Récupération des image_id déjà traités
    done_ids = fetch_done_ids(args.server)

    print(f"[ADRES] {len(images)} images trouvées. "
          f"Serveur: {args.server}, node: {args.node_id}")

    ok_count   = 0
    fail_count = 0
    skip_count = 0
    missing_masks = 0
    t_start    = time.time()

    for i, img_path in enumerate(images, 1):
        image_id = img_path.stem  # "img_0001"
        print(f"[{i}/{len(images)}] {image_id}")

        # Skip local si déjà traité
        if image_id in done_ids:
            print("  -> Skip (déjà traité)")
            skip_count += 1
            continue

        # Masque SAM correspondant
        sam_mask_path = None
        if mask_dir.is_dir():
            sam_mask_path = find_sam_mask(mask_dir, image_id)
            if sam_mask_path is None:
                print(f"  [!] Masque SAM non trouvé pour {image_id}")
                missing_masks += 1

        # Dossiers de travail
        img_work    = work_dir / image_id
        adres_q_dir = img_work / "adres_q"
        adres_e_dir = img_work / "adres_e"
        for d in (adres_q_dir, adres_e_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Encodage Q et E
        ok  = run_encoder(adres_bin,
                          [str(img_path), str(adres_q_dir), "Q"], "ADRES-Q")
        ok &= run_encoder(adres_bin,
                          [str(img_path), str(adres_e_dir), "E"], "ADRES-E")

        if not ok:
            fail_count += 1
            import shutil; shutil.rmtree(img_work, ignore_errors=True)
            continue

        # Upload vers le serveur
        if upload_result(args.server, args.node_id, image_id,
                         img_path, sam_mask_path, adres_q_dir, adres_e_dir):
            ok_count += 1
            done_ids.add(image_id)
            print("  -> OK")
        else:
            fail_count += 1

        # Nettoyage local
        import shutil
        shutil.rmtree(img_work, ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\n[ADRES] Terminé en {elapsed:.1f}s — "
          f"OK: {ok_count}  Échecs: {fail_count}  Skips: {skip_count}")
    if missing_masks > 0:
        print(f"[Avertissement] {missing_masks} masques SAM manquants sur {len(images)} images")


if __name__ == "__main__":
    main()
