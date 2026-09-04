#!/usr/bin/env python3
"""
pipeline_test.py -- branche reference/jpeg — version 2.1
------------------------------------------------------------------
Changements par rapport à v1 :
  - Images source en PNG directement (plus de conversion PPM).
  - Masques SAM lus depuis ~/DATASET/mask/ et uploadés comme ground
    truth pour IoU/Dice côté serveur.
  - Skip automatique des images déjà traitées (GET /test/done_ids).
  - Suppression mesure énergie UART/ESP8266 (INA219 saturait au boot).
  - --dataset et --script-* ont des valeurs par défaut cohérentes
    avec ~/firmware/.

Layout attendu sur le Pi :
    ~/firmware/
        pipeline_test.py        (ce fichier)
        jpeg_test.py
        jpeg2000_roi_test.py
    ~/DATASET/
        images/img_xxxx.png
        mask/img_xxxx_mask.png  <- masques SAM

Usage :
    python3 pipeline_test.py \
        --server http://<laptop-ip>:8000 \
        --node-id pi-test-01

Dépendances :
    pip3 install --break-system-packages pillow requests
    sudo apt install libopenjp2-tools
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Masques SAM
# ---------------------------------------------------------------------------
def find_sam_mask(mask_dir: Path, image_stem: str) -> Path | None:
    for p in [mask_dir / f"{image_stem}_mask.png",
              mask_dir / f"{image_stem}.png"]:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Skip
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
# Encodage
# ---------------------------------------------------------------------------
def run_encoder(script: str, args: list, label: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            print(f"  [!] {label} a échoué (code {result.returncode}): "
                  f"{result.stderr.strip()[-400:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [!] {label} a dépassé le timeout")
        return False
    except FileNotFoundError:
        print(f"  [!] Script introuvable: {script}")
        return False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload_result(
    server: str, node_id: str, image_id: str,
    img_path: Path,
    jpeg_dir: Path, jp2roi_dir: Path,
    sam_mask_path: Path | None,
) -> bool:
    opened = []
    try:
        files = {
            "original":          (img_path.name,        open(img_path, "rb")),
            "jpeg_compressed":   ("compressed.jpg",      open(jpeg_dir   / "compressed.jpg",    "rb")),
            "jpeg_metrics":      ("node_metrics.txt",    open(jpeg_dir   / "node_metrics.txt",  "rb")),
            "jp2roi_compressed": ("compressed.jp2roi",   open(jp2roi_dir / "compressed.jp2roi", "rb")),
            "jp2roi_roi_mask":   ("roi_mask.png",        open(jp2roi_dir / "roi_mask.png",      "rb")),
            "jp2roi_metrics":    ("node_metrics.txt",    open(jp2roi_dir / "node_metrics.txt",  "rb")),
        }

        if sam_mask_path is not None:
            files["sam_mask"] = (sam_mask_path.name, open(sam_mask_path, "rb"))
        else:
            print("  [!] Masque SAM absent — le serveur utilisera le fallback")

        opened = [f for _, f in files.values()]

        url  = f"{server}/test/submit/reference/{node_id}/{image_id}"
        resp = requests.post(url, files=files, timeout=90)

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
    fw = Path.home() / "firmware"
    parser = argparse.ArgumentParser(
        description="Pipeline test JPEG + JPEG2000-ROI — PNG, masques SAM, sans énergie.")
    parser.add_argument("--dataset",      default=str(Path.home() / "DATASET"),
                        help="Chemin vers DATASET/. Défaut: ~/DATASET")
    parser.add_argument("--server",       required=True,
                        help="URL du serveur ex: http://192.168.1.10:8000")
    parser.add_argument("--node-id",      required=True)
    parser.add_argument("--work-dir",     default="./work")
    parser.add_argument("--script-jpeg",  default=str(fw / "jpeg_test.py"),
                        help="Défaut: ~/firmware/jpeg_test.py")
    parser.add_argument("--script-jp2roi",default=str(fw / "jpeg2000_roi_test.py"),
                        help="Défaut: ~/firmware/jpeg2000_roi_test.py")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir  = dataset_dir / "images"
    mask_dir    = dataset_dir / "mask"
    work_dir    = Path(args.work_dir)

    # Vérifications
    if not images_dir.is_dir():
        print(f"Dossier images introuvable : {images_dir}"); sys.exit(1)
    for label, script in [("JPEG", args.script_jpeg), ("JPEG2000-ROI", args.script_jp2roi)]:
        if not Path(script).exists():
            print(f"Script {label} introuvable : {script}"); sys.exit(1)
    if not mask_dir.is_dir():
        print(f"[Avertissement] Dossier mask introuvable : {mask_dir}")

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"Aucune image trouvée dans {images_dir}"); sys.exit(1)

    done_ids = fetch_done_ids(args.server)

    print(f"[reference] {len(images)} images trouvées. "
          f"Serveur: {args.server}, node: {args.node_id}")

    ok_count      = 0
    fail_count    = 0
    skip_count    = 0
    missing_masks = 0
    t_start       = time.time()

    for i, img_path in enumerate(images, 1):
        image_id = img_path.stem
        print(f"[{i}/{len(images)}] {image_id}")

        if image_id in done_ids:
            print("  -> Skip (déjà traité)")
            skip_count += 1
            continue

        # Masque SAM
        sam_mask_path = None
        if mask_dir.is_dir():
            sam_mask_path = find_sam_mask(mask_dir, image_id)
            if sam_mask_path is None:
                print(f"  [!] Masque SAM non trouvé pour {image_id}")
                missing_masks += 1

        # Dossiers de travail
        img_work   = work_dir / image_id
        jpeg_dir   = img_work / "jpeg"
        jp2roi_dir = img_work / "jp2roi"
        for d in (jpeg_dir, jp2roi_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Encodage — PNG passé directement aux scripts
        ok  = run_encoder(args.script_jpeg,   [str(img_path), str(jpeg_dir)],   "JPEG")
        ok &= run_encoder(args.script_jp2roi, [str(img_path), str(jp2roi_dir)], "JPEG2000-ROI")

        if not ok:
            fail_count += 1
            import shutil; shutil.rmtree(img_work, ignore_errors=True)
            continue

        if upload_result(args.server, args.node_id, image_id,
                         img_path, jpeg_dir, jp2roi_dir, sam_mask_path):
            ok_count += 1
            done_ids.add(image_id)
            print("  -> OK")
        else:
            fail_count += 1

        import shutil
        shutil.rmtree(img_work, ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\n[reference] Terminé en {elapsed:.1f}s — "
          f"OK: {ok_count}  Échecs: {fail_count}  Skips: {skip_count}")
    if missing_masks > 0:
        print(f"[Avertissement] {missing_masks} masques SAM manquants "
              f"sur {len(images)} images")


if __name__ == "__main__":
    main()
