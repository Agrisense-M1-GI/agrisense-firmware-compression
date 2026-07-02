#!/usr/bin/env python3
"""
pipeline_test.py -- ADRES branch (algo/adres)
------------------------------------------------------------------
Self-contained: only needs adres/encode from THIS branch. Does not
reference wz-oseg/ at all -- checking out this branch and running
this script works with no other branch involved.

Layout expected on the Pi after `git checkout algo/adres`:
    ~/agrisense-test/
        pipeline_test.py     (this file)
        adres/encode          <- gcc -O2 -o encode encode.cpp -lpng -lz -lm

Usage:
    python3 pipeline_test.py --dataset ~/DATASET/640X480-PPM \
        --server http://<laptop-ip>:8000 --node-id pi-test-01

Requires: pillow, requests (pip3 install --break-system-packages pillow requests)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
import requests

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm"}


def to_ppm(src_path: Path, dst_ppm: Path) -> None:
    img = Image.open(src_path).convert("RGB")
    if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BICUBIC)
    img.save(dst_ppm, "PPM")


def run_encoder(binary: Path, args: list[str], label: str) -> bool:
    try:
        result = subprocess.run([str(binary)] + args, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  [!] {label} a échoué (code {result.returncode}): {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [!] {label} a dépassé le timeout")
        return False
    except FileNotFoundError:
        print(f"  [!] Binaire introuvable: {binary}")
        return False


def upload_result(server: str, node_id: str, image_id: str, ppm_path: Path,
                   adres_q_dir: Path, adres_e_dir: Path) -> bool:
    opened = []
    try:
        files = {
            "original": (ppm_path.name, open(ppm_path, "rb")),
            "adres_q_compressed": ("compressed.p2", open(adres_q_dir / "compressed.p2", "rb")),
            "adres_q_roi_mask": ("roi_mask.ppm", open(adres_q_dir / "roi_mask.ppm", "rb")),
            "adres_q_metrics": ("node_metrics.txt", open(adres_q_dir / "node_metrics.txt", "rb")),
            "adres_e_compressed": ("compressed.p2", open(adres_e_dir / "compressed.p2", "rb")),
            "adres_e_roi_mask": ("roi_mask.ppm", open(adres_e_dir / "roi_mask.ppm", "rb")),
            "adres_e_metrics": ("node_metrics.txt", open(adres_e_dir / "node_metrics.txt", "rb")),
        }
        opened = [f for _, f in files.values()]

        url = f"{server}/test/submit/adres/{node_id}/{image_id}"
        resp = requests.post(url, files=files, timeout=60)

        if resp.status_code != 200:
            print(f"  [!] Upload refusé ({resp.status_code}): {resp.text[:300]}")
            return False
        return True
    except requests.exceptions.RequestException as exc:
        print(f"  [!] Erreur réseau lors de l'upload: {exc}")
        return False
    finally:
        for f in opened:
            f.close()


def main():
    parser = argparse.ArgumentParser(description="Test dataset complet sur ADRES uniquement (profils Q et E).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--adres-bin", default="./adres/encode")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    work_dir = Path(args.work_dir)
    adres_bin = Path(args.adres_bin)

    if not dataset_dir.is_dir():
        print(f"Dataset introuvable: {dataset_dir}")
        sys.exit(1)
    if not adres_bin.exists():
        print(f"Binaire ADRES introuvable: {adres_bin} (compile-le d'abord: gcc -O2 -o encode encode.cpp -lpng -lz -lm)")
        sys.exit(1)

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not images:
        print(f"Aucune image trouvée dans {dataset_dir}")
        sys.exit(1)

    print(f"[ADRES] {len(images)} images à traiter. Serveur: {args.server}, node: {args.node_id}")

    ok_count, fail_count = 0, 0
    t_start = time.time()

    for i, img_path in enumerate(images, 1):
        image_id = img_path.stem
        print(f"[{i}/{len(images)}] {image_id}")

        img_work = work_dir / image_id
        adres_q_dir = img_work / "adres_q"
        adres_e_dir = img_work / "adres_e"
        for d in (adres_q_dir, adres_e_dir):
            d.mkdir(parents=True, exist_ok=True)

        ppm_path = img_work / "input.ppm"
        try:
            to_ppm(img_path, ppm_path)
        except Exception as exc:
            print(f"  [!] Conversion PPM échouée: {exc}")
            fail_count += 1
            continue

        ok = True
        ok &= run_encoder(adres_bin, [str(ppm_path), str(adres_q_dir), "Q"], "ADRES-Q")
        ok &= run_encoder(adres_bin, [str(ppm_path), str(adres_e_dir), "E"], "ADRES-E")
        if not ok:
            fail_count += 1
            continue

        if upload_result(args.server, args.node_id, image_id, ppm_path, adres_q_dir, adres_e_dir):
            ok_count += 1
            print("  -> OK")
        else:
            fail_count += 1

    elapsed = time.time() - t_start
    print(f"\n[ADRES] Terminé en {elapsed:.1f}s. Succès: {ok_count}, Échecs: {fail_count}")


if __name__ == "__main__":
    main()
