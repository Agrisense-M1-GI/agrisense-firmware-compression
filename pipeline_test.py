#!/usr/bin/env python3
"""
pipeline_test.py -- WZ-OSEG branch (algo/wz-oseg)
------------------------------------------------------------------
Self-contained: only needs wz-oseg/encode from THIS branch. Does not
reference adres/ at all -- checking out this branch and running this
script works with no other branch involved.

Layout expected on the Pi after `git checkout algo/wz-oseg`:
    ~/agrisense-test/
        pipeline_test.py     (this file)
        wz-oseg/encode        <- gcc -O2 -o encode encode.cpp -lm

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


def upload_result(server: str, node_id: str, image_id: str, ppm_path: Path, wz_dir: Path) -> bool:
    opened = []
    try:
        files = {
            "original": (ppm_path.name, open(ppm_path, "rb")),
            "wz_compressed": ("compressed.p1", open(wz_dir / "compressed.p1", "rb")),
            "wz_otsu_mask": ("otsu_mask.ppm", open(wz_dir / "otsu_mask.ppm", "rb")),
            "wz_metrics": ("node_metrics.txt", open(wz_dir / "node_metrics.txt", "rb")),
        }
        opened = [f for _, f in files.values()]

        url = f"{server}/test/submit/wz-oseg/{node_id}/{image_id}"
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
    parser = argparse.ArgumentParser(description="Test dataset complet sur WZ-OSEG uniquement.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--wz-bin", default="./wz-oseg/encode")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    work_dir = Path(args.work_dir)
    wz_bin = Path(args.wz_bin)

    if not dataset_dir.is_dir():
        print(f"Dataset introuvable: {dataset_dir}")
        sys.exit(1)
    if not wz_bin.exists():
        print(f"Binaire WZ-OSEG introuvable: {wz_bin} (compile-le d'abord: gcc -O2 -o encode encode.cpp -lm)")
        sys.exit(1)

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not images:
        print(f"Aucune image trouvée dans {dataset_dir}")
        sys.exit(1)

    print(f"[WZ-OSEG] {len(images)} images à traiter. Serveur: {args.server}, node: {args.node_id}")

    ok_count, fail_count = 0, 0
    t_start = time.time()

    for i, img_path in enumerate(images, 1):
        image_id = img_path.stem
        print(f"[{i}/{len(images)}] {image_id}")

        img_work = work_dir / image_id
        wz_dir = img_work / "wz-oseg"
        wz_dir.mkdir(parents=True, exist_ok=True)

        ppm_path = img_work / "input.ppm"
        try:
            to_ppm(img_path, ppm_path)
        except Exception as exc:
            print(f"  [!] Conversion PPM échouée: {exc}")
            fail_count += 1
            continue

        if not run_encoder(wz_bin, [str(ppm_path), str(wz_dir)], "WZ-OSEG"):
            fail_count += 1
            continue

        if upload_result(args.server, args.node_id, image_id, ppm_path, wz_dir):
            ok_count += 1
            print("  -> OK")
        else:
            fail_count += 1

    elapsed = time.time() - t_start
    print(f"\n[WZ-OSEG] Terminé en {elapsed:.1f}s. Succès: {ok_count}, Échecs: {fail_count}")


if __name__ == "__main__":
    main()
