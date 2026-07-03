#!/usr/bin/env python3
"""
pipeline_test.py -- branch reference/jpeg
------------------------------------------------------------------
Self-contained: only needs jpeg_test.py and jpeg2000_roi_test.py from
THIS branch. Runs both methods on every image of the dataset already
present on the Pi, and uploads original + artifacts to the reception
server (main.py) for server-side decoding, PSNR/SSIM/IoU/Dice
computation and results.csv logging -- same architecture already used
by the algo/adres and algo/wz-oseg branches, so the four methods stay
directly comparable in one results.csv.

Layout expected on the Pi after `git checkout reference/jpeg`:
    ~/agrisense-test/
        pipeline_test.py       (this file)
        jpeg_test.py
        jpeg2000_roi_test.py

Usage:
    python3 pipeline_test.py --dataset ~/DATASET/640X480-PPM \
        --server http://<laptop-ip>:8000 --node-id pi-test-01

Requires: pillow, requests (pip3 install --break-system-packages pillow requests)
Requires: libopenjp2-tools installed on the Pi (opj_compress in PATH)
    -> sudo apt install libopenjp2-tools
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
    """Same conversion convention as the ADRES/WZ-OSEG branches: force
    640x480 so all four methods are compared on identical geometry."""
    img = Image.open(src_path).convert("RGB")
    if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BICUBIC)
    img.save(dst_ppm, "PPM")


def run_encoder(script: str, args: list[str], label: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, script] + args, capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0:
            print(f"  [!] {label} a échoué (code {result.returncode}): {result.stderr.strip()[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [!] {label} a dépassé le timeout")
        return False
    except FileNotFoundError:
        print(f"  [!] Script introuvable: {script}")
        return False


def upload_result(server: str, node_id: str, image_id: str, ppm_path: Path,
                   jpeg_dir: Path, jp2roi_dir: Path) -> bool:
    opened = []
    try:
        files = {
            "original": (ppm_path.name, open(ppm_path, "rb")),
            "jpeg_compressed": ("compressed.jpg", open(jpeg_dir / "compressed.jpg", "rb")),
            "jpeg_metrics": ("node_metrics.txt", open(jpeg_dir / "node_metrics.txt", "rb")),
            "jp2roi_compressed": ("compressed.jp2roi", open(jp2roi_dir / "compressed.jp2roi", "rb")),
            "jp2roi_roi_mask": ("roi_mask.ppm", open(jp2roi_dir / "roi_mask.ppm", "rb")),
            "jp2roi_metrics": ("node_metrics.txt", open(jp2roi_dir / "node_metrics.txt", "rb")),
        }
        opened = [f for _, f in files.values()]

        url = f"{server}/test/submit/reference/{node_id}/{image_id}"
        resp = requests.post(url, files=files, timeout=90)

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
    parser = argparse.ArgumentParser(
        description="Test dataset complet sur JPEG (Pillow) + JPEG2000-ROI-2stream (branche reference/jpeg)."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--script-jpeg", default="./jpeg_test.py")
    parser.add_argument("--script-jp2roi", default="./jpeg2000_roi_test.py")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    work_dir = Path(args.work_dir)
    script_jpeg = args.script_jpeg
    script_jp2roi = args.script_jp2roi

    if not dataset_dir.is_dir():
        print(f"Dataset introuvable: {dataset_dir}")
        sys.exit(1)
    if not Path(script_jpeg).exists():
        print(f"Script JPEG introuvable: {script_jpeg}")
        sys.exit(1)
    if not Path(script_jp2roi).exists():
        print(f"Script JPEG2000-ROI introuvable: {script_jp2roi}")
        sys.exit(1)

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not images:
        print(f"Aucune image trouvée dans {dataset_dir}")
        sys.exit(1)

    print(f"[reference/jpeg] {len(images)} images à traiter. Serveur: {args.server}, node: {args.node_id}")

    ok_count, fail_count = 0, 0
    t_start = time.time()

    for i, img_path in enumerate(images, 1):
        image_id = img_path.stem
        print(f"[{i}/{len(images)}] {image_id}")

        img_work = work_dir / image_id
        jpeg_dir = img_work / "jpeg"
        jp2roi_dir = img_work / "jp2roi"
        for d in (jpeg_dir, jp2roi_dir):
            d.mkdir(parents=True, exist_ok=True)

        ppm_path = img_work / "input.ppm"
        try:
            to_ppm(img_path, ppm_path)
        except Exception as exc:
            print(f"  [!] Conversion PPM échouée: {exc}")
            fail_count += 1
            continue

        ok = True
        ok &= run_encoder(script_jpeg, [str(ppm_path), str(jpeg_dir)], "JPEG")
        ok &= run_encoder(script_jp2roi, [str(ppm_path), str(jp2roi_dir)], "JPEG2000-ROI-2stream")
        if not ok:
            fail_count += 1
            continue

        if upload_result(args.server, args.node_id, image_id, ppm_path, jpeg_dir, jp2roi_dir):
            ok_count += 1
            print("  -> OK")
        else:
            fail_count += 1

    elapsed = time.time() - t_start
    print(f"\n[reference/jpeg] Terminé en {elapsed:.1f}s. Succès: {ok_count}, Échecs: {fail_count}")


if __name__ == "__main__":
    main()
