#!/usr/bin/env python3
"""
AgriSense - Pipeline algo/mjpeg-webcam
Compression MJPEG embarquée dans la webcam USB.
La webcam compresse elle-même en MJPEG/JPEG, on récupère directement le flux.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import requests

sys.path.insert(0, str(Path(__file__).parent))
from capture.capture import capture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_IP   = "192.168.0.150"
SERVER_PORT = 8000
NODE_ID     = "NODE01"
OUTPUT_DIR  = Path(__file__).parent / "output"
INPUT_SIZE  = 1920 * 1080 * 2  # YUYV : 2 octets par pixel

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t_start = time.time()

    # Étape 1 : capture
    t_capture = time.time()
    image_path = capture()
    t_capture = (time.time() - t_capture) * 1000

    # Étape 2 : métriques
    output_size = image_path.stat().st_size
    compression_ratio = INPUT_SIZE / output_size if output_size > 0 else 0.0

    t_total = (time.time() - t_start) * 1000

    metadata = {
        "timestamp":             datetime.now().isoformat(),
        "algorithm":             "mjpeg-webcam",
        "input_size_bytes":      INPUT_SIZE,
        "output_size_bytes":     output_size,
        "compression_ratio":     round(compression_ratio, 3),
        "psnr_db":               None,  # non calculable sans image brute
        "ssim":                  None,  # non calculable sans image brute
        "compression_time_ms":   round(t_capture, 1),
        "total_pipeline_time_ms":round(t_total, 1),
        "resolution":            "1920x1080",
    }

    # Sauvegarder le JSON
    json_path = OUTPUT_DIR / f"metadata_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Métadonnées : {json_path}")
    print(f"Taux de compression : {compression_ratio:.2f}x")

    # Étape 3 : envoi au serveur
    base_url = f"http://{SERVER_IP}:{SERVER_PORT}/node/{NODE_ID}"
    try:
        with open(image_path, "rb") as f:
            r = requests.post(f"{base_url}/upload/image",
                              files={"file": (image_path.name, f, "image/jpeg")},
                              timeout=30)
            r.raise_for_status()
        print(f"Image envoyée : {r.json()}")

        with open(json_path, "rb") as f:
            r = requests.post(f"{base_url}/upload/metrics",
                              files={"file": (json_path.name, f, "application/json")},
                              timeout=10)
            r.raise_for_status()
        print(f"Métriques envoyées : {r.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Erreur envoi serveur : {e}", file=sys.stderr)
        sys.exit(1)

    print("Pipeline terminé avec succès")
    sys.exit(0)

if __name__ == "__main__":
    main()
