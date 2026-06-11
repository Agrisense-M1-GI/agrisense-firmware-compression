#!/usr/bin/env python3
"""
AgriSense - Pipeline algo/mjpeg-webcam
Compression MJPEG embarquée dans la webcam USB.

Métriques collectées :
- Taille entrée / sortie / taux de compression
- Temps de capture et temps total pipeline
- RAM utilisée (pic)
- Fréquence CPU (via vcgencmd)
- Température CPU (via vcgencmd)
- Énergie estimée (modèle linéaire basé sur fréquence + durée)

Note : PSNR et SSIM ne sont pas calculables pour cette branche car
la webcam compresse en interne — l'image brute n'est pas accessible.
"""
import json
import os
import sys
import time
import subprocess
import resource
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
INPUT_SIZE  = 1920 * 1080 * 2  # YUYV : 2 octets par pixel avant compression

# Modèle énergétique Pi Zero 2W (valeurs constructeur / mesures publiées)
# Idle ~ 0.5W, charge maximale ~ 2.5W à 1GHz
# On estime : P(W) = 0.5 + 2.0 * (freq_hz / 1_000_000_000)
# E(J) = P * t(s)
POWER_IDLE_W    = 0.5
POWER_SCALE_W   = 2.0
FREQ_MAX_HZ     = 1_000_000_000


# ---------------------------------------------------------------------------
# Métriques système
# ---------------------------------------------------------------------------
def get_cpu_freq_hz() -> int:
    """Fréquence CPU actuelle via vcgencmd (Hz)."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_clock", "arm"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        # format : "frequency(48)=600062000"
        return int(out.split("=")[1])
    except Exception:
        return 600_000_000  # valeur par défaut si échec


def get_cpu_temp_c() -> float:
    """Température CPU via vcgencmd (°C)."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        # format : "temp=42.4'C"
        return float(out.replace("temp=", "").replace("'C", ""))
    except Exception:
        return -1.0


def get_ram_used_mb() -> float:
    """RAM utilisée par le processus courant (pic, en Mo)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024, 2)  # Linux : ru_maxrss en Ko


def estimate_energy_j(freq_hz: int, duration_s: float) -> float:
    """
    Estimation de l'énergie consommée pendant le pipeline (Joules).
    Modèle linéaire : P = P_idle + P_scale * (freq / freq_max)
    E = P * t
    """
    power_w = POWER_IDLE_W + POWER_SCALE_W * (freq_hz / FREQ_MAX_HZ)
    return round(power_w * duration_s, 4)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Métriques système au démarrage
    freq_hz   = get_cpu_freq_hz()
    temp_c    = get_cpu_temp_c()
    t_start   = time.time()

    # Étape 1 : capture
    t_capture_start = time.time()
    image_path = capture()
    capture_time_ms = round((time.time() - t_capture_start) * 1000, 1)

    # Étape 2 : métriques de compression
    output_size      = image_path.stat().st_size
    compression_ratio = round(INPUT_SIZE / output_size, 3) if output_size > 0 else 0.0
    total_time_s     = time.time() - t_start
    total_time_ms    = round(total_time_s * 1000, 1)
    ram_mb           = get_ram_used_mb()
    energy_j         = estimate_energy_j(freq_hz, total_time_s)

    metadata = {
        "timestamp":               datetime.now().isoformat(),
        "algorithm":               "mjpeg-webcam",
        "resolution":              "1920x1080",
        # Compression
        "input_size_bytes":        INPUT_SIZE,
        "output_size_bytes":       output_size,
        "compression_ratio":       compression_ratio,
        "psnr_db":                 None,
        "ssim":                    None,
        # Temps
        "capture_time_ms":         capture_time_ms,
        "compression_time_ms":     capture_time_ms,  # intégré dans la capture webcam
        "total_pipeline_time_ms":  total_time_ms,
        # Système
        "cpu_freq_hz":             freq_hz,
        "cpu_temp_c":              temp_c,
        "ram_used_mb":             ram_mb,
        # Énergie (estimée)
        "energy_j":                energy_j,
        "energy_model":            "P=0.5+2.0*(freq/1GHz), E=P*t",
    }

    # Sauvegarder le JSON
    json_path = OUTPUT_DIR / f"metadata_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Taux de compression : {compression_ratio:.2f}x")
    print(f"Temps total         : {total_time_ms:.0f} ms")
    print(f"RAM utilisée        : {ram_mb} Mo")
    print(f"Température CPU     : {temp_c} °C")
    print(f"Fréquence CPU       : {freq_hz/1e6:.0f} MHz")
    print(f"Énergie estimée     : {energy_j} J")

    # Étape 3 : envoi au serveur
    base_url = f"http://{SERVER_IP}:{SERVER_PORT}/node/{NODE_ID}"
    try:
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{base_url}/upload/image",
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=30
            )
            r.raise_for_status()
        print(f"Image envoyée : {r.json()}")

        with open(json_path, "rb") as f:
            r = requests.post(
                f"{base_url}/upload/metrics",
                files={"file": (json_path.name, f, "application/json")},
                timeout=10
            )
            r.raise_for_status()
        print(f"Métriques envoyées : {r.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Erreur envoi serveur : {e}", file=sys.stderr)
        sys.exit(1)

    print("Pipeline terminé avec succès")
    sys.exit(0)


if __name__ == "__main__":
    main()