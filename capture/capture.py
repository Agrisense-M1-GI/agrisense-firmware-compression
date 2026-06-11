#!/usr/bin/env python3
"""
Capture une image MJPEG depuis la webcam USB via v4l2-ctl.
Résolution maximale : 1920x1080
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DEVICE     = "/dev/video0"
WIDTH      = 1920
HEIGHT     = 1080
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def capture() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f"capture_{timestamp}.jpg"

    cmd = [
        "v4l2-ctl",
        f"--device={DEVICE}",
        f"--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=MJPG",
        "--stream-mmap",
        "--stream-count=1",
        f"--stream-to={str(out_path)}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur capture : {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Image capturée : {out_path} ({out_path.stat().st_size} octets)")
    return out_path

if __name__ == "__main__":
    capture()