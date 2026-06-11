#!/usr/bin/env python3
"""
Capture une image YUYV depuis la webcam USB et la convertit en PPM.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DEVICE      = "/dev/video0"
WIDTH       = 1920
HEIGHT      = 1080
OUTPUT_DIR  = Path(__file__).parent.parent / "output"

def capture() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f"capture_{timestamp}.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", DEVICE,
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur capture : {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Image capturée : {out_path}")
    return out_path

if __name__ == "__main__":
    capture()
