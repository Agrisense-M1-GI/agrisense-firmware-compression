#!/usr/bin/env python3
"""
jpeg_test.py -- branche reference/jpeg — version 2.1
------------------------------------------------------------------
Changements par rapport à v1 :
  - Lecture PNG directe (plus de PPM en entrée).
  - Suppression du modèle énergie (Eq. 1) : INA219 non fonctionnel.
  - Métriques retenues : cpu_time_ms, memory_kb, compressed_bytes,
    compression_ratio.

Usage :
    python3 jpeg_test.py <input.png> <output_dir>
"""
import os
import resource
import sys
import time

from PIL import Image


def get_memory_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    if len(sys.argv) != 3:
        print("Usage: jpeg_test.py <input.png> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    mem_before  = get_memory_kb()
    t_start     = time.time()

    img = Image.open(input_file).convert("RGB")
    # Taille originale en bytes bruts (RGB non compressé)
    original_size = img.width * img.height * 3

    compressed_path = os.path.join(output_dir, "compressed.jpg")
    img.save(compressed_path, "JPEG", quality=75, optimize=True)

    cpu_time_ms = (time.time() - t_start) * 1000
    mem_used    = max(get_memory_kb() - mem_before, 512)

    compressed_size    = os.path.getsize(compressed_path)
    compression_ratio  = original_size / compressed_size if compressed_size > 0 else 1.0

    metrics_path = os.path.join(output_dir, "node_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"cpu_time_ms,{cpu_time_ms:.3f}\n")
        f.write(f"memory_kb,{int(mem_used)}\n")
        f.write(f"compressed_bytes,{compressed_size}\n")
        f.write(f"compression_ratio,{compression_ratio:.2f}\n")

    print(f"JPEG: {compression_ratio:.2f}x, {compressed_size} bytes")


if __name__ == "__main__":
    main()
