#!/usr/bin/env python3
import sys
import os
import time
import resource
from PIL import Image

def estimate_energy(cpu_ms, mem_kb, tx_bytes):
    """Estimation énergie (modèle capteur)"""
    e_cpu = cpu_ms * 0.0054
    e_mem = mem_kb * 0.00001
    e_tx = tx_bytes * 0.18
    return e_cpu + e_mem + e_tx

def get_memory_kb():
    """Mémoire utilisée en KB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def main():
    if len(sys.argv) != 3:
        print("Usage: jpeg_test.py <input.ppm> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Métriques initiales
    mem_before = get_memory_kb()
    start_time = time.time()
    
    # Chargement image
    img = Image.open(input_file)
    original_size = os.path.getsize(input_file)
    
    # Compression JPEG (quality 75)
    compressed_path = os.path.join(output_dir, "compressed.jpg")
    img.save(compressed_path, "JPEG", quality=75, optimize=True)
    
    # Reconstruction (décodage) - sauvegarde PNG pour visualisation
    reconstructed_path = os.path.join(output_dir, "reconstructed.png")
    img_reconstructed = Image.open(compressed_path)
    img_reconstructed.save(reconstructed_path, "PNG")
    
    # Métriques finales
    end_time = time.time()
    cpu_time = (end_time - start_time) * 1000  # ms
    
    mem_after = get_memory_kb()
    mem_used = max(mem_after - mem_before, 512)  # minimum 512KB
    
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size
    
    energy = estimate_energy(cpu_time, mem_used, compressed_size)
    
    # Sauvegarde métriques nœud
    metrics_path = os.path.join(output_dir, "node_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"cpu_time_ms,{cpu_time:.3f}\n")
        f.write(f"memory_kb,{int(mem_used)}\n")
        f.write(f"compressed_bytes,{compressed_size}\n")
        f.write(f"compression_ratio,{compression_ratio:.2f}\n")
        f.write(f"energy_mj,{energy:.3f}\n")
    
    print(f"JPEG: {compression_ratio:.2f}x, {compressed_size} bytes")

if __name__ == "__main__":
    main()