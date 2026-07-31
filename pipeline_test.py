#!/usr/bin/env python3
"""
pipeline_test.py — algo/jpeg-baseline, TEST mode
==================================================
Processes the entire reference dataset (Campaign 0) in one batch
(Section 3.2 / 9.1). This branch does not run the change-detection gate
or segmentation/VARI classification: only what jpeg-baseline itself
needs (compression) sits inside the energy-measurement window. Gate and
segmentation validation are handled on their own dedicated branches, so
that neither introduces uncontrolled overhead into this branch's
energy/performance numbers. Compression and transmission always happen
for every image in the dataset.

No shutdown notification, no auto shutdown (Section 3.2): the Pi is
stopped manually by SSH once the whole dataset has been processed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from common import config, metrics, energy_uart, transmit, system_metrics
from compression import encode as compression


def iter_dataset_images(dataset_dir: str = config.DATASET_DIR):
    """
    Yields (image_id, png_path, ppm_path) for every img_NNNN.png in the
    dataset, sorted. Each PNG has a lossless PPM twin with the same
    basename (provided ahead of time in the dataset), so compression can
    read the PPM directly without decoding/re-encoding on the fly.
    """
    filenames = sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith(".png"))
    for filename in filenames:
        image_id = os.path.splitext(filename)[0]  # e.g. "img_0001"
        png_path = os.path.join(dataset_dir, filename)
        ppm_path = os.path.join(dataset_dir, image_id + ".ppm")
        if not os.path.exists(ppm_path):
            raise FileNotFoundError(f"Missing PPM twin for {filename}: {ppm_path}")
        yield image_id, png_path, ppm_path


def main():
    meter = energy_uart.EnergyMeter()
    processed, transmitted = 0, 0

    for image_id, png_path, ppm_path in iter_dataset_images():
        total_t0 = time.time()

        t_read0 = time.time()
        input_size_bytes = os.path.getsize(png_path)  # native reference size
        capture_time_ms = (time.time() - t_read0) * 1000.0  # "capture" = file lookup here

        meter.start(image_id)

        compressed_bytes, extra = compression.encode_from_ppm(ppm_path)
        compression_time_ms = extra.get("compression_time_ms", 0.0)

        meter.stop()

        total_time_ms = (time.time() - total_t0) * 1000.0

        cpu_freq_hz = system_metrics.read_cpu_freq_hz()
        cpu_temp_c = system_metrics.read_cpu_temp_c()
        ram_used_mb = system_metrics.read_ram_used_mb()

        m = metrics.NodeMetrics(
            algorithm=config.BRANCH_NAME,
            image_id=image_id,
            mode="TEST",
            input_size_bytes=input_size_bytes,
            output_size_bytes=len(compressed_bytes),
            capture_time_ms=capture_time_ms,
            compression_time_ms=compression_time_ms,
            total_pipeline_time_ms=total_time_ms,
            cpu_freq_hz=cpu_freq_hz,
            cpu_temp_c=cpu_temp_c,
            ram_used_mb=ram_used_mb,
        )
        m = metrics.finalize(m)
        metrics.append_csv(m)

        ok = transmit.send_to_station(image_id, compressed_bytes, m.__dict__)
        transmitted += int(ok)
        processed += 1

        print(f"[pipeline_test] {image_id}: "
              f"{input_size_bytes} -> {len(compressed_bytes)} bytes")

    meter.close()
    print(f"[pipeline_test] done: {processed} images processed, "
          f"{transmitted} successfully sent to the station.")
    print("[pipeline_test] Pi left running — stop it manually via SSH when ready.")


if __name__ == "__main__":
    main()
