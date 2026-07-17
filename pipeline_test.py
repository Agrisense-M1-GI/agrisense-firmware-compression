#!/usr/bin/env python3
"""
pipeline_test.py — algo/jpeg-baseline, TEST mode
==================================================
Processes the entire reference dataset (Campaign 0) in one batch
(Section 3.2 / 9.1). Unlike NORMAL mode, the gate NEVER actually blocks
anything here: it is still evaluated and logged on every image (needed
for Campaign 2's later validation against ground truth), but compression
and transmission always happen, so Campaign 1's ablation study gets a
full set of quality/performance/energy metrics for every image in the
dataset regardless of what the gate would have decided in the field.

No shutdown notification, no auto shutdown (Section 3.2): the Pi is
stopped manually by SSH once the whole dataset has been processed.
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from common import config, gate, segmentation, metrics, energy_uart, transmit
from compression import encode as compression


def iter_dataset_images(dataset_dir: str = config.DATASET_DIR):
    """Yields (image_id, path) for every img_NNNN.png in the dataset, sorted."""
    filenames = sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith(".png"))
    for filename in filenames:
        image_id = os.path.splitext(filename)[0]  # e.g. "img_0001"
        yield image_id, os.path.join(dataset_dir, filename)


def main():
    meter = energy_uart.EnergyMeter()
    previous_img = None
    processed, transmitted = 0, 0

    for image_id, path in iter_dataset_images():
        total_t0 = time.time()

        t_read0 = time.time()
        img_bgr = cv2.imread(path)
        capture_time_ms = (time.time() - t_read0) * 1000.0  # "capture" = read from disk here
        input_size_bytes = os.path.getsize(path)

        if img_bgr is None:
            print(f"[pipeline_test] WARNING: could not read {path}, skipping.")
            continue

        reference_img = previous_img if previous_img is not None else img_bgr

        meter.start(image_id)

        gate_result = gate.evaluate_gate(img_bgr, reference_img)
        _ = segmentation.segment_otsu(img_bgr)  # computed for consistency, see Section 4.2 step 3

        compressed_bytes, extra = compression.encode(img_bgr)
        compression_time_ms = extra.get("compression_time_ms", 0.0)

        meter.stop()

        total_time_ms = (time.time() - total_t0) * 1000.0

        m = metrics.NodeMetrics(
            algorithm=config.BRANCH_NAME,
            image_id=image_id,
            mode="TEST",
            input_size_bytes=input_size_bytes,
            output_size_bytes=len(compressed_bytes),
            gate_decision=gate_result["gate_decision"],
            gate_d_hist=gate_result["gate_d_hist"],
            gate_d_mean=gate_result["gate_d_mean"],
            gate_p_blocks=gate_result["gate_p_blocks"],
            capture_time_ms=capture_time_ms,
            compression_time_ms=compression_time_ms,
            total_pipeline_time_ms=total_time_ms,
        )
        m = metrics.finalize(m)
        metrics.append_csv(m)

        # TEST mode always transmits, regardless of the gate's decision
        # (see module docstring above).
        ok = transmit.send_to_station(image_id, compressed_bytes, m.__dict__)
        transmitted += int(ok)
        processed += 1

        previous_img = img_bgr

        print(f"[pipeline_test] {image_id}: "
              f"{input_size_bytes} -> {len(compressed_bytes)} bytes "
              f"(gate would transmit: {gate_result['gate_decision']})")

    meter.close()
    print(f"[pipeline_test] done: {processed} images processed, "
          f"{transmitted} successfully sent to the station.")
    print("[pipeline_test] Pi left running — stop it manually via SSH when ready.")


if __name__ == "__main__":
    main()
