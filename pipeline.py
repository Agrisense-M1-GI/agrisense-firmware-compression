#!/usr/bin/env python3
"""
pipeline.py — algo/jpeg-baseline, NORMAL mode
==============================================
One full production cycle (Section 3.2): live capture -> gate ->
segmentation -> compression -> energy measurement -> transmission ->
shutdown notification -> auto shutdown.

Called by startup.py (at the node root, outside git) when the station
reports mode = NORMAL. Not used in TEST mode — see pipeline_test.py.
"""

import os
import sys
import time
import uuid

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from capture import camera
from common import config, gate, segmentation, metrics, energy_uart, transmit
from compression import encode as compression


def main():
    image_id = f"live_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    total_t0 = time.time()

    # --- 1. Capture ------------------------------------------------
    frame_bgr, capture_time_ms = camera.capture_frame()
    input_size_bytes = frame_bgr.nbytes

    # --- Reference image for the gate -------------------------------
    if os.path.exists(config.LAST_CAPTURE_PATH):
        reference_img = cv2.imread(config.LAST_CAPTURE_PATH)
    else:
        # First run ever: nothing to compare to, force transmission.
        reference_img = frame_bgr.copy()

    meter = energy_uart.EnergyMeter()
    meter.start(image_id)

    # --- 2. Gate -----------------------------------------------------
    gate_result = gate.evaluate_gate(frame_bgr, reference_img)

    # --- 3. Segmentation (always run, per Section 4.2 step 3) --------
    seg_mask = segmentation.segment_otsu(frame_bgr)

    should_transmit = gate_result["gate_decision"] or not config.GATE_BLOCKS_TRANSMISSION

    compressed_bytes = b""
    compression_time_ms = 0.0
    if should_transmit:
        # --- 4. Compression (branch-specific) -------------------------
        compressed_bytes, extra = compression.encode(frame_bgr)
        compression_time_ms = extra.get("compression_time_ms", 0.0)

    meter.stop()
    meter.close()

    total_time_ms = (time.time() - total_t0) * 1000.0

    # --- 5. Node metrics ----------------------------------------------
    m = metrics.NodeMetrics(
        algorithm=config.BRANCH_NAME,
        image_id=image_id,
        mode="NORMAL",
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

    # --- 6. Transmission (WiFi) ----------------------------------------
    if should_transmit:
        transmit.send_to_station(image_id, compressed_bytes, m.__dict__)
        print(f"[pipeline] {image_id}: transmitted "
              f"({input_size_bytes} -> {len(compressed_bytes)} bytes)")
    else:
        print(f"[pipeline] {image_id}: gate decided NOT to transmit "
              f"(d_hist={gate_result['gate_d_hist']:.3f}, "
              f"d_mean={gate_result['gate_d_mean']:.3f}, "
              f"p_blocks={gate_result['gate_p_blocks']:.3f})")

    # --- 7. Update reference image for the next cycle -------------------
    cv2.imwrite(config.LAST_CAPTURE_PATH, frame_bgr)

    # --- 8. Notify ESP8266 that we're done, then shut down --------------
    _notify_shutdown_ready()
    _maybe_auto_shutdown()


def _notify_shutdown_ready():
    try:
        meter = energy_uart.EnergyMeter()
        meter._send(config.SHUTDOWN_READY_CMD)
        meter.close()
    except Exception as exc:
        print(f"[pipeline] WARNING: could not notify ESP8266 ({exc})")


def _maybe_auto_shutdown():
    # Guarded by an env var so this is safe to run on a dev machine.
    if os.environ.get("AGRISENSE_AUTO_SHUTDOWN", "0") == "1":
        os.system("sudo shutdown -h now")
    else:
        print("[pipeline] AGRISENSE_AUTO_SHUTDOWN not set to 1: "
              "skipping auto shutdown (dev mode).")


if __name__ == "__main__":
    main()
