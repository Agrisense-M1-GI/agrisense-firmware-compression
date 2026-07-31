#!/usr/bin/env python3
"""
pipeline.py — algo/jpeg-baseline, NORMAL mode
==============================================
One full production cycle (Section 3.2): live capture -> compression ->
energy measurement -> transmission -> shutdown notification -> auto
shutdown.

Called by startup.py (at the node root, outside git) when the station
reports mode = NORMAL. Not used in TEST mode — see pipeline_test.py.

Note: this branch does NOT run the change-detection gate or
segmentation/VARI classification. jpeg-baseline only measures raw JPEG
compression performance, so those steps are intentionally omitted from
the measured path — running them here would add uncontrolled overhead
to the energy measurement and is out of scope for this branch (gate
validation is done on its own dedicated branch). Transmission is
therefore systematic: every capture is compressed and sent.
"""

import os
import sys
import time
import uuid

from capture import camera
from common import config, metrics, energy_uart, transmit, system_metrics
from compression import encode as compression


def main():
    image_id = f"live_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    total_t0 = time.time()

    # --- 1. Capture ------------------------------------------------
    frame_bgr, capture_time_ms = camera.capture_frame()
    input_size_bytes = frame_bgr.nbytes

    meter = energy_uart.EnergyMeter()
    meter.start(image_id)

    # --- 2. Compression (branch-specific) -------------------------
    compressed_bytes, extra = compression.encode(frame_bgr)
    compression_time_ms = extra.get("compression_time_ms", 0.0)

    meter.stop()
    meter.close()

    total_time_ms = (time.time() - total_t0) * 1000.0

    # --- 3. System metrics (Section 6.1), read once per image -------------
    cpu_freq_hz = system_metrics.read_cpu_freq_hz()
    cpu_temp_c = system_metrics.read_cpu_temp_c()
    ram_used_mb = system_metrics.read_ram_used_mb()

    # --- 4. Node metrics ----------------------------------------------
    m = metrics.NodeMetrics(
        algorithm=config.BRANCH_NAME,
        image_id=image_id,
        mode="NORMAL",
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

    # --- 5. Transmission (WiFi) — systematic, no gate on this branch ---
    transmit.send_to_station(image_id, compressed_bytes, m.__dict__)
    print(f"[pipeline] {image_id}: transmitted "
          f"({input_size_bytes} -> {len(compressed_bytes)} bytes)")

    # --- 6. Notify ESP8266 that we're done, then shut down --------------
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