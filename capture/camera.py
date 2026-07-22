"""
camera.py
=========
Live capture from the USB webcam, used only in NORMAL mode (pipeline.py).
TEST mode (pipeline_test.py) reads from the fixed reference dataset instead
and does not use this module.

Captures at 1920x1080 (Section 2.2) in MJPG, this webcam's native mode at
that resolution: `v4l2-ctl --list-formats-ext` shows YUYV only reaches
5 fps at 1920x1080 (0.2s/frame) versus 30 fps in MJPG, and `--get-fmt-video`
confirms the driver already defaults to MJPG. OpenCV's V4L2 backend decodes
MJPG to a BGR array transparently on cap.read(), so there's no need for raw
YUYV here. Auto-exposure is on by default on this camera
(`auto_exposure = Aperture Priority Mode`), so a few frames are grabbed and
discarded first to let it converge before the kept frame is captured.
"""

import time

import cv2

from common import config

WARMUP_FRAMES = 5


def capture_frame():
    """
    Opens the webcam, discards a few warm-up frames (auto-exposure
    settling), grabs one frame, releases it.
    Returns (frame_bgr, capture_time_ms).
    """
    t0 = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam (index 0).")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT):
        cap.release()
        raise RuntimeError(
            f"Webcam negotiated {actual_w}x{actual_h}, "
            f"expected {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT} "
            f"(Section 2.2). Check v4l2-ctl --list-formats-ext."
        )

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Webcam did not return a frame.")

    capture_time_ms = (time.time() - t0) * 1000.0
    return frame, capture_time_ms
