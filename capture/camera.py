"""
camera.py
=========
Live capture from the USB webcam, used only in NORMAL mode (pipeline.py).
TEST mode (pipeline_test.py) reads from the fixed reference dataset instead
and does not use this module.

Captures at the webcam's maximum resolution (Section 2.2: 1920x1080 YUYV)
and returns a BGR numpy array (OpenCV convention) ready for the rest of
the pipeline.
"""

import time

import cv2

from common import config


def capture_frame():
    """
    Opens the webcam, grabs one frame, releases it.
    Returns (frame_bgr, capture_time_ms).
    """
    t0 = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam (index 0).")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Webcam did not return a frame.")

    capture_time_ms = (time.time() - t0) * 1000.0
    return frame, capture_time_ms
