"""
transmit.py
===========
Sends the compressed file + image_id + node metrics to the station over
WiFi (Section 4.2, step 8). The energy measurement itself travels
separately, from the ESP8266 to the station over LoRa, and is appended
station-side by matching timestamp/image_id.

Kept intentionally simple (a single HTTP POST) — swap for whatever your
station's actual receiver expects (see station/metrics_receiver.py in the
top-level project delivery).
"""

import json

import requests

from . import config


def send_to_station(image_id: str, compressed_bytes: bytes, metrics_dict: dict) -> bool:
    """
    POSTs the compressed file as multipart/form-data, with the node
    metrics as a JSON field alongside it. Returns True on success, False
    otherwise (never raises — a failed transmission should not crash the
    pipeline in the field).
    """
    try:
        files = {"file": (f"{image_id}.bin", compressed_bytes)}
        data = {"image_id": image_id, "metrics": json.dumps(metrics_dict)}
        resp = requests.post(config.STATION_UPLOAD_URL, files=files, data=data, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        print(f"[transmit] WARNING: could not reach station ({exc}). "
              f"Metrics were still logged locally.")
        return False
