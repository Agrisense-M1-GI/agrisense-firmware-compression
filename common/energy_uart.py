"""
energy_uart.py
==============
UART protocol to the ESP8266 for per-image energy measurement
(Section 7.3). The Raspberry Pi only sends START/STOP commands with the
image_id and timestamp; the ESP8266 does the actual INA219 sampling and
sends the computed energy_mj to the STATION via LoRa (not back to the Pi).

If no ESP8266 is wired (e.g. developing on a laptop), this module degrades
gracefully: it logs a warning instead of crashing the pipeline, so you can
still run/debug the rest of the pipeline without the hardware attached.
"""

import time

from . import config

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


class EnergyMeter:
    def __init__(self, port: str = config.UART_PORT, baudrate: int = config.UART_BAUDRATE):
        self._conn = None
        if _SERIAL_AVAILABLE:
            try:
                self._conn = serial.Serial(port, baudrate, timeout=1)
            except Exception as exc:
                print(f"[energy_uart] WARNING: could not open {port} ({exc}). "
                      f"Continuing without energy measurement.")
        else:
            print("[energy_uart] WARNING: pyserial not installed. "
                  "Continuing without energy measurement.")

    def start(self, image_id: str) -> None:
        self._send(f"{config.ENERGY_START_CMD} {image_id} {time.time()}")

    def stop(self) -> None:
        self._send(config.ENERGY_STOP_CMD)

    def _send(self, message: str) -> None:
        if self._conn is None:
            return
        try:
            self._conn.write((message + "\n").encode("utf-8"))
        except Exception as exc:
            print(f"[energy_uart] WARNING: failed to send '{message}' ({exc})")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
