"""
metrics.py
==========
Node-side metrics (Section 6.1). Collected for every image, regardless of
branch or mode (NORMAL/TEST), and appended to results/resultats_<branch>.csv.

Quality metrics (PSNR/SSIM) and agronomic metrics (Section 6.2, 6.3) are
computed on the STATION side after decoding, not here.
"""

import csv
import os
import time
from dataclasses import dataclass, asdict, field

from . import config


@dataclass
class NodeMetrics:
    timestamp: float = field(default_factory=time.time)
    algorithm: str = config.BRANCH_NAME
    image_id: str = ""
    mode: str = "NORMAL"          # "NORMAL" or "TEST"

    input_size_bytes: int = 0
    output_size_bytes: int = 0
    compression_ratio: float = 0.0

    gate_decision: bool = True
    gate_d_hist: float = 0.0
    gate_d_mean: float = 0.0
    gate_p_blocks: float = 0.0

    capture_time_ms: float = 0.0
    compression_time_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0

    cpu_freq_hz: int = 0
    cpu_temp_c: float = 0.0
    ram_used_mb: float = 0.0

    # Filled in once the ESP8266 energy reading has been appended
    # station-side (LoRa channel) by timestamp/image_id matching.
    # Kept here at 0 on the node; the authoritative value lives in the
    # station's merged results file.
    energy_mj: float = 0.0


def finalize(m: NodeMetrics) -> NodeMetrics:
    """Fill compression_ratio from input/output sizes if not already set."""
    if m.input_size_bytes > 0:
        m.compression_ratio = m.input_size_bytes / max(m.output_size_bytes, 1)
    return m


def append_csv(m: NodeMetrics, path: str = config.RESULTS_CSV) -> None:
    """Append one row to the branch's results CSV, writing the header once."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = asdict(m)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
