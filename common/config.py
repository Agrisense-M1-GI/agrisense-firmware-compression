"""
config.py
=========
Fixed constants from the test protocol. Do NOT tune these to "improve"
results — they are protocol-fixed (Section 4.2) and identical across every
branch, except the few explicitly marked CALIBRATE below.

This file is duplicated identically in every branch on purpose (see project
README): each branch must be fully self-contained and readable on its own.
"""

import os

# --- Filesystem layout (Section 3.1) -----------------------------------
# These live OUTSIDE the git repo, at the root of the node. They are the
# same regardless of which branch is currently checked out.
NODE_ROOT = os.path.expanduser("~")
DATASET_DIR = os.path.join(NODE_ROOT, "DATASET")
LOGS_DIR = os.path.join(NODE_ROOT, "logs")
RESULTS_DIR = os.path.join(NODE_ROOT, "results")

# Identifies this branch/configuration in logs and result filenames.
BRANCH_NAME = "algo/jpeg-4x4x0"
RESULTS_CSV = os.path.join(RESULTS_DIR, "resultats_jpeg-4x4x0.csv")

# --- Change-detection gate thresholds (Section 4.2, step 2) ------------
# Ported from the real "garde_fou_energetique.py" v2 implementation:
# per-channel histograms compared with Bhattacharyya distance, per-channel
# MEDIAN deviation (robust to a localized reflection/shadow), and blocks
# compared the same way as the global histogram.
#
# CALIBRATE: these are NOT the protocol's original values (0.38 / 6.0 / 18.0)
# — those were calibrated for the old v1 metric scale (single merged
# histogram + L1 distance + mean). The v2 metrics below (Bhattacharyya +
# median) live on a different scale entirely. The values here are the
# current best working guess (from garde_fou_energetique.py's main()),
# NOT a properly calibrated result — re-run calibrate_seuils.py on your
# dataset before trusting these for Campaign 2.
GATE_TAU_HIST = 0.08     # tau_H : per-channel histogram Bhattacharyya distance
GATE_TAU_MEAN = 2.2      # tau_C : per-channel median deviation
GATE_TAU_BLOCKS = 6.0    # tau_B : proportion of modified blocks, in PERCENT (0-100)
GATE_MIN_VOTES = 2       # majority vote: transmit if >= 2 of 3 criteria fire

# Internal threshold on a single block's Bhattacharyya distance, used to
# decide whether that block itself counts as "changed" before computing
# the tau_B proportion above (this is the protocol's delta = 0.50).
# CALIBRATE alongside the three thresholds above.
GATE_BLOCK_CHANGE_THRESHOLD = 0.50

GATE_BLOCK_SIZE = 32     # pixel size of the blocks used for criterion 3
GATE_HIST_BINS = 4       # histogram bins PER CHANNEL (so 3 x this, total)

# Whether the gate is allowed to block transmission in NORMAL mode for THIS
# branch. True for every branch except agrijpeg-core and jpeg-qveg-qbg-4x1x4
# (Section 4.3: "transmission systematique" for those two).
GATE_BLOCKS_TRANSMISSION = True

# --- VARI vegetation classification (Section 4.2, step 4) --------------
VARI_MEAN_THRESHOLD = 0.1     # block is "vegetation" if VARI_mean > 0.1
VARI_VEG_FRACTION = 0.50      # and vegetation-pixel fraction > 50%

# --- Composite ROI mask (Otsu block-conversion + VARI, combined by OR) --
OTSU_ROI_BLOCK_FRACTION = 0.25  # block counts as ROI if >=25% of its pixels are Otsu-class 255

# --- Capture format (Section 2.2) ---------------------------------------
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# --- Energy measurement UART link to the ESP8266 (Section 7.3) ---------
UART_PORT = "/dev/ttyAMA0"
UART_BAUDRATE = 115200
ENERGY_START_CMD = "ENERGY:START"
ENERGY_STOP_CMD = "ENERGY:STOP"
SHUTDOWN_READY_CMD = "SHUTDOWN_READY"  # sent to ESP8266 at end of NORMAL cycle

# --- NORMAL-mode state (Section 3.2) ------------------------------------
# The gate needs a "reference" image to compare against. In NORMAL mode
# each run is a fresh process (one cycle, then shutdown), so the previous
# capture is cached to disk between runs. This lives at the node root,
# outside the git repo, so it survives across branch switches too.
LAST_CAPTURE_PATH = os.path.join(NODE_ROOT, "last_capture.png")

# --- Station link (Section 4.2, step 8) ---------------------------------
# CALIBRATE: replace with the station's real address on your network.
STATION_UPLOAD_URL = os.environ.get(
    "AGRISENSE_STATION_URL", "http://192.168.1.50:8000/upload"
)

# --- Branch-specific compression parameters (Section 4.3) --------------
# algo/jpeg-4x4x0: identical to algo/jpeg-baseline except for chroma
# subsampling -- JPEG standard libjpeg IJG v9e, quality Q75, 4:4:0.
JPEG_QUALITY = 75
JPEG_SAMPLE_FACTORS = "1x2,1x1,1x1"  # cjpeg -sample argument -> 4:4:0

# Path to the compiled cjpeg/djpeg binaries for THIS branch's private copy
# of jpeg-9e (Section: "une copie de la bibliotheque par branche").
# Build it once with: compression/lib/build.sh
JPEG_LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "compression", "lib")
CJPEG_BIN = os.path.join(JPEG_LIB_DIR, "jpeg-9e", "cjpeg")
DJPEG_BIN = os.path.join(JPEG_LIB_DIR, "jpeg-9e", "djpeg")
