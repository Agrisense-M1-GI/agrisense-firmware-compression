# algo/jpeg-4x2x2

**Role in the study (Section 4.3):** subsampling variant of Experience A (4:2:2) -- identical pipeline to `algo/jpeg-baseline`, only the chroma subsampling changes. Compared against `algo/jpeg-baseline` to isolate the effect of subsampling alone.

**Configuration:** standard JPEG, libjpeg IJG v9e, quality 75, default
quantization tables, 4:2:2 chroma subsampling (`JPEG_SAMPLE_FACTORS = "2x1,1x1,1x1"`).

## What's fixed by the protocol (do not change)

- Gate thresholds, VARI thresholds, capture resolution: `common/config.py`
- JPEG quality (75) and subsampling (4:2:2): `common/config.py`
  (`JPEG_QUALITY`, `JPEG_SAMPLE_FACTORS`)
- Pipeline order: capture -> compression -> energy measurement -> node
  metrics -> transmission (`pipeline.py` / `pipeline_test.py`) -- no gate,
  no segmentation on this branch (see `common/vari.py` docstring)

## What you need to do before running this branch

1. Copy your `jpeg-9e/` source tree into `compression/lib/`, then run
   `compression/lib/build.sh` (see `compression/lib/README.md`). This
   branch does not modify the library — quality/subsampling are passed as
   command-line flags to `cjpeg`.
2. Set `AGRISENSE_STATION_URL` (env var) or edit `STATION_UPLOAD_URL` in
   `common/config.py` — **CALIBRATE**, currently a placeholder IP.
3. Set `AGRISENSE_MODE_URL` for `startup.py` (lives at the node root, not
   in this branch — see `00-node-root-setup/`).
4. Install Python dependencies: `pip install -r requirements.txt`

## Files

```
README.md
pipeline.py           NORMAL mode: one live capture cycle
pipeline_test.py      TEST mode: processes the whole reference dataset
capture/
  camera.py           webcam capture (NORMAL mode only)
common/
  config.py           protocol-fixed constants (thresholds, paths, quality)
  gate.py             change-detection gate (3 criteria, majority vote)
  segmentation.py     Otsu segmentation on Y + morphological refine
  vari.py             VARI index + block classification (NOT used by this
                       branch — kept for consistency, see file docstring)
  metrics.py          node-side metrics (Section 6.1) + CSV logging
  energy_uart.py       UART protocol to the ESP8266 (Section 7.3)
  transmit.py         sends compressed file + metrics to the station (WiFi)
compression/
  encode.py           wraps this branch's cjpeg (subprocess)
  lib/                your private copy of jpeg-9e goes here (gitignored
                       binaries, tracked source) + build.sh
```

## Git workflow

```bash
cd ~/firmware
git checkout -b algo/jpeg-4x2x2
# copy the contents of this folder to the repo root
git add .
git commit -m "algo/jpeg-4x2x2: JPEG Q75 4:2:2"
```

To switch configurations later, you only ever do:
```bash
git checkout algo/<other-branch>
```
`startup.py` and the `DATASET/`, `logs/`, `results/` folders never move —
only `pipeline.py`, `pipeline_test.py`, `capture/`, `common/`, `compression/`
change.
