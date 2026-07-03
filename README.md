# reference/jpeg

## Principle

This branch produces the JPEG and JPEG2000-ROI reference measurements
needed to complete Table 4 of the AgriJPEG article (`JPEG` and
`JPEG 2000 ROI` rows), but measured directly on the target embedded
hardware (Raspberry Pi Zero 2W) rather than on the desktop machine used
for the original 294-image run described in the paper (Section 5.2 /
5.3, "Software-only prototype" limitation).

Two methods are implemented:

1. **JPEG** -- standard JPEG encoding via Pillow (its bundled
   libjpeg), quality 75. **Note the caveat explicitly:** this is
   Pillow's own libjpeg build, not the IJG libjpeg-9e source that was
   separately downloaded and compiled (`cjpeg`/`djpeg`, see
   `tree.txt` / the project context doc). It was kept as-is on request
   rather than switched to the compiled `cjpeg`/`djpeg` binaries. If a
   result closer to "the exact library I downloaded" is needed later,
   swap `jpeg_test.py`'s encoder call for a subprocess call to
   `cjpeg`/`djpeg` -- the rest of the pipeline (upload, decode,
   metrics) does not need to change.

2. **JPEG2000-ROI-2stream** -- a real, working JPEG2000 encoder
   (OpenJPEG, via `opj_compress`/`opj_decompress`), with genuine
   region-differentiated quality allocation. **Important limitation,
   stated plainly:** OpenJPEG does not implement the JPEG2000
   standard's native spatial Region-of-Interest feature (Annex H,
   "Maxshift"). Its own `-ROI` flag only up-shifts an entire colour
   *component*, not an arbitrary spatial mask -- this is documented by
   the OpenJPEG project itself and confirmed on their mailing list: no
   spatial Annex H implementation has ever been merged. So instead of
   claiming something the library cannot do, this branch implements
   a **two-stream workaround**, which is a real and reproducible use
   of JPEG2000, just not the codestream-native ROI marker:
   - A block-level ROI mask is computed with the *same* Otsu
     convention already used by `algo/adres` and `algo/wz-oseg`
     (global Otsu on luminance, 16×16 blocks, block flagged ROI if
     >25% of its pixels are above threshold) -- kept identical across
     branches so the four methods stay comparable.
   - A full-resolution "ROI stream" is encoded with `opj_compress`,
     after flattening every non-ROI block to a neutral grey (this
     lets OpenJPEG's rate control spend its bit budget on ROI content,
     since flattened regions contribute near-zero entropy).
   - A bicubic-downsampled "background stream" is encoded separately,
     at a coarser target ratio.
   - Both streams + the block mask are packed into one container file
     (`compressed.jp2roi`, format `AGRIJ2K_ROI`, see
     `jpeg2000_roi_test.py` header for the exact layout).
   - The sink decodes both real JP2 streams with `opj_decompress` and
     composites: ROI pixels from the ROI stream, everything else from
     the up-sampled background stream.

   In `results.csv` this method is logged as **`JPEG2000-ROI-2stream`**,
   deliberately not just `JPEG2000-ROI` or `JPEG2000`, so nobody
   downstream mistakes it for a measurement of the standard's native
   ROI feature.

## Architecture (same client/server split as algo/adres, algo/wz-oseg)

```
Pi (node)                              Laptop (server, main.py)
----------                             -------------------------
pipeline_test.py
  for each image in dataset:
    -> jpeg_test.py            ---upload--->  /test/submit/reference/{node}/{image}
    -> jpeg2000_roi_test.py    ---upload--->      decodes both methods,
                                                    computes PSNR/SSIM
                                                    (+ IoU/Dice for
                                                    JPEG2000-ROI-2stream),
                                                    appends 2 rows to
                                                    data/results.csv
```

The original PPM is uploaded alongside both compressed artifacts so
the server can compute quality metrics against ground truth, exactly
like the ADRES/WZ-OSEG branches. As with those branches, the "mask
IoU/Dice" reference is a server-side Otsu recomputation on the
uploaded original -- it measures **mask transmission fidelity**, not
accuracy against a hand-annotated ground truth (none exists for this
dataset).

## Prerequisites

**On the node (Raspberry Pi):**
```bash
sudo apt update && sudo apt install -y libopenjp2-tools
python3 -m pip install --break-system-packages pillow requests numpy
```
Verify: `opj_compress -h` should print usage, not "command not found".

**On the server (laptop):**
```bash
sudo apt install -y libopenjp2-tools   # opj_decompress needed here too
python3 -m pip install --break-system-packages fastapi uvicorn python-multipart pillow numpy scikit-image
```

## Running the test

```bash
# server (laptop) -- keep running
cd server && python3 main.py

# node (Pi)
python3 pipeline_test.py --dataset <path-to-640x480-ppm-dataset> \
    --server http://<laptop-ip>:8000 --node-id <any-id>
```
Results are appended to `server/data/results.csv`, two rows per image
(`algorithm = JPEG` and `algorithm = JPEG2000-ROI-2stream`).

## Known limitations of this branch (stated directly, not hidden)

- JPEG uses Pillow's bundled libjpeg, not the separately compiled
  IJG libjpeg-9e binaries (see point 1 above) -- a deliberate choice
  for now, easy to change later if needed.
- JPEG2000-ROI-2stream is not a measurement of JPEG2000 Annex H ROI;
  it is a documented two-stream workaround using real JPEG2000
  encoding. Any comparison against literature figures that assume
  native ROI encoding should account for this difference.
- The flattening-based bit allocation trick has no formal
  rate-distortion guarantee (unlike true Maxshift coefficient
  up-shifting); its effectiveness was verified empirically on this
  branch's own test images, not derived analytically.
