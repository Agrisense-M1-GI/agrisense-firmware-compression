# ADRES

## Principle

ADRES (region-aware adaptive compression) is a node-side image pipeline
designed for scenarios where each frame must be processed
independently (no reliance on temporal correlation between frames),
trading that independence for a simpler, deterministic-latency
encoder.

1. **Region-of-interest detection.** The node computes a global Otsu
   threshold on the luminance channel, then evaluates each 16x16 block:
   a block is flagged as ROI if more than 25% of its pixels are above
   the Otsu threshold (bright / vegetation-like), matching the same
   foreground convention used elsewhere (Otsu: `gray > threshold` =
   foreground).

2. **Independent quantization of ROI and background.** ROI pixels are
   quantized with a fine step (`q_roi`); background pixels are
   point-sampled at a coarser stride (`subsample`) and quantized with a
   coarser step (`q_bg`). Two operating profiles are available:
   - **Quality (Q):** finer quantization, larger transmitted payload.
   - **Economy (E):** coarser quantization/sub-sampling, smaller payload.

3. **Lossless entropy coding.** Rather than transmitting raw quantized
   bytes with per-pixel position tags, two images are built and each is
   compressed independently with **PNG lossless compression**
   (libpng, level 9): a full-resolution canvas holding the quantized
   ROI pixels (zero elsewhere — highly compressible under PNG's DEFLATE
   stage) and the sub-sampled background. At the sink, the background
   is bicubic-upsampled and the ROI canvas is overlaid using the
   transmitted block-level ROI mask.

**Scope note:** no per-pixel position metadata is transmitted; pixel
placement is entirely reconstructed from the compact block-level ROI
mask, which is what allows ADRES to avoid the storage overhead of
earlier prototype iterations.

## Prerequisites to re-run the test after cloning this branch

1. A C/C++ toolchain **and libpng development headers** to compile the
   node-side encoder (PNG compression is done in C via libpng, not
   deferred to Python).
2. Python 3 with the packages below, on both the node (Pi) and the
   server (laptop) sides.
3. A dataset of RGB images accessible to the node.
4. The server process reachable over the network from the node
   (same LAN, correct IP, port 8000 open).

## Dependencies

**On the node (Raspberry Pi or equivalent):**
```bash
sudo apt update && sudo apt install -y build-essential libpng-dev
python3 -m pip install --break-system-packages pillow requests
```
Compile the encoder:
```bash
gcc -O2 -Wall -o adres/encode adres/encode.cpp -lpng -lz -lm
```

**On the server (laptop):**
```bash
python3 -m pip install --break-system-packages fastapi uvicorn python-multipart pillow numpy scikit-image
```
No extra native library is needed server-side: the Python decoder reads
the PNG streams directly with Pillow (which links its own libpng), it
does not call libpng itself.

## Running the test

```bash
# server (laptop) -- keep running
cd server && python3 main.py

# node (Pi)
python3 pipeline_test.py --dataset <path-to-images> \
    --server http://<laptop-ip>:8000 --node-id <any-id>
```
Results are appended to `server/data/results.csv` on the server, two
rows per image (`algorithm = ADRES`, `profile = Q` and `profile = E`).
