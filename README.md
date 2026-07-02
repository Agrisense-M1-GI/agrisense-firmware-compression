# WZ-OSEG

## Principle

WZ-OSEG is a node-side image pipeline for wireless multimedia sensor
nodes that combines two ideas:

1. **Segmentation (Otsu + HSV hue mask).** The node computes a global
   Otsu threshold on the luminance channel and a hue-range mask (HSV)
   over the RGB image. Both masks are transmitted as compact,
   sub-sampled side channels alongside the compressed image, so the
   sink can locate vegetation regions without re-running segmentation
   on a lossy-reconstructed image.

2. **Distributed-coding-inspired luminance compression.** The
   luminance channel is split into 8x8 blocks; a DCT is applied to
   each block and only the 21 lowest-frequency coefficients are kept
   and uniformly quantized. Rather than transmitting these
   coefficients raw, they are **entropy-coded**: zero-runs are
   run-length encoded, then the resulting token stream is compressed
   with a canonical Huffman code built on the fly for each image. At
   the sink, blocks are reconstructed via inverse DCT and blended with
   a correlated **side-information** image (60% reconstructed block /
   40% side info), in the spirit of Wyner-Ziv distributed source
   coding, which shifts reconstruction complexity to the receiver.

**Important scope note:** this implementation covers the luminance
compression + segmentation-mask transmission stage only. It does not
include LDPC syndrome coding or an explicit rate-distortion-optimal
Slepian-Wolf decoder — the side-information blending described above
is the mechanism actually implemented and tested. The chrominance
channels are not transmitted; reconstruction replicates luminance into
R=G=B, which caps color-fidelity metrics (PSNR/SSIM against a full-color
original) — this is a known, documented limitation, not a bug.

**Side information caveat.** Wyner-Ziv decoding is only meaningful with
side information correlated to the source. When testing on a static,
non-sequential image dataset, the test pipeline uses the original image
itself as side info — this yields an *upper bound* on reconstruction
quality, not a realistic deployment measurement. See the top-level
project README for the full discussion.

## Prerequisites to re-run the test after cloning this branch

1. A C/C++ toolchain to compile the node-side encoder.
2. Python 3 with the packages below, on both the node (Pi) and the
   server (laptop) sides.
3. A dataset of RGB images accessible to the node.
4. The server process reachable over the network from the node
   (same LAN, correct IP, port 8000 open).

## Dependencies

**On the node (Raspberry Pi or equivalent):**
```bash
sudo apt update && sudo apt install -y build-essential
python3 -m pip install --break-system-packages pillow requests
```
Compile the encoder:
```bash
gcc -O2 -Wall -o wz-oseg/encode wz-oseg/encode.cpp -lm
```

**On the server (laptop):**
```bash
python3 -m pip install --break-system-packages fastapi uvicorn python-multipart pillow numpy scikit-image
```
The decoder (`wz-oseg/decode.py` / `server/decoders/wz_oseg_decode.py`)
additionally needs `scipy` (for the inverse DCT):
```bash
python3 -m pip install --break-system-packages scipy
```

## Running the test

```bash
# server (laptop) -- keep running
cd server && python3 main.py

# node (Pi)
python3 pipeline_test.py --dataset <path-to-images> \
    --server http://<laptop-ip>:8000 --node-id <any-id>
```
Results are appended to `server/data/results.csv` on the server, one
row per image with `algorithm = WZ-OSEG`.
