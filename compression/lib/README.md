# compression/lib/ — private copy of jpeg-9e for this branch

Per your choice: **each branch gets its own copy** of the IJG libjpeg v9e
source, built and modified independently of every other branch.

## Setup (one time per branch)

1. Copy your `jpeg-9e/` source folder into this directory, so you end up
   with:
   ```
   compression/lib/jpeg-9e/   (the full IJG source tree you already have)
   compression/lib/build.sh   (this script)
   ```
2. Run:
   ```
   ./build.sh
   ```
   This configures and builds `cjpeg` and `djpeg` inside
   `compression/lib/jpeg-9e/`. `compression/encode.py` calls these two
   binaries directly by path (see `common/config.py`:
   `CJPEG_BIN` / `DJPEG_BIN`).

For `algo/jpeg-baseline`, **the library is used unmodified** — quality 75,
default IJG quantization tables, subsampling forced to 4:2:0 via the
`-sample 2x2,1x1,1x1` command-line flag. Nothing to patch in the C source
for this branch.

## Why cjpeg/djpeg as binaries and not a Python binding?

cjpeg exposes a `-sample H0xV0,H1xV1,H2xV2` flag that sets the sampling
factor **per component** (Y, Cb, Cr independently) at the command line —
that alone covers 4:2:0, 4:2:2, 4:4:0, 4:4:4, and the asymmetric 4:1:4
scheme used elsewhere in this study, with no C changes needed. Calling the
compiled binary from Python via `subprocess` is enough for this branch.

Branches that need finer control (per-block quantization table switching,
or replacing Huffman with rANS) modify the C source directly in their own
copy — see the `README.md` of those branches.
