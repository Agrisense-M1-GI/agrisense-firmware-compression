/*
 * roi_jpeg_codec.c
 *
 * AgriJPEG Qveg/Qbg calibration codec (branch: calibration/qveg-qbg-search)
 *
 * Implements "Option B" agreed with the user:
 *   - Split an image into 8x8 blocks, classify each block veg/bg using a
 *     ground-truth block map (computed separately in Python from the
 *     pixel-level mask, 25% rule).
 *   - Build two COMPACT images (one per class) by packing only the pixels
 *     of blocks belonging to that class, in raster order (left-to-right,
 *     top-to-bottom over the classified blocks), padding only the final
 *     row of the compact rectangle with a neutral value (128) if needed.
 *   - Encode each compact image with libjpeg's standard high-level API
 *     (jpeg_write_scanlines), each with its OWN quantization table
 *     (Q_veg or Q_bg), injected via jpeg_add_quant_table.
 *   - Decode: standard libjpeg decode of each stream separately, then
 *     un-pack blocks back to their original grid position using the same
 *     block map and the same raster fill order used at encode time.
 *
 * Scope decision (documented, not hidden): the custom quantization table
 * is applied to the LUMINANCE (Y) component only. Chrominance (Cb, Cr)
 * uses the standard JPEG chrominance table. This matches the scope of
 * the Li/De Sa/Sampson paper (single-channel Q-table study) and is called
 * out explicitly in the README as an assumption, not a silent shortcut.
 *
 * This program only uses the PUBLIC libjpeg API (jpeglib.h). No internal
 * files (jcdctmgr.c/jddctmgr.c) are touched or patched.
 *
 * Build: see Makefile. Requires libjpeg (turbo or IJG) development headers.
 *
 * Usage:
 *   roi_jpeg_codec encode <in.ppm> <blockmap.bin> <bw> <bh> \
 *                  <qveg.txt> <qbg.txt> <out_veg.jpg> <out_bg.jpg>
 *
 *   roi_jpeg_codec decode <veg.jpg> <bg.jpg> <blockmap.bin> <bw> <bh> \
 *                  <orig_w> <orig_h> <out.ppm>
 *
 * orig_w/orig_h: original (pre-padding) image dimensions, needed to crop
 * the reassembled block-multiple-sized image back to its true size
 * (mirrors the edge-replication padding done at encode time).
 *
 * Image format: binary PPM (P6), 8-bit per channel, RGB, dimensions that
 * are NOT required to be a multiple of 8 (edge blocks are handled by
 * padding with edge-replication before splitting, and cropped back after
 * reassembly -- this matches what libjpeg itself does internally for
 * non-multiple-of-8 images).
 *
 * blockmap.bin: raw bytes, bw*bh entries, row-major, each byte is
 * 1 = vegetation block, 0 = background block. Produced by the Python
 * side (block_map.py) from the ground-truth mask using the 25% rule.
 * This file is a PROTOCOL CONSTANT format (not a placeholder): fixed,
 * documented, simple, reproducible.
 *
 * qveg.txt / qbg.txt: 64 whitespace-separated integers in [1,255],
 * natural (row-major) 8x8 order (NOT zig-zag) -- the Python side is
 * responsible for producing the table in this order (it internally
 * generates candidates in zig-zag order per the paper's method, then
 * converts to row-major before writing the file; see qtable_search.py).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <setjmp.h>

#define BLOCK 8

/* ---------- error handling ---------- */

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

static void my_error_exit(j_common_ptr cinfo) {
  struct my_error_mgr *myerr = (struct my_error_mgr *) cinfo->err;
  (*cinfo->err->output_message)(cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

/* ---------- PPM I/O ---------- */

typedef struct {
  int width, height;
  unsigned char *data; /* RGB, width*height*3 */
} Image;

static void ppm_skip_ws_and_comments(FILE *f) {
  int c;
  for (;;) {
    while ((c = fgetc(f)) != EOF && (c==' '||c=='\t'||c=='\n'||c=='\r')) ;
    if (c == '#') {
      while ((c = fgetc(f)) != EOF && c != '\n') ;
      continue; /* there may be more whitespace/comments after this line */
    }
    if (c != EOF) ungetc(c, f);
    return;
  }
}

static Image *ppm_read(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
    fprintf(stderr, "%s: not a binary PPM (P6)\n", path); exit(1);
  }
  int w, h, maxval;
  ppm_skip_ws_and_comments(f);
  if (fscanf(f, "%d", &w) != 1) { fprintf(stderr,"bad ppm header\n"); exit(1);}
  ppm_skip_ws_and_comments(f);
  if (fscanf(f, "%d", &h) != 1) { fprintf(stderr,"bad ppm header\n"); exit(1);}
  ppm_skip_ws_and_comments(f);
  if (fscanf(f, "%d", &maxval) != 1) { fprintf(stderr,"bad ppm header\n"); exit(1);}
  fgetc(f); /* single whitespace after maxval */
  if (maxval != 255) { fprintf(stderr, "%s: only maxval=255 supported\n", path); exit(1); }

  Image *img = malloc(sizeof(Image));
  img->width = w; img->height = h;
  img->data = malloc((size_t)w*h*3);
  size_t got = fread(img->data, 1, (size_t)w*h*3, f);
  if (got != (size_t)w*h*3) { fprintf(stderr, "%s: truncated pixel data\n", path); exit(1); }
  fclose(f);
  return img;
}

static void ppm_write(const char *path, Image *img) {
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
  fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
  fwrite(img->data, 1, (size_t)img->width*img->height*3, f);
  fclose(f);
}

static void image_free(Image *img) { free(img->data); free(img); }

/* ---------- quant table file I/O ---------- */

static void read_qtable(const char *path, unsigned int q[64]) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open qtable %s\n", path); exit(1); }
  for (int i = 0; i < 64; i++) {
    if (fscanf(f, "%u", &q[i]) != 1) {
      fprintf(stderr, "%s: expected 64 integers, failed at index %d\n", path, i);
      exit(1);
    }
    if (q[i] < 1 || q[i] > 255) {
      fprintf(stderr, "%s: value at index %d out of [1,255]: %u\n", path, i, q[i]);
      exit(1);
    }
  }
  fclose(f);
}

/* ---------- block map I/O ---------- */

static unsigned char *read_blockmap(const char *path, int bw, int bh) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open blockmap %s\n", path); exit(1); }
  size_t n = (size_t)bw*bh;
  unsigned char *map = malloc(n);
  size_t got = fread(map, 1, n, f);
  if (got != n) { fprintf(stderr, "blockmap %s: expected %zu bytes, got %zu\n", path, n, got); exit(1); }
  fclose(f);
  return map;
}

/* ---------- padding to multiple of BLOCK (edge replication) ---------- */

static Image *pad_to_block_multiple(Image *src, int *out_bw, int *out_bh) {
  int bw = (src->width  + BLOCK - 1) / BLOCK;
  int bh = (src->height + BLOCK - 1) / BLOCK;
  int pw = bw * BLOCK, ph = bh * BLOCK;
  Image *out = malloc(sizeof(Image));
  out->width = pw; out->height = ph;
  out->data = malloc((size_t)pw*ph*3);
  for (int y = 0; y < ph; y++) {
    int sy = y < src->height ? y : src->height - 1;
    for (int x = 0; x < pw; x++) {
      int sx = x < src->width ? x : src->width - 1;
      memcpy(&out->data[((size_t)y*pw+x)*3], &src->data[((size_t)sy*src->width+sx)*3], 3);
    }
  }
  *out_bw = bw; *out_bh = bh;
  return out;
}

/* ---------- packing: extract classified blocks into a compact raster image ---------- */

/* count blocks of the given class (1=veg, 0=bg) */
static int count_class(unsigned char *map, int bw, int bh, int cls) {
  int n = 0;
  for (int i = 0; i < bw*bh; i++) if (map[i] == cls) n++;
  return n;
}

/*
 * Build compact image: width = padded->width (same block width as source,
 * per agreed design), height = ceil(n_blocks / bw) * BLOCK. Blocks are
 * filled in raster order (row-major over the ORIGINAL block grid,
 * skipping blocks not of this class) into the compact grid, also in
 * raster order. Leftover cells in the last row are filled with 128.
 */
static Image *pack_class(Image *padded, unsigned char *map, int bw, int bh, int cls) {
  int n = count_class(map, bw, bh, cls);
  int out_bw = bw; /* keep same block-width as source, per design decision */
  int out_bh = (n + out_bw - 1) / out_bw;
  if (out_bh == 0) out_bh = 1; /* guard: class may be empty on some images */

  Image *out = malloc(sizeof(Image));
  out->width = out_bw * BLOCK;
  out->height = out_bh * BLOCK;
  out->data = malloc((size_t)out->width*out->height*3);
  /* fill with neutral value 128 first (covers padding cells) */
  memset(out->data, 128, (size_t)out->width*out->height*3);

  int placed = 0;
  for (int by = 0; by < bh; by++) {
    for (int bx = 0; bx < bw; bx++) {
      if (map[by*bw+bx] != cls) continue;
      int out_bx = placed % out_bw;
      int out_by = placed / out_bw;
      /* copy 8x8 block from padded[by,bx] to out[out_by,out_bx] */
      for (int yy = 0; yy < BLOCK; yy++) {
        int sy = by*BLOCK+yy, dy = out_by*BLOCK+yy;
        memcpy(&out->data[((size_t)dy*out->width + out_bx*BLOCK)*3],
               &padded->data[((size_t)sy*padded->width + bx*BLOCK)*3],
               BLOCK*3);
      }
      placed++;
    }
  }
  return out;
}

/* Inverse of pack_class: given a decoded compact image, scatter its
 * blocks back to their original position in an (already-allocated,
 * block-multiple-sized) output image, using the same raster fill order. */
static void unpack_class(Image *compact, unsigned char *map, int bw, int bh, int cls, Image *out) {
  int out_bw = compact->width / BLOCK;
  int placed = 0;
  for (int by = 0; by < bh; by++) {
    for (int bx = 0; bx < bw; bx++) {
      if (map[by*bw+bx] != cls) continue;
      int c_bx = placed % out_bw;
      int c_by = placed / out_bw;
      for (int yy = 0; yy < BLOCK; yy++) {
        int dy = by*BLOCK+yy, sy = c_by*BLOCK+yy;
        memcpy(&out->data[((size_t)dy*out->width + bx*BLOCK)*3],
               &compact->data[((size_t)sy*compact->width + c_bx*BLOCK)*3],
               BLOCK*3);
      }
      placed++;
    }
  }
}

/* ---------- JPEG encode/decode (standard high-level API) ---------- */

/* natural (row-major) 8x8 -> libjpeg's own natural order is also
 * row-major for jpeg_add_quant_table's basic_table argument, so no
 * reordering is needed here; qtable files are already row-major. */

static void jpeg_encode_custom_q(Image *img, unsigned int qtable[64], const char *out_path,
                                  const int *samp_h, const int *samp_v) {
  struct jpeg_compress_struct cinfo;
  struct my_error_mgr jerr;
  FILE *outfile = fopen(out_path, "wb");
  if (!outfile) { fprintf(stderr, "cannot write %s\n", out_path); exit(1); }

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    exit(1);
  }
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = img->width;
  cinfo.image_height = img->height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);

  /* Custom table on slot 0 (luminance), forced non-baseline-safe scaling
   * off (we pass literal values, scale_factor=100 means "use as-is"). */
  jpeg_add_quant_table(&cinfo, 0, (const unsigned int *)qtable, 100, TRUE);
  /* Leave slot 1 (chrominance) at whatever jpeg_set_defaults put there
   * (standard JPEG chrominance table) -- documented scope decision:
   * custom Q applies to luminance only. */
  cinfo.comp_info[0].quant_tbl_no = 0; /* Y  -> custom table */
  cinfo.comp_info[1].quant_tbl_no = 1; /* Cb -> standard table */
  cinfo.comp_info[2].quant_tbl_no = 1; /* Cr -> standard table */

  /* jpeg_set_colorspace() (re)sets default sampling factors for the new
   * colorspace, so any --sample override must be applied AFTER this
   * call, not before -- otherwise it gets silently overwritten. */
  jpeg_set_colorspace(&cinfo, JCS_YCbCr);
  if (samp_h != NULL && samp_v != NULL) {
    for (int c = 0; c < 3; c++) {
      cinfo.comp_info[c].h_samp_factor = samp_h[c];
      cinfo.comp_info[c].v_samp_factor = samp_v[c];
    }
  }
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW row_pointer[1];
  int row_stride = img->width * 3;
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &img->data[(size_t)cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
}

/* Parses "H1xV1,H2xV2,H3xV3" (same convention as cjpeg's -sample), e.g.
 * "4x1,1x1,4x1" for 4:1:4. Returns 1 on success, 0 on malformed input. */
static int parse_sample_factors(const char *spec, int h[3], int v[3]) {
  int n = sscanf(spec, "%dx%d,%dx%d,%dx%d",
                 &h[0], &v[0], &h[1], &v[1], &h[2], &v[2]);
  return n == 6;
}

static Image *jpeg_decode(const char *in_path) {
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
  FILE *infile = fopen(in_path, "rb");
  if (!infile) { fprintf(stderr, "cannot open %s\n", in_path); exit(1); }

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    exit(1);
  }
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);

  Image *img = malloc(sizeof(Image));
  img->width = cinfo.output_width;
  img->height = cinfo.output_height;
  img->data = malloc((size_t)img->width*img->height*3);

  int row_stride = img->width * 3;
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row_pointer[1] = { &img->data[(size_t)cinfo.output_scanline*row_stride] };
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
  }
  jpeg_finish_decompress(&cinfo);
  fclose(infile);
  jpeg_destroy_decompress(&cinfo);
  return img;
}

/* ---------- main ---------- */

static void usage(const char *prog) {
  fprintf(stderr,
    "Usage:\n"
    "  %s encode <in.ppm> <blockmap.bin> <bw> <bh> <qveg.txt> <qbg.txt> <out_veg.jpg> <out_bg.jpg> [--sample H1xV1,H2xV2,H3xV3]\n"
    "  %s decode <veg.jpg> <bg.jpg> <blockmap.bin> <bw> <bh> <orig_w> <orig_h> <out.ppm>\n",
    prog, prog);
}

static void crop_image(Image *img, int w, int h, Image *out) {
  out->width = w; out->height = h;
  out->data = malloc((size_t)w*h*3);
  for (int y = 0; y < h; y++)
    memcpy(&out->data[(size_t)y*w*3], &img->data[(size_t)y*img->width*3], (size_t)w*3);
}

int main(int argc, char **argv) {
  if (argc < 2) { usage(argv[0]); return 1; }

  if (strcmp(argv[1], "encode") == 0) {
    if (argc != 10 && argc != 12) { usage(argv[0]); return 1; }
    const char *in_path = argv[2];
    const char *blockmap_path = argv[3];
    int bw = atoi(argv[4]), bh = atoi(argv[5]);
    const char *qveg_path = argv[6], *qbg_path = argv[7];
    const char *out_veg = argv[8], *out_bg = argv[9];

    int samp_h[3], samp_v[3];
    int has_sample = 0;
    if (argc == 12) {
      if (strcmp(argv[10], "--sample") != 0) { usage(argv[0]); return 1; }
      if (!parse_sample_factors(argv[11], samp_h, samp_v)) {
        fprintf(stderr, "malformed --sample value: %s (expected H1xV1,H2xV2,H3xV3)\n", argv[11]);
        return 1;
      }
      has_sample = 1;
    }

    Image *src = ppm_read(in_path);
    int actual_bw, actual_bh;
    Image *padded = pad_to_block_multiple(src, &actual_bw, &actual_bh);
    if (actual_bw != bw || actual_bh != bh) {
      fprintf(stderr,
        "block grid mismatch: image implies %dx%d blocks, blockmap says %dx%d "
        "(check that blockmap.bin was generated for THIS image)\n",
        actual_bw, actual_bh, bw, bh);
      return 1;
    }
    unsigned char *map = read_blockmap(blockmap_path, bw, bh);
    unsigned int qveg[64], qbg[64];
    read_qtable(qveg_path, qveg);
    read_qtable(qbg_path, qbg);

    Image *veg_img = pack_class(padded, map, bw, bh, 1);
    Image *bg_img  = pack_class(padded, map, bw, bh, 0);

    jpeg_encode_custom_q(veg_img, qveg, out_veg, has_sample ? samp_h : NULL, has_sample ? samp_v : NULL);
    jpeg_encode_custom_q(bg_img,  qbg,  out_bg,  has_sample ? samp_h : NULL, has_sample ? samp_v : NULL);

    image_free(src); image_free(padded); image_free(veg_img); image_free(bg_img);
    free(map);
    return 0;
  }

  if (strcmp(argv[1], "decode") == 0) {
    if (argc != 10) { usage(argv[0]); return 1; }
    const char *veg_path = argv[2], *bg_path = argv[3];
    const char *blockmap_path = argv[4];
    int bw = atoi(argv[5]), bh = atoi(argv[6]);
    int orig_w = atoi(argv[7]), orig_h = atoi(argv[8]);
    const char *out_path = argv[9];

    unsigned char *map = read_blockmap(blockmap_path, bw, bh);
    Image *veg_compact = jpeg_decode(veg_path);
    Image *bg_compact  = jpeg_decode(bg_path);

    Image padded_out;
    padded_out.width = bw * BLOCK;
    padded_out.height = bh * BLOCK;
    padded_out.data = calloc((size_t)padded_out.width*padded_out.height*3, 1);

    unpack_class(veg_compact, map, bw, bh, 1, &padded_out);
    unpack_class(bg_compact,  map, bw, bh, 0, &padded_out);

    Image cropped;
    crop_image(&padded_out, orig_w, orig_h, &cropped);
    ppm_write(out_path, &cropped);

    free(padded_out.data); free(cropped.data);
    image_free(veg_compact); image_free(bg_compact);
    free(map);
    return 0;
  }

  usage(argv[0]);
  return 1;
}
