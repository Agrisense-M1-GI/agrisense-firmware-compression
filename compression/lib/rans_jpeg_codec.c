/*
 * rans_jpeg_codec.c -- algo/agrijpeg-core (v2, RLE alphabet)
 * ==============================================================
 * v2: encodes the same (run,size)/category symbol alphabet standard
 * JPEG Huffman coding uses (zigzag order, EOB/ZRL) instead of raw
 * per-position coefficient values -- v1 rANS-coded all 63 AC values
 * individually including every trailing zero, losing badly to
 * Huffman+RLE's near-free EOB. Magnitude "extra bits" (the actual value
 * within a category) are NOT rANS-coded, exactly like real JPEG: packed
 * as a raw bitstream alongside the rANS-coded symbol stream.
 *
 * Usage:
 *   rans_jpeg_codec transcode-encode <in.jpg> <out.rans>
 *   rans_jpeg_codec transcode-decode <in.rans> <bw> <n_class_blocks> <qtable.txt> <out.jpg> [--sample H1xV1,H2xV2,H3xV3]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <jpeglib.h>

#include "rans_byte.h"
#include "rans_tables.h"

static const int ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

static int category_of(int v) {
    if (v < 0) v = -v;
    int cat = 0;
    while (v) { cat++; v >>= 1; }
    return cat;
}

/* JPEG's standard magnitude encoding: for a category-`size` value v,
 * the "extra bits" are v itself if v>0, or v + (2^size - 1) if v<0
 * (maps negatives into the lower half of the size-bit range). */
static unsigned int extra_bits_of(int v, int size) {
    if (size == 0) return 0;
    if (v >= 0) return (unsigned int)v;
    return (unsigned int)(v + (1 << size) - 1);
}

static int value_from_extra_bits(unsigned int bits, int size) {
    if (size == 0) return 0;
    if (bits >= (unsigned int)(1 << (size - 1))) return (int)bits;
    return (int)bits - (1 << size) + 1;
}

/* ---------------------------------------------------------------------
 * A simple MSB-first bit writer/reader for the raw "extra bits" stream.
 * ------------------------------------------------------------------- */

typedef struct { uint8_t *buf; size_t cap; size_t byte_pos; int bit_pos; } BitWriter;

static void bw_init(BitWriter *w, size_t cap) {
    w->buf = calloc(1, cap);
    w->cap = cap;
    w->byte_pos = 0;
    w->bit_pos = 0;
}

static void bw_put(BitWriter *w, unsigned int bits, int n) {
    for (int i = n - 1; i >= 0; i--) {
        int bit = (bits >> i) & 1;
        w->buf[w->byte_pos] |= (uint8_t)(bit << (7 - w->bit_pos));
        w->bit_pos++;
        if (w->bit_pos == 8) { w->bit_pos = 0; w->byte_pos++; }
    }
}

static size_t bw_byte_len(const BitWriter *w) {
    return w->byte_pos + (w->bit_pos > 0 ? 1 : 0);
}

typedef struct { const uint8_t *buf; size_t byte_pos; int bit_pos; } BitReader;

static void br_init(BitReader *r, const uint8_t *buf) {
    r->buf = buf; r->byte_pos = 0; r->bit_pos = 0;
}

static unsigned int br_get(BitReader *r, int n) {
    unsigned int v = 0;
    for (int i = 0; i < n; i++) {
        int bit = (r->buf[r->byte_pos] >> (7 - r->bit_pos)) & 1;
        v = (v << 1) | (unsigned int)bit;
        r->bit_pos++;
        if (r->bit_pos == 8) { r->bit_pos = 0; r->byte_pos++; }
    }
    return v;
}

/* ---------------------------------------------------------------------
 * rANS model (generic size, works for both the 16-symbol DC alphabet
 * and the 256-symbol AC alphabet).
 * ------------------------------------------------------------------- */

typedef struct { uint32_t *freq; uint32_t *cum; int n; } RansModel;

static void build_model(const uint32_t *freqs, int n, RansModel *m) {
    m->n = n;
    m->freq = malloc(sizeof(uint32_t) * (size_t)n);
    m->cum = malloc(sizeof(uint32_t) * (size_t)(n + 1));
    m->cum[0] = 0;
    for (int i = 0; i < n; i++) {
        m->freq[i] = freqs[i];
        m->cum[i + 1] = m->cum[i] + freqs[i];
    }
}

static void free_model(RansModel *m) { free(m->freq); free(m->cum); }

static int symbol_from_slot(const RansModel *m, uint32_t slot) {
    int lo = 0, hi = m->n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (m->cum[mid] <= slot) lo = mid; else hi = mid;
    }
    return lo;
}

/* ---------------------------------------------------------------------
 * Per-block symbol lists, built once per component then rANS-encoded
 * together (symbols must be fed to RansEncPut in REVERSE order).
 * ------------------------------------------------------------------- */

typedef struct {
    int *symbols;      /* DC category or AC (run<<4|size) byte, per emitted symbol */
    int *extra_vals;   /* the actual coefficient value, for extra-bit packing (AC/DC) */
    int n;
    int cap;
} SymbolList;

static void sl_init(SymbolList *l, int cap) {
    l->symbols = malloc(sizeof(int) * (size_t)cap);
    l->extra_vals = malloc(sizeof(int) * (size_t)cap);
    l->n = 0;
    l->cap = cap;
}

static void sl_push(SymbolList *l, int symbol, int value) {
    l->symbols[l->n] = symbol;
    l->extra_vals[l->n] = value;
    l->n++;
}

/* ---------------------------------------------------------------------
 * Collect DC/AC symbol lists from one component's coefficients
 * (jpeg_read_coefficients side -- encode path).
 * ------------------------------------------------------------------- */

static void collect_component(j_decompress_ptr cinfo, jvirt_barray_ptr *coef_arrays, int ci,
                               SymbolList *dc, SymbolList *ac) {
    jpeg_component_info *comp = &cinfo->comp_info[ci];
    int n_blocks = (int)(comp->width_in_blocks * comp->height_in_blocks);
    sl_init(dc, n_blocks);
    sl_init(ac, n_blocks * 64); /* generous upper bound incl. ZRL/EOB */

    short prev_dc = 0;
    for (JDIMENSION by = 0; by < comp->height_in_blocks; by++) {
        JBLOCKARRAY buffer = (*cinfo->mem->access_virt_barray)(
            (j_common_ptr)cinfo, coef_arrays[ci], by, 1, FALSE);
        for (JDIMENSION bx = 0; bx < comp->width_in_blocks; bx++) {
            JCOEFPTR blockptr = buffer[0][bx];

            short dc_val = blockptr[0];
            short dc_diff = (short)(dc_val - prev_dc);
            prev_dc = dc_val;
            int dc_size = category_of(dc_diff);
            sl_push(dc, dc_size, dc_diff);

            int last_nz = 0;
            for (int k = 1; k < 64; k++)
                if (blockptr[ZIGZAG[k]] != 0) last_nz = k;

            int run = 0;
            for (int k = 1; k <= last_nz; k++) {
                int v = blockptr[ZIGZAG[k]];
                if (v == 0) {
                    run++;
                } else {
                    while (run >= 16) { sl_push(ac, 0xF0, 0); run -= 16; }
                    int size = category_of(v);
                    sl_push(ac, (run << 4) | size, v);
                    run = 0;
                }
            }
            if (last_nz < 63) sl_push(ac, 0x00, 0);
        }
    }
}

/* ---------------------------------------------------------------------
 * rANS-encode a SymbolList's `symbols` (reverse order, per rANS's own
 * requirement) and pack `extra_vals`' magnitude bits separately.
 * ------------------------------------------------------------------- */

static void encode_stream(const SymbolList *l, const RansModel *m, int is_dc,
                           uint8_t **rans_buf, size_t *rans_len,
                           uint8_t **bits_buf, size_t *bits_len) {
    if (l->n == 0) {
        *rans_buf = NULL; *rans_len = 0;
        *bits_buf = NULL; *bits_len = 0;
        return;
    }

    size_t cap = (size_t)l->n * 4 + 64;
    uint8_t *buf = malloc(cap);
    uint8_t *ptr = buf + cap;

    RansState rs;
    RansEncInit(&rs);
    for (int i = l->n - 1; i >= 0; i--) {
        int s = l->symbols[i];
        RansEncPut(&rs, &ptr, m->cum[s], m->freq[s], RANS_PROB_BITS);
    }
    RansEncFlush(&rs, &ptr);

    *rans_len = (size_t)((buf + cap) - ptr);
    *rans_buf = malloc(*rans_len);
    memcpy(*rans_buf, ptr, *rans_len);
    free(buf);

    /* Extra bits: for DC, size = symbol itself; for AC, size = symbol & 0xF
     * (and EOB/ZRL, symbol 0x00/0xF0, carry no extra bits). */
    BitWriter bw;
    bw_init(&bw, (size_t)l->n * 2 + 16);
    for (int i = 0; i < l->n; i++) {
        int size = is_dc ? l->symbols[i] : (l->symbols[i] & 0x0F);
        if (size > 0 && !(l->symbols[i] == 0x00 || l->symbols[i] == 0xF0)) {
            bw_put(&bw, extra_bits_of(l->extra_vals[i], size), size);
        } else if (is_dc && size > 0) {
            bw_put(&bw, extra_bits_of(l->extra_vals[i], size), size);
        }
    }
    *bits_len = bw_byte_len(&bw);
    *bits_buf = bw.buf; /* caller frees */
}

/* ---------------------------------------------------------------------
 * File I/O: 4 streams (Y_DC, Y_AC, C_DC, C_AC), each as
 * [u32 rans_len][rans bytes][u32 bits_len][bits bytes].
 * Symbol counts are NOT stored -- both sides derive them identically
 * from (bw, n_class_blocks, sampling factors), same as v1.
 * ------------------------------------------------------------------- */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static uint32_t read_u32(FILE *f) {
    uint32_t v;
    if (fread(&v, sizeof(v), 1, f) != 1) { fprintf(stderr, "unexpected EOF in .rans\n"); exit(1); }
    return v;
}

static void write_block(FILE *f, const uint8_t *buf, size_t len) {
    write_u32(f, (uint32_t)len);
    if (len > 0) fwrite(buf, 1, len, f);
}

static uint8_t *read_block(FILE *f, size_t *len_out) {
    uint32_t len = read_u32(f);
    uint8_t *buf = NULL;
    if (len > 0) {
        buf = malloc(len);
        if (fread(buf, 1, len, f) != len) { fprintf(stderr, "truncated .rans block\n"); exit(1); }
    }
    *len_out = len;
    return buf;
}

/* ---------------------------------------------------------------------
 * Component block-grid geometry (same formula libjpeg itself uses).
 * ------------------------------------------------------------------- */

static void compute_block_dims(int image_width, int image_height,
                                const int samp_h[3], const int samp_v[3],
                                int width_in_blocks[3], int height_in_blocks[3]) {
    int max_h = samp_h[0], max_v = samp_v[0];
    for (int c = 1; c < 3; c++) {
        if (samp_h[c] > max_h) max_h = samp_h[c];
        if (samp_v[c] > max_v) max_v = samp_v[c];
    }
    for (int c = 0; c < 3; c++) {
        width_in_blocks[c] = (image_width * samp_h[c] + max_h * 8 - 1) / (max_h * 8);
        height_in_blocks[c] = (image_height * samp_v[c] + max_v * 8 - 1) / (max_v * 8);
    }
}

/* ---------------------------------------------------------------------
 * transcode-encode
 * ------------------------------------------------------------------- */

static int cmd_transcode_encode(const char *in_path, const char *out_path) {
    FILE *infile = fopen(in_path, "rb");
    if (!infile) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jvirt_barray_ptr *coef_arrays = jpeg_read_coefficients(&cinfo);
    if (!coef_arrays) { fprintf(stderr, "jpeg_read_coefficients failed\n"); return 1; }

    SymbolList y_dc, y_ac, cb_dc, cb_ac, cr_dc, cr_ac;
    collect_component(&cinfo, coef_arrays, 0, &y_dc, &y_ac);
    collect_component(&cinfo, coef_arrays, 1, &cb_dc, &cb_ac);
    collect_component(&cinfo, coef_arrays, 2, &cr_dc, &cr_ac);

    /* Pool Cb+Cr: concatenate symbol lists (DC prediction already reset
     * per component inside collect_component). */
    SymbolList c_dc, c_ac;
    c_dc.n = cb_dc.n + cr_dc.n;
    c_dc.symbols = malloc(sizeof(int) * (size_t)c_dc.n);
    c_dc.extra_vals = malloc(sizeof(int) * (size_t)c_dc.n);
    memcpy(c_dc.symbols, cb_dc.symbols, sizeof(int) * (size_t)cb_dc.n);
    memcpy(c_dc.symbols + cb_dc.n, cr_dc.symbols, sizeof(int) * (size_t)cr_dc.n);
    memcpy(c_dc.extra_vals, cb_dc.extra_vals, sizeof(int) * (size_t)cb_dc.n);
    memcpy(c_dc.extra_vals + cb_dc.n, cr_dc.extra_vals, sizeof(int) * (size_t)cr_dc.n);

    c_ac.n = cb_ac.n + cr_ac.n;
    c_ac.symbols = malloc(sizeof(int) * (size_t)c_ac.n);
    c_ac.extra_vals = malloc(sizeof(int) * (size_t)c_ac.n);
    memcpy(c_ac.symbols, cb_ac.symbols, sizeof(int) * (size_t)cb_ac.n);
    memcpy(c_ac.symbols + cb_ac.n, cr_ac.symbols, sizeof(int) * (size_t)cr_ac.n);
    memcpy(c_ac.extra_vals, cb_ac.extra_vals, sizeof(int) * (size_t)cb_ac.n);
    memcpy(c_ac.extra_vals + cb_ac.n, cr_ac.extra_vals, sizeof(int) * (size_t)cr_ac.n);

    RansModel m_y_dc, m_y_ac, m_c_dc, m_c_ac;
    build_model(RANS_FREQ_Y_DC, RANS_DC_N_SYMBOLS, &m_y_dc);
    build_model(RANS_FREQ_Y_AC, RANS_AC_N_SYMBOLS, &m_y_ac);
    build_model(RANS_FREQ_C_DC, RANS_DC_N_SYMBOLS, &m_c_dc);
    build_model(RANS_FREQ_C_AC, RANS_AC_N_SYMBOLS, &m_c_ac);

    uint8_t *r_y_dc, *b_y_dc, *r_y_ac, *b_y_ac, *r_c_dc, *b_c_dc, *r_c_ac, *b_c_ac;
    size_t rl_y_dc, bl_y_dc, rl_y_ac, bl_y_ac, rl_c_dc, bl_c_dc, rl_c_ac, bl_c_ac;

    encode_stream(&y_dc, &m_y_dc, 1, &r_y_dc, &rl_y_dc, &b_y_dc, &bl_y_dc);
    encode_stream(&y_ac, &m_y_ac, 0, &r_y_ac, &rl_y_ac, &b_y_ac, &bl_y_ac);
    encode_stream(&c_dc, &m_c_dc, 1, &r_c_dc, &rl_c_dc, &b_c_dc, &bl_c_dc);
    encode_stream(&c_ac, &m_c_ac, 0, &r_c_ac, &rl_c_ac, &b_c_ac, &bl_c_ac);

    FILE *outfile = fopen(out_path, "wb");
    if (!outfile) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    write_block(outfile, r_y_dc, rl_y_dc); write_block(outfile, b_y_dc, bl_y_dc);
    write_block(outfile, r_y_ac, rl_y_ac); write_block(outfile, b_y_ac, bl_y_ac);
    write_block(outfile, r_c_dc, rl_c_dc); write_block(outfile, b_c_dc, bl_c_dc);
    write_block(outfile, r_c_ac, rl_c_ac); write_block(outfile, b_c_ac, bl_c_ac);
    fclose(outfile);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
}

/* ---------------------------------------------------------------------
 * transcode-decode
 * ------------------------------------------------------------------- */

static void read_qtable(const char *path, unsigned int qtable[64]) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open qtable %s\n", path); exit(1); }
    for (int i = 0; i < 64; i++)
        if (fscanf(f, "%u", &qtable[i]) != 1) { fprintf(stderr, "malformed qtable\n"); exit(1); }
    fclose(f);
}

/* Decodes n_blocks blocks' worth of DC+AC symbols back into per-block
 * (dc_value, ac[63]) arrays, undoing DC prediction and the RLE. */
static void decode_component_blocks(const uint8_t *rans_dc, const uint8_t *bits_dc,
                                     const uint8_t *rans_ac, const uint8_t *bits_ac,
                                     int n_blocks, int dc_reset_at,
                                     const RansModel *m_dc, const RansModel *m_ac,
                                     short *dc_out, short *ac_out /* n_blocks*63 */) {
    if (n_blocks == 0) return;

    uint8_t *dc_ptr = (uint8_t *)rans_dc;
    RansState rs_dc;
    RansDecInit(&rs_dc, &dc_ptr);
    BitReader br_dc; br_init(&br_dc, bits_dc);

    uint8_t *ac_ptr = (uint8_t *)rans_ac;
    RansState rs_ac;
    RansDecInit(&rs_ac, &ac_ptr);
    BitReader br_ac; br_init(&br_ac, bits_ac);

    short prev_dc = 0;
    for (int b = 0; b < n_blocks; b++) {
        /* Cb and Cr are two independently DC-predicted sequences pooled
         * into one rANS/bitstream -- reset at the boundary, exactly like
         * collect_component() resets prev_dc=0 for each component call. */
        if (b == dc_reset_at) prev_dc = 0;

        uint32_t slot = RansDecGet(&rs_dc, RANS_PROB_BITS);
        int dc_size = symbol_from_slot(m_dc, slot);
        RansDecAdvance(&rs_dc, &dc_ptr, m_dc->cum[dc_size], m_dc->freq[dc_size], RANS_PROB_BITS);
        int dc_diff = 0;
        if (dc_size > 0) dc_diff = value_from_extra_bits(br_get(&br_dc, dc_size), dc_size);
        short dc_val = (short)(prev_dc + dc_diff);
        prev_dc = dc_val;
        dc_out[b] = dc_val;

        short *ac_block = ac_out + (size_t)b * 63;
        for (int i = 0; i < 63; i++) ac_block[i] = 0;

        int pos = 0; /* zigzag AC position 1..63, tracked 0-based here (0 == position 1) */
        while (pos < 63) {
            uint32_t ac_slot = RansDecGet(&rs_ac, RANS_PROB_BITS);
            int sym = symbol_from_slot(m_ac, ac_slot);
            RansDecAdvance(&rs_ac, &ac_ptr, m_ac->cum[sym], m_ac->freq[sym], RANS_PROB_BITS);

            if (sym == 0x00) { /* EOB */
                break;
            } else if (sym == 0xF0) { /* ZRL: 16 zeros */
                pos += 16;
            } else {
                int run = (sym >> 4) & 0x0F;
                int size = sym & 0x0F;
                pos += run;
                int val = value_from_extra_bits(br_get(&br_ac, size), size);
                if (pos < 63) ac_block[pos] = (short)val;
                pos += 1;
            }
        }
    }
}

static int cmd_transcode_decode(const char *in_path, int bw, int n_class_blocks,
                                 const char *qtable_path, const char *out_path,
                                 const int *samp_h, const int *samp_v) {
    int canvas_w = bw * 8;
    int canvas_h = ((n_class_blocks + bw - 1) / bw) * 8;

    int width_in_blocks[3], height_in_blocks[3];
    compute_block_dims(canvas_w, canvas_h, samp_h, samp_v, width_in_blocks, height_in_blocks);
    int n_y_blocks = width_in_blocks[0] * height_in_blocks[0];
    int n_cb_blocks = width_in_blocks[1] * height_in_blocks[1];
    int n_cr_blocks = width_in_blocks[2] * height_in_blocks[2];
    int n_c_blocks = n_cb_blocks + n_cr_blocks;

    FILE *infile = fopen(in_path, "rb");
    if (!infile) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    size_t l1, l2, l3, l4, l5, l6, l7, l8;
    uint8_t *r_y_dc = read_block(infile, &l1), *b_y_dc = read_block(infile, &l2);
    uint8_t *r_y_ac = read_block(infile, &l3), *b_y_ac = read_block(infile, &l4);
    uint8_t *r_c_dc = read_block(infile, &l5), *b_c_dc = read_block(infile, &l6);
    uint8_t *r_c_ac = read_block(infile, &l7), *b_c_ac = read_block(infile, &l8);
    fclose(infile);
    (void)l1; (void)l2; (void)l3; (void)l4; (void)l5; (void)l6; (void)l7; (void)l8;

    RansModel m_y_dc, m_y_ac, m_c_dc, m_c_ac;
    build_model(RANS_FREQ_Y_DC, RANS_DC_N_SYMBOLS, &m_y_dc);
    build_model(RANS_FREQ_Y_AC, RANS_AC_N_SYMBOLS, &m_y_ac);
    build_model(RANS_FREQ_C_DC, RANS_DC_N_SYMBOLS, &m_c_dc);
    build_model(RANS_FREQ_C_AC, RANS_AC_N_SYMBOLS, &m_c_ac);

    short *y_dc = malloc(sizeof(short) * (size_t)n_y_blocks);
    short *y_ac = malloc(sizeof(short) * (size_t)n_y_blocks * 63);
    decode_component_blocks(r_y_dc, b_y_dc, r_y_ac, b_y_ac, n_y_blocks, -1, &m_y_dc, &m_y_ac, y_dc, y_ac);

    short *c_dc = malloc(sizeof(short) * (size_t)n_c_blocks);
    short *c_ac = malloc(sizeof(short) * (size_t)n_c_blocks * 63);
    decode_component_blocks(r_c_dc, b_c_dc, r_c_ac, b_c_ac, n_c_blocks, n_cb_blocks, &m_c_dc, &m_c_ac, c_dc, c_ac);

    unsigned int qtable[64];
    read_qtable(qtable_path, qtable);

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile = fopen(out_path, "wb");
    if (!outfile) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = canvas_w;
    cinfo.image_height = canvas_h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    cinfo.jpeg_width = canvas_w;
    cinfo.jpeg_height = canvas_h;
    jpeg_add_quant_table(&cinfo, 0, (const unsigned int *)qtable, 100, TRUE);
    cinfo.comp_info[0].quant_tbl_no = 0;
    cinfo.comp_info[1].quant_tbl_no = 1;
    cinfo.comp_info[2].quant_tbl_no = 1;
    jpeg_set_colorspace(&cinfo, JCS_YCbCr);
    for (int c = 0; c < 3; c++) {
        cinfo.comp_info[c].h_samp_factor = samp_h[c];
        cinfo.comp_info[c].v_samp_factor = samp_v[c];
    }

    jvirt_barray_ptr coef_arrays[3];
    for (int ci = 0; ci < 3; ci++) {
        jpeg_component_info *comp = &cinfo.comp_info[ci];
        comp->width_in_blocks = (JDIMENSION)width_in_blocks[ci];
        comp->height_in_blocks = (JDIMENSION)height_in_blocks[ci];
        comp->DCT_h_scaled_size = DCTSIZE;
        comp->DCT_v_scaled_size = DCTSIZE;
        coef_arrays[ci] = (cinfo.mem->request_virt_barray)(
            (j_common_ptr)&cinfo, JPOOL_IMAGE, TRUE,
            (JDIMENSION)width_in_blocks[ci], (JDIMENSION)height_in_blocks[ci],
            (JDIMENSION)comp->v_samp_factor);
    }
    cinfo.min_DCT_h_scaled_size = DCTSIZE;
    cinfo.min_DCT_v_scaled_size = DCTSIZE;

    jpeg_write_coefficients(&cinfo, coef_arrays);

    int y_idx = 0;
    for (JDIMENSION by = 0; by < (JDIMENSION)height_in_blocks[0]; by++) {
        JBLOCKARRAY buffer = (*cinfo.mem->access_virt_barray)(
            (j_common_ptr)&cinfo, coef_arrays[0], by, 1, TRUE);
        for (JDIMENSION bx = 0; bx < (JDIMENSION)width_in_blocks[0]; bx++) {
            JCOEFPTR blockptr = buffer[0][bx];
            blockptr[0] = (JCOEF)y_dc[y_idx];
            for (int k = 1; k < 64; k++) blockptr[ZIGZAG[k]] = (JCOEF)y_ac[(size_t)y_idx * 63 + (k - 1)];
            y_idx++;
        }
    }
    int c_idx = 0;
    for (int ci = 1; ci <= 2; ci++) {
        for (JDIMENSION by = 0; by < (JDIMENSION)height_in_blocks[ci]; by++) {
            JBLOCKARRAY buffer = (*cinfo.mem->access_virt_barray)(
                (j_common_ptr)&cinfo, coef_arrays[ci], by, 1, TRUE);
            for (JDIMENSION bx = 0; bx < (JDIMENSION)width_in_blocks[ci]; bx++) {
                JCOEFPTR blockptr = buffer[0][bx];
                blockptr[0] = (JCOEF)c_dc[c_idx];
                for (int k = 1; k < 64; k++) blockptr[ZIGZAG[k]] = (JCOEF)c_ac[(size_t)c_idx * 63 + (k - 1)];
                c_idx++;
            }
        }
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    return 0;
}

/* ---------------------------------------------------------------------
 * CLI
 * ------------------------------------------------------------------- */

static int parse_sample_factors(const char *spec, int h[3], int v[3]) {
    return sscanf(spec, "%dx%d,%dx%d,%dx%d", &h[0], &v[0], &h[1], &v[1], &h[2], &v[2]) == 6;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s transcode-encode <in.jpg> <out.rans>\n"
        "  %s transcode-decode <in.rans> <bw> <n_class_blocks> <qtable.txt> <out.jpg> [--sample H1xV1,H2xV2,H3xV3]\n",
        prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    if (strcmp(argv[1], "transcode-encode") == 0) {
        if (argc != 4) { usage(argv[0]); return 1; }
        return cmd_transcode_encode(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "transcode-decode") == 0) {
        if (argc != 7 && argc != 9) { usage(argv[0]); return 1; }
        int bw = atoi(argv[3]);
        int n_class_blocks = atoi(argv[4]);
        int samp_h[3] = {2, 1, 1}, samp_v[3] = {2, 1, 1};
        if (argc == 9) {
            if (strcmp(argv[7], "--sample") != 0) { usage(argv[0]); return 1; }
            if (!parse_sample_factors(argv[8], samp_h, samp_v)) {
                fprintf(stderr, "malformed --sample: %s\n", argv[8]);
                return 1;
            }
        }
        return cmd_transcode_decode(argv[2], bw, n_class_blocks, argv[5], argv[6], samp_h, samp_v);
    }

    usage(argv[0]);
    return 1;
}
