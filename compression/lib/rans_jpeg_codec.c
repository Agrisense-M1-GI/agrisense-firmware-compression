/*
 * rans_jpeg_codec.c -- algo/agrijpeg-core
 * ==========================================
 * Replaces standard JPEG's Huffman entropy stage with rANS (ryg_rans,
 * rans_byte.h -- public domain), using the static frequency tables
 * produced by calibrate_rans.py (rans_tables.h, #include'd below --
 * recalibrating never requires touching this file, only rans_tables.h).
 *
 * Does NOT replace roi_jpeg_codec -- it wraps around it. Pipeline:
 *
 *   encode: mask -> roi_jpeg_codec encode -> veg.jpg/bg.jpg
 *           -> rans_jpeg_codec transcode-encode (x2) -> veg.rans/bg.rans
 *   decode: veg.rans/bg.rans -> rans_jpeg_codec transcode-decode (x2)
 *           -> veg.jpg/bg.jpg (reconstructed, byte-for-byte the same
 *              QUANTIZED COEFFICIENTS as the originals, just re-entropy-
 *              coded with standard Huffman instead of rANS)
 *           -> roi_jpeg_codec decode (UNCHANGED, reuses tested reassembly)
 *
 * No JPEG header/quant-table/sampling metadata is embedded in the .rans
 * file -- the station already knows all of it (same branch, same local
 * Qveg/Qbg table files, same --sample convention as roi_jpeg_codec), so
 * transcode-decode takes it as CLI arguments, exactly like roi_jpeg_codec
 * does. Component block counts (how many DC/AC symbols belong to Y vs
 * Cb vs Cr) are NOT transmitted either -- they are fully deterministic
 * from (bw, n_class_blocks, sampling factors), recomputed identically on
 * both sides with the same formula libjpeg itself uses internally.
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

/* ---------------------------------------------------------------------
 * rANS model: cumulative frequency table built once from a static
 * RANS_FREQ_* array.
 * ------------------------------------------------------------------- */

typedef struct {
    uint32_t freq[RANS_TOTAL_SYMBOLS];
    uint32_t cum[RANS_TOTAL_SYMBOLS + 1];
} RansModel;

static void build_model(const uint32_t *freqs, RansModel *m) {
    m->cum[0] = 0;
    for (int i = 0; i < RANS_TOTAL_SYMBOLS; i++) {
        m->freq[i] = freqs[i];
        m->cum[i + 1] = m->cum[i] + freqs[i];
    }
}

static int value_to_symbol(int v) {
    if (v >= -RANS_BIAS && v <= RANS_BIAS) return v + RANS_BIAS;
    return RANS_ESCAPE_SYMBOL;
}

/* Binary search: which symbol's [cum[s], cum[s+1]) range contains slot? */
static int symbol_from_slot(const RansModel *m, uint32_t slot) {
    int lo = 0, hi = RANS_TOTAL_SYMBOLS;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (m->cum[mid] <= slot) lo = mid; else hi = mid;
    }
    return lo;
}

/* ---------------------------------------------------------------------
 * Encode/decode one logical stream (e.g. all Y DC-diffs) against one
 * model. Escapes are stored separately, in original (forward) order.
 * ------------------------------------------------------------------- */

static void rans_encode_stream(const int *values, int n, const RansModel *m,
                                uint8_t **out_buf, size_t *out_len,
                                int16_t **escapes_out, int *n_escapes_out) {
    int16_t *escapes = n > 0 ? malloc(sizeof(int16_t) * (size_t)n) : NULL;
    int n_escapes = 0;
    int *symbols = n > 0 ? malloc(sizeof(int) * (size_t)n) : NULL;

    for (int i = 0; i < n; i++) {
        int s = value_to_symbol(values[i]);
        symbols[i] = s;
        if (s == RANS_ESCAPE_SYMBOL) escapes[n_escapes++] = (int16_t)values[i];
    }

    if (n == 0) {
        *out_buf = NULL;
        *out_len = 0;
        *escapes_out = NULL;
        *n_escapes_out = 0;
        free(escapes);
        free(symbols);
        return;
    }

    /* Worst case: every symbol needs up to 4 bytes, plus flush. Safe
     * upper bound, shrunk to the real size below. */
    size_t cap = (size_t)n * 4 + 64;
    uint8_t *buf = malloc(cap);
    uint8_t *ptr = buf + cap; /* rANS encodes backward from the end */

    RansState rs;
    RansEncInit(&rs);
    /* NOTE: symbols must be fed in REVERSE order -- a property of rANS,
     * not a stylistic choice (see rans_byte.h's own comment on this). */
    for (int i = n - 1; i >= 0; i--) {
        int s = symbols[i];
        RansEncPut(&rs, &ptr, m->cum[s], m->freq[s], RANS_PROB_BITS);
    }
    RansEncFlush(&rs, &ptr);

    *out_len = (size_t)((buf + cap) - ptr);
    *out_buf = malloc(*out_len);
    memcpy(*out_buf, ptr, *out_len);
    free(buf);
    free(symbols);

    *escapes_out = n_escapes > 0 ? realloc(escapes, sizeof(int16_t) * (size_t)n_escapes) : (free(escapes), NULL);
    *n_escapes_out = n_escapes;
}

static void rans_decode_stream(const uint8_t *in_buf, int n, const RansModel *m,
                                const int16_t *escapes, int *values_out) {
    if (n == 0) return;

    uint8_t *ptr = (uint8_t *)in_buf;
    RansState rs;
    RansDecInit(&rs, &ptr);

    int escape_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t slot = RansDecGet(&rs, RANS_PROB_BITS);
        int sym = symbol_from_slot(m, slot);
        RansDecAdvance(&rs, &ptr, m->cum[sym], m->freq[sym], RANS_PROB_BITS);

        values_out[i] = (sym == RANS_ESCAPE_SYMBOL)
                             ? escapes[escape_idx++]
                             : (sym - RANS_BIAS);
    }
}

/* ---------------------------------------------------------------------
 * .rans file format: 4 streams back to back, each self-delimited by a
 * length prefix. Symbol counts per stream are NOT stored -- both encode
 * and decode derive them independently from the same geometry inputs
 * (see compute_block_dims below), so storing them would be redundant.
 *
 *   [u32 len][rans bytes][u32 n_escapes][escapes, int16 each]   x4
 *   order: Y_DC, Y_AC, C_DC, C_AC
 * ------------------------------------------------------------------- */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }

static uint32_t read_u32(FILE *f) {
    uint32_t v;
    if (fread(&v, sizeof(v), 1, f) != 1) {
        fprintf(stderr, "unexpected end of .rans file\n");
        exit(1);
    }
    return v;
}

static void write_stream(FILE *f, const uint8_t *buf, size_t len,
                          const int16_t *escapes, int n_escapes) {
    write_u32(f, (uint32_t)len);
    if (len > 0) fwrite(buf, 1, len, f);
    write_u32(f, (uint32_t)n_escapes);
    if (n_escapes > 0) fwrite(escapes, sizeof(int16_t), (size_t)n_escapes, f);
}

static void read_stream(FILE *f, uint8_t **buf_out, size_t *len_out,
                         int16_t **escapes_out, int *n_escapes_out) {
    uint32_t len = read_u32(f);
    uint8_t *buf = NULL;
    if (len > 0) {
        buf = malloc(len);
        if (fread(buf, 1, len, f) != len) { fprintf(stderr, "truncated .rans stream\n"); exit(1); }
    }
    uint32_t n_escapes = read_u32(f);
    int16_t *escapes = NULL;
    if (n_escapes > 0) {
        escapes = malloc(sizeof(int16_t) * n_escapes);
        if (fread(escapes, sizeof(int16_t), n_escapes, f) != n_escapes) {
            fprintf(stderr, "truncated .rans escape list\n"); exit(1);
        }
    }
    *buf_out = buf;
    *len_out = len;
    *escapes_out = escapes;
    *n_escapes_out = (int)n_escapes;
}

/* ---------------------------------------------------------------------
 * Component block-grid geometry -- same formula libjpeg itself uses
 * (jcmaster.c), so both encode (which reads it straight from libjpeg
 * after jpeg_read_coefficients) and decode (which must recompute it
 * from scratch before jpeg_write_coefficients) agree exactly.
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

static void collect_component(j_decompress_ptr cinfo, jvirt_barray_ptr *coef_arrays, int ci,
                               int **dc_out, int *n_dc_out, int **ac_out, int *n_ac_out) {
    jpeg_component_info *comp = &cinfo->comp_info[ci];
    int n_blocks = (int)(comp->width_in_blocks * comp->height_in_blocks);

    int *dc = malloc(sizeof(int) * (size_t)n_blocks);
    int *ac = malloc(sizeof(int) * (size_t)n_blocks * (DCTSIZE2 - 1));
    int dc_idx = 0, ac_idx = 0;
    short prev_dc = 0;

    for (JDIMENSION by = 0; by < comp->height_in_blocks; by++) {
        JBLOCKARRAY buffer = (*cinfo->mem->access_virt_barray)(
            (j_common_ptr)cinfo, coef_arrays[ci], by, 1, FALSE);
        for (JDIMENSION bx = 0; bx < comp->width_in_blocks; bx++) {
            JCOEFPTR blockptr = buffer[0][bx];
            short dc_val = blockptr[0];
            dc[dc_idx++] = (short)(dc_val - prev_dc);
            prev_dc = dc_val;
            for (int k = 1; k < DCTSIZE2; k++) ac[ac_idx++] = blockptr[k];
        }
    }

    *dc_out = dc; *n_dc_out = dc_idx;
    *ac_out = ac; *n_ac_out = ac_idx;
}

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

    int *y_dc, *y_ac, *cb_dc, *cb_ac, *cr_dc, *cr_ac;
    int n_y_dc, n_y_ac, n_cb_dc, n_cb_ac, n_cr_dc, n_cr_ac;
    collect_component(&cinfo, coef_arrays, 0, &y_dc, &n_y_dc, &y_ac, &n_y_ac);
    collect_component(&cinfo, coef_arrays, 1, &cb_dc, &n_cb_dc, &cb_ac, &n_cb_ac);
    collect_component(&cinfo, coef_arrays, 2, &cr_dc, &n_cr_dc, &cr_ac, &n_cr_ac);

    /* Chroma DC/AC pooled: Cb block then Cr blocks, concatenated -- DC
     * prediction resets at the Cb/Cr boundary (matches collect_component
     * resetting prev_dc=0 per component call, done independently above). */
    int n_c_dc = n_cb_dc + n_cr_dc;
    int *c_dc = malloc(sizeof(int) * (size_t)n_c_dc);
    memcpy(c_dc, cb_dc, sizeof(int) * (size_t)n_cb_dc);
    memcpy(c_dc + n_cb_dc, cr_dc, sizeof(int) * (size_t)n_cr_dc);

    int n_c_ac = n_cb_ac + n_cr_ac;
    int *c_ac = malloc(sizeof(int) * (size_t)n_c_ac);
    memcpy(c_ac, cb_ac, sizeof(int) * (size_t)n_cb_ac);
    memcpy(c_ac + n_cb_ac, cr_ac, sizeof(int) * (size_t)n_cr_ac);

    RansModel m_y_dc, m_y_ac, m_c_dc, m_c_ac;
    build_model(RANS_FREQ_Y_DC, &m_y_dc);
    build_model(RANS_FREQ_Y_AC, &m_y_ac);
    build_model(RANS_FREQ_C_DC, &m_c_dc);
    build_model(RANS_FREQ_C_AC, &m_c_ac);

    uint8_t *buf_y_dc, *buf_y_ac, *buf_c_dc, *buf_c_ac;
    size_t len_y_dc, len_y_ac, len_c_dc, len_c_ac;
    int16_t *esc_y_dc, *esc_y_ac, *esc_c_dc, *esc_c_ac;
    int n_esc_y_dc, n_esc_y_ac, n_esc_c_dc, n_esc_c_ac;

    rans_encode_stream(y_dc, n_y_dc, &m_y_dc, &buf_y_dc, &len_y_dc, &esc_y_dc, &n_esc_y_dc);
    rans_encode_stream(y_ac, n_y_ac, &m_y_ac, &buf_y_ac, &len_y_ac, &esc_y_ac, &n_esc_y_ac);
    rans_encode_stream(c_dc, n_c_dc, &m_c_dc, &buf_c_dc, &len_c_dc, &esc_c_dc, &n_esc_c_dc);
    rans_encode_stream(c_ac, n_c_ac, &m_c_ac, &buf_c_ac, &len_c_ac, &esc_c_ac, &n_esc_c_ac);

    FILE *outfile = fopen(out_path, "wb");
    if (!outfile) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    write_stream(outfile, buf_y_dc, len_y_dc, esc_y_dc, n_esc_y_dc);
    write_stream(outfile, buf_y_ac, len_y_ac, esc_y_ac, n_esc_y_ac);
    write_stream(outfile, buf_c_dc, len_c_dc, esc_c_dc, n_esc_c_dc);
    write_stream(outfile, buf_c_ac, len_c_ac, esc_c_ac, n_esc_c_ac);
    fclose(outfile);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    free(y_dc); free(y_ac); free(cb_dc); free(cb_ac); free(cr_dc); free(cr_ac);
    free(c_dc); free(c_ac);
    free(buf_y_dc); free(buf_y_ac); free(buf_c_dc); free(buf_c_ac);
    free(esc_y_dc); free(esc_y_ac); free(esc_c_dc); free(esc_c_ac);
    return 0;
}

/* ---------------------------------------------------------------------
 * transcode-decode
 * ------------------------------------------------------------------- */

static void read_qtable(const char *path, unsigned int qtable[64]) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open qtable %s\n", path); exit(1); }
    for (int i = 0; i < 64; i++) {
        if (fscanf(f, "%u", &qtable[i]) != 1) {
            fprintf(stderr, "malformed qtable %s\n", path); exit(1);
        }
    }
    fclose(f);
}

static int cmd_transcode_decode(const char *in_path, int bw, int n_class_blocks,
                                 const char *qtable_path, const char *out_path,
                                 const int *samp_h, const int *samp_v) {
    int canvas_w = bw * 8;
    int canvas_h = (int)(((n_class_blocks + bw - 1) / bw)) * 8;

    int width_in_blocks[3], height_in_blocks[3];
    compute_block_dims(canvas_w, canvas_h, samp_h, samp_v, width_in_blocks, height_in_blocks);

    int n_y_blocks = width_in_blocks[0] * height_in_blocks[0];
    int n_cb_blocks = width_in_blocks[1] * height_in_blocks[1];
    int n_cr_blocks = width_in_blocks[2] * height_in_blocks[2];
    int n_c_blocks = n_cb_blocks + n_cr_blocks;

    FILE *infile = fopen(in_path, "rb");
    if (!infile) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }

    uint8_t *buf_y_dc, *buf_y_ac, *buf_c_dc, *buf_c_ac;
    size_t len_y_dc, len_y_ac, len_c_dc, len_c_ac;
    int16_t *esc_y_dc, *esc_y_ac, *esc_c_dc, *esc_c_ac;
    int n_esc_y_dc, n_esc_y_ac, n_esc_c_dc, n_esc_c_ac;
    read_stream(infile, &buf_y_dc, &len_y_dc, &esc_y_dc, &n_esc_y_dc);
    read_stream(infile, &buf_y_ac, &len_y_ac, &esc_y_ac, &n_esc_y_ac);
    read_stream(infile, &buf_c_dc, &len_c_dc, &esc_c_dc, &n_esc_c_dc);
    read_stream(infile, &buf_c_ac, &len_c_ac, &esc_c_ac, &n_esc_c_ac);
    fclose(infile);
    (void)len_y_dc; (void)len_y_ac; (void)len_c_dc; (void)len_c_ac;

    RansModel m_y_dc, m_y_ac, m_c_dc, m_c_ac;
    build_model(RANS_FREQ_Y_DC, &m_y_dc);
    build_model(RANS_FREQ_Y_AC, &m_y_ac);
    build_model(RANS_FREQ_C_DC, &m_c_dc);
    build_model(RANS_FREQ_C_AC, &m_c_ac);

    int *y_dc = malloc(sizeof(int) * (size_t)n_y_blocks);
    int *y_ac = malloc(sizeof(int) * (size_t)n_y_blocks * (DCTSIZE2 - 1));
    int *c_dc = malloc(sizeof(int) * (size_t)n_c_blocks);
    int *c_ac = malloc(sizeof(int) * (size_t)n_c_blocks * (DCTSIZE2 - 1));

    rans_decode_stream(buf_y_dc, n_y_blocks, &m_y_dc, esc_y_dc, y_dc);
    rans_decode_stream(buf_y_ac, n_y_blocks * (DCTSIZE2 - 1), &m_y_ac, esc_y_ac, y_ac);
    rans_decode_stream(buf_c_dc, n_c_blocks, &m_c_dc, esc_c_dc, c_dc);
    rans_decode_stream(buf_c_ac, n_c_blocks * (DCTSIZE2 - 1), &m_c_ac, esc_c_ac, c_ac);

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
    jpeg_add_quant_table(&cinfo, 0, (const unsigned int *)qtable, 100, TRUE);
    cinfo.comp_info[0].quant_tbl_no = 0;
    cinfo.comp_info[1].quant_tbl_no = 1;
    cinfo.comp_info[2].quant_tbl_no = 1;
    jpeg_set_colorspace(&cinfo, JCS_YCbCr);
    for (int c = 0; c < 3; c++) {
        cinfo.comp_info[c].h_samp_factor = samp_h[c];
        cinfo.comp_info[c].v_samp_factor = samp_v[c];
    }

    /* Allocate virtual coefficient arrays from scratch (not copied from
     * a decompress struct) -- the documented libjpeg pattern for
     * constructing a JPEG purely from coefficient data (same technique
     * jpegtran uses for lossless transforms). Must happen before
     * jpeg_write_coefficients(). */
    jvirt_barray_ptr coef_arrays[3];
    for (int ci = 0; ci < 3; ci++) {
        jpeg_component_info *comp = &cinfo.comp_info[ci];
        coef_arrays[ci] = (cinfo.mem->request_virt_barray)(
            (j_common_ptr)&cinfo, JPOOL_IMAGE, TRUE,
            (JDIMENSION)width_in_blocks[ci], (JDIMENSION)height_in_blocks[ci],
            (JDIMENSION)comp->v_samp_factor);
    }

    jpeg_write_coefficients(&cinfo, coef_arrays);

    /* Now fill the arrays with our decoded coefficients (write access). */
    int y_dc_idx = 0, y_ac_idx = 0;
    for (JDIMENSION by = 0; by < (JDIMENSION)height_in_blocks[0]; by++) {
        JBLOCKARRAY buffer = (*cinfo.mem->access_virt_barray)(
            (j_common_ptr)&cinfo, coef_arrays[0], by, 1, TRUE);
        for (JDIMENSION bx = 0; bx < (JDIMENSION)width_in_blocks[0]; bx++) {
            JCOEFPTR blockptr = buffer[0][bx];
            blockptr[0] = (JCOEF)y_dc[y_dc_idx++];
            for (int k = 1; k < DCTSIZE2; k++) blockptr[k] = (JCOEF)y_ac[y_ac_idx++];
        }
    }
    /* Undo DC differencing per chroma sub-component (Cb first, then Cr),
     * matching how collect_component() reset prev_dc=0 for each. */
    int c_idx = 0, c_ac_idx = 0;
    for (int ci = 1; ci <= 2; ci++) {
        short prev_dc = 0;
        for (JDIMENSION by = 0; by < (JDIMENSION)height_in_blocks[ci]; by++) {
            JBLOCKARRAY buffer = (*cinfo.mem->access_virt_barray)(
                (j_common_ptr)&cinfo, coef_arrays[ci], by, 1, TRUE);
            for (JDIMENSION bx = 0; bx < (JDIMENSION)width_in_blocks[ci]; bx++) {
                JCOEFPTR blockptr = buffer[0][bx];
                short dc_diff = (short)c_dc[c_idx++];
                short dc_val = (short)(prev_dc + dc_diff);
                prev_dc = dc_val;
                blockptr[0] = (JCOEF)dc_val;
                for (int k = 1; k < DCTSIZE2; k++) blockptr[k] = (JCOEF)c_ac[c_ac_idx++];
            }
        }
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);

    free(y_dc); free(y_ac); free(c_dc); free(c_ac);
    free(buf_y_dc); free(buf_y_ac); free(buf_c_dc); free(buf_c_ac);
    free(esc_y_dc); free(esc_y_ac); free(esc_c_dc); free(esc_c_ac);
    return 0;
}

/* ---------------------------------------------------------------------
 * CLI
 * ------------------------------------------------------------------- */

static int parse_sample_factors(const char *spec, int h[3], int v[3]) {
    int n = sscanf(spec, "%dx%d,%dx%d,%dx%d", &h[0], &v[0], &h[1], &v[1], &h[2], &v[2]);
    return n == 6;
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
        const char *in_rans = argv[2];
        int bw = atoi(argv[3]);
        int n_class_blocks = atoi(argv[4]);
        const char *qtable_path = argv[5];
        const char *out_jpg = argv[6];

        int samp_h[3] = {2, 1, 1}, samp_v[3] = {2, 1, 1}; /* default 4:2:0 */
        if (argc == 9) {
            if (strcmp(argv[7], "--sample") != 0) { usage(argv[0]); return 1; }
            if (!parse_sample_factors(argv[8], samp_h, samp_v)) {
                fprintf(stderr, "malformed --sample value: %s\n", argv[8]);
                return 1;
            }
        }
        return cmd_transcode_decode(in_rans, bw, n_class_blocks, qtable_path, out_jpg, samp_h, samp_v);
    }

    usage(argv[0]);
    return 1;
}
