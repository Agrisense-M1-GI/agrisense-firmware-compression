/*
 * dump_coeffs.c -- calibration helper for algo/agrijpeg-core (v2, RLE)
 * =========================================================================
 * v1 histogrammed raw coefficient values -- rANS encoded every one of the
 * 63 AC positions individually, including every trailing zero, which
 * loses badly against Huffman+RLE's near-free EOB. v2 instead builds the
 * SAME symbol alphabet standard JPEG itself uses: zigzag order, DC coded
 * as a size CATEGORY, AC coded as (RUNLENGTH-of-zeros, SIZE) byte pairs
 * with EOB (0x00) and ZRL (0xF0, "16 more zeros") -- this is what lets
 * rANS actually beat Huffman, by modeling the same compact alphabet
 * Huffman does, just closer to the entropy bound.
 *
 * Dumps ONE BYTE PER SYMBOL (not the coefficient value) to 4 accumulator
 * files, appended across the whole dataset:
 *   <prefix>_y_dc.sym   -- luminance DC size category (0-15)
 *   <prefix>_y_ac.sym   -- luminance AC (run<<4)|size bytes, incl. EOB/ZRL
 *   <prefix>_c_dc.sym   -- chrominance (Cb+Cr pooled) DC category
 *   <prefix>_c_ac.sym   -- chrominance (Cb+Cr pooled) AC bytes
 *
 * The actual coefficient magnitude bits ("extra bits") are NOT part of
 * calibration -- only the symbol distribution matters here, exactly like
 * how JPEG's own Huffman tables only cover (run,size), never the
 * magnitude bits (those are stored raw either way).
 *
 * Usage: ./dump_coeffs <in.jpg> <output_prefix>
 */

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

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

static void dump_component(j_decompress_ptr cinfo, jvirt_barray_ptr *coef_arrays,
                            int ci, FILE *dc_f, FILE *ac_f) {
    jpeg_component_info *comp = &cinfo->comp_info[ci];
    short prev_dc = 0;

    for (JDIMENSION by = 0; by < comp->height_in_blocks; by++) {
        JBLOCKARRAY buffer = (*cinfo->mem->access_virt_barray)(
            (j_common_ptr)cinfo, coef_arrays[ci], by, 1, FALSE);
        for (JDIMENSION bx = 0; bx < comp->width_in_blocks; bx++) {
            JCOEFPTR blockptr = buffer[0][bx];

            /* DC: size category of the difference from the previous block. */
            short dc = blockptr[0];
            short dc_diff = (short)(dc - prev_dc);
            prev_dc = dc;
            unsigned char dc_sym = (unsigned char)category_of(dc_diff);
            fwrite(&dc_sym, 1, 1, dc_f);

            /* AC: zigzag order, standard (run, size) + EOB/ZRL. */
            int last_nz = 0;
            for (int k = 1; k < 64; k++)
                if (blockptr[ZIGZAG[k]] != 0) last_nz = k;

            int run = 0;
            for (int k = 1; k <= last_nz; k++) {
                int v = blockptr[ZIGZAG[k]];
                if (v == 0) {
                    run++;
                } else {
                    while (run >= 16) {
                        unsigned char zrl = 0xF0;
                        fwrite(&zrl, 1, 1, ac_f);
                        run -= 16;
                    }
                    unsigned char sym = (unsigned char)((run << 4) | category_of(v));
                    fwrite(&sym, 1, 1, ac_f);
                    run = 0;
                }
            }
            if (last_nz < 63) {
                unsigned char eob = 0x00;
                fwrite(&eob, 1, 1, ac_f);
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <in.jpg> <output_prefix>\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *prefix = argv[2];

    FILE *infile = fopen(in_path, "rb");
    if (!infile) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    jvirt_barray_ptr *coef_arrays = jpeg_read_coefficients(&cinfo);
    if (coef_arrays == NULL) {
        fprintf(stderr, "jpeg_read_coefficients failed for %s\n", in_path);
        return 1;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s_y_dc.sym", prefix);
    FILE *dc_y = fopen(path, "ab");
    snprintf(path, sizeof(path), "%s_y_ac.sym", prefix);
    FILE *ac_y = fopen(path, "ab");
    snprintf(path, sizeof(path), "%s_c_dc.sym", prefix);
    FILE *dc_c = fopen(path, "ab");
    snprintf(path, sizeof(path), "%s_c_ac.sym", prefix);
    FILE *ac_c = fopen(path, "ab");

    if (!dc_y || !ac_y || !dc_c || !ac_c) {
        fprintf(stderr, "cannot open output files with prefix %s\n", prefix);
        return 1;
    }

    for (int ci = 0; ci < cinfo.num_components; ci++) {
        FILE *dc_f = (ci == 0) ? dc_y : dc_c;
        FILE *ac_f = (ci == 0) ? ac_y : ac_c;
        dump_component(&cinfo, coef_arrays, ci, dc_f, ac_f);
    }

    fclose(dc_y);
    fclose(ac_y);
    fclose(dc_c);
    fclose(ac_c);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
}
