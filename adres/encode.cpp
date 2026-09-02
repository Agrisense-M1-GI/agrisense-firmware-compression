/*
 * ADRES encoder (v3)
 * ---------------------------------------------------------------
 * Changements par rapport à v2 :
 *   - Lecture d'images PNG en entrée (via libpng) au lieu de PPM,
 *     pour correspondre au pipeline_test.py qui travaille en PNG.
 *   - Sauvegarde du masque ROI en PNG (save_mask_png) au lieu de PPM,
 *     pour que le serveur puisse l'ouvrir directement avec Pillow
 *     sans conversion intermédiaire.
 *   - Suppression du modèle énergie (Eq. 1 de l'article) : l'INA219
 *     saturait à 3.2 A au boot du Raspberry Pi. Les métriques retenues
 *     sont cpu_time_ms, memory_kb, compressed_bytes, compression_ratio.
 *   - Lecture PPM conservée pour compatibilité tests unitaires.
 *
 * Build (requires libpng-dev) :
 *   gcc -O2 -Wall -o encode encode.cpp -lpng -lz -lm
 *
 * Usage :
 *   ./encode <input.png> <output_dir> <profile:E|Q>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <png.h>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    unsigned char *r, *g, *b;
} Image;

/* ---------------------------------------------------------------- */
/* I/O PNG                                                           */
/* ---------------------------------------------------------------- */

Image* read_png(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Impossible d'ouvrir : %s\n", filename); return NULL; }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(f); return NULL; }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, NULL, NULL); fclose(f); return NULL; }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(f);
        return NULL;
    }

    png_init_io(png_ptr, f);
    png_read_info(png_ptr, info_ptr);

    int width  = (int)png_get_image_width(png_ptr, info_ptr);
    int height = (int)png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    /* Normalisation vers RGB 8 bits */
    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGBA ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_strip_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    /* Lecture des lignes */
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    for (int y = 0; y < height; y++)
        row_pointers[y] = (png_bytep)malloc(row_bytes);
    png_read_image(png_ptr, row_pointers);

    /* Remplissage de la structure Image (planar R/G/B) */
    Image *img = (Image*)malloc(sizeof(Image));
    img->width  = width;
    img->height = height;
    int size = width * height;
    img->r = (unsigned char*)malloc(size);
    img->g = (unsigned char*)malloc(size);
    img->b = (unsigned char*)malloc(size);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            img->r[idx] = row[x * 3 + 0];
            img->g[idx] = row[x * 3 + 1];
            img->b[idx] = row[x * 3 + 2];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(f);
    return img;
}

/* Conservé pour tests unitaires uniquement */
Image* read_ppm(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f); return NULL;
    }
    Image *img = (Image*)malloc(sizeof(Image));
    int maxval;
    if (fscanf(f, "%d %d %d", &img->width, &img->height, &maxval) != 3) {
        fclose(f); free(img); return NULL;
    }
    fgetc(f);
    int size = img->width * img->height;
    img->r = (unsigned char*)malloc(size);
    img->g = (unsigned char*)malloc(size);
    img->b = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++) {
        img->r[i] = (unsigned char)fgetc(f);
        img->g[i] = (unsigned char)fgetc(f);
        img->b[i] = (unsigned char)fgetc(f);
    }
    fclose(f);
    return img;
}

/* ---------------------------------------------------------------- */
/* Segmentation                                                       */
/* ---------------------------------------------------------------- */

unsigned char* rgb_to_gray(Image *img) {
    int size = img->width * img->height;
    unsigned char *gray = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++)
        gray[i] = (unsigned char)(0.299 * img->r[i] + 0.587 * img->g[i] + 0.114 * img->b[i]);
    return gray;
}

int otsu_threshold(unsigned char *gray, int size) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[gray[i]]++;
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];
    float sumB = 0; int wB = 0, wF = 0; float maxVar = 0; int threshold = 0;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        wF = size - wB;
        if (wF == 0) break;
        sumB += (float)(t * hist[t]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; threshold = t; }
    }
    return threshold;
}

void detect_roi_blocks(Image *img, unsigned char *roi_mask, float threshold,
                       unsigned char **otsu_full) {
    int bw = (img->width  + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bh = (img->height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int size = img->width * img->height;

    unsigned char *gray = rgb_to_gray(img);
    int otsu_th = otsu_threshold(gray, size);
    *otsu_full = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++)
        (*otsu_full)[i] = (gray[i] > otsu_th) ? 255 : 0;

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            int bright_pixels = 0, total_pixels = 0;
            for (int y = by * BLOCK_SIZE; y < (by + 1) * BLOCK_SIZE && y < img->height; y++) {
                for (int x = bx * BLOCK_SIZE; x < (bx + 1) * BLOCK_SIZE && x < img->width; x++) {
                    int idx = y * img->width + x;
                    if ((*otsu_full)[idx] == 255) bright_pixels++;
                    total_pixels++;
                }
            }
            roi_mask[by * bw + bx] = (bright_pixels > total_pixels * 0.25) ? 1 : 0;
        }
    }
    free(gray);
    (void)threshold;   /* paramètre de profil conservé pour symétrie, non utilisé */
}

/* ---------------------------------------------------------------- */
/* Compression PNG en mémoire (libpng)                               */
/* ---------------------------------------------------------------- */

typedef struct { unsigned char *buf; size_t len; size_t cap; } PngMemBuf;

static void png_mem_write(png_structp png_ptr, png_bytep data, png_size_t length) {
    PngMemBuf *mem = (PngMemBuf*)png_get_io_ptr(png_ptr);
    if (mem->len + length > mem->cap) {
        while (mem->len + length > mem->cap) mem->cap *= 2;
        mem->buf = (unsigned char*)realloc(mem->buf, mem->cap);
    }
    memcpy(mem->buf + mem->len, data, length);
    mem->len += length;
}
static void png_mem_flush(png_structp png_ptr) { (void)png_ptr; }

static unsigned char* png_compress_memory(unsigned char *rgb, int width, int height,
                                           size_t *out_size) {
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr  = png_create_info_struct(png_ptr);
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Erreur encodage PNG\n"); exit(1);
    }
    PngMemBuf mem;
    mem.cap = (size_t)width * height / 2 + 1024;
    mem.buf = (unsigned char*)malloc(mem.cap);
    mem.len = 0;
    png_set_write_fn(png_ptr, &mem, png_mem_write, png_mem_flush);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png_ptr, 9);
    png_write_info(png_ptr, info_ptr);
    png_bytep *rows = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) rows[y] = rgb + (size_t)y * width * 3;
    png_write_image(png_ptr, rows);
    png_write_end(png_ptr, NULL);
    free(rows);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    *out_size = mem.len;
    return mem.buf;
}

/* ---------------------------------------------------------------- */
/* Construction des buffers ROI / fond                               */
/* ---------------------------------------------------------------- */

static void build_roi_canvas(Image *img, unsigned char *roi_mask, int bw, int bh,
                              int q_roi, unsigned char **canvas_out) {
    int width = img->width, height = img->height;
    unsigned char *canvas = (unsigned char*)calloc((size_t)width * height * 3, 1);
    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            if (!roi_mask[by * bw + bx]) continue;
            int y0 = by * BLOCK_SIZE, y1 = (by + 1) * BLOCK_SIZE; if (y1 > height) y1 = height;
            int x0 = bx * BLOCK_SIZE, x1 = (bx + 1) * BLOCK_SIZE; if (x1 > width)  x1 = width;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int idx = y * width + x, oidx = idx * 3;
                    canvas[oidx + 0] = (unsigned char)((img->r[idx] / q_roi) * q_roi);
                    canvas[oidx + 1] = (unsigned char)((img->g[idx] / q_roi) * q_roi);
                    canvas[oidx + 2] = (unsigned char)((img->b[idx] / q_roi) * q_roi);
                }
            }
        }
    }
    *canvas_out = canvas;
}

static void build_background_buffer(Image *img, int q_bg, int subsample,
                                     unsigned char **bg_buf, int *bg_w, int *bg_h) {
    int width = img->width, height = img->height;
    int bgw = (width  + subsample - 1) / subsample;
    int bgh = (height + subsample - 1) / subsample;
    unsigned char *bgbuf = (unsigned char*)malloc((size_t)bgw * bgh * 3);
    for (int by = 0; by < bgh; by++) {
        for (int bx = 0; bx < bgw; bx++) {
            int x = bx * subsample; if (x >= width)  x = width  - 1;
            int y = by * subsample; if (y >= height) y = height - 1;
            int idx = y * width + x, oidx = (by * bgw + bx) * 3;
            bgbuf[oidx + 0] = (unsigned char)((img->r[idx] / q_bg) * q_bg);
            bgbuf[oidx + 1] = (unsigned char)((img->g[idx] / q_bg) * q_bg);
            bgbuf[oidx + 2] = (unsigned char)((img->b[idx] / q_bg) * q_bg);
        }
    }
    *bg_buf = bgbuf; *bg_w = bgw; *bg_h = bgh;
}

/* ---------------------------------------------------------------- */
/* Sauvegarde du fichier compressé                                    */
/* ---------------------------------------------------------------- */

void save_compressed(const char *filename, int width, int height,
                     int q_roi, int q_bg, int subsample,
                     unsigned char *roi_mask, int bw, int bh,
                     unsigned char *roi_png, size_t roi_png_size,
                     unsigned char *bg_png,  size_t bg_png_size,
                     int bg_w, int bg_h) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "PROTO2v3\n");
    fprintf(f, "%d %d\n", width, height);
    fprintf(f, "%d %d %d\n", q_roi, q_bg, subsample);
    fprintf(f, "%d %d\n", bw, bh);
    fprintf(f, "%zu\n", roi_png_size);
    fprintf(f, "%d %d %zu\n", bg_w, bg_h, bg_png_size);
    fprintf(f, "DATA\n");
    fwrite(roi_mask, 1, (size_t)bw * bh, f);
    fwrite(roi_png,  1, roi_png_size, f);
    fwrite(bg_png,   1, bg_png_size,  f);
    fclose(f);
}

/*
 * save_mask_png : sauvegarde le masque ROI (bloc-level) en PNG niveaux
 * de gris (0 = fond, 255 = ROI). Remplace save_mask_image (PPM) pour
 * que Pillow côté serveur puisse l'ouvrir sans conversion.
 */
void save_mask_png(const char *filename, unsigned char *roi_mask,
                   int width, int height, int bw, int bh) {
    /* Construction d'un buffer RGB niveaux de gris pleine résolution */
    unsigned char *buf = (unsigned char*)malloc((size_t)width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int bx = x / BLOCK_SIZE;
            int by_idx = y / BLOCK_SIZE;
            /* Clamp au cas où l'image ne serait pas un multiple exact de BLOCK_SIZE */
            if (bx >= bw) bx = bw - 1;
            if (by_idx >= bh) by_idx = bh - 1;
            unsigned char val = roi_mask[by_idx * bw + bx] ? 255 : 0;
            int oidx = (y * width + x) * 3;
            buf[oidx + 0] = val;
            buf[oidx + 1] = val;
            buf[oidx + 2] = val;
        }
    }
    size_t dummy;
    unsigned char *png_bytes = png_compress_memory(buf, width, height, &dummy);
    free(buf);

    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(png_bytes, 1, dummy, f); fclose(f); }
    free(png_bytes);
    (void)bh;
}

/* ---------------------------------------------------------------- */
/* Main                                                               */
/* ---------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.png> <output_dir> <profile:E|Q>\n", argv[0]);
        return 1;
    }

    char profile = argv[3][0];
    int q_roi          = (profile == 'E') ? 32 : 16;
    int q_bg           = (profile == 'E') ? 64 : 48;
    int subsample      = (profile == 'E') ? 4  : 2;
    float roi_threshold = (profile == 'E') ? 0.15f : 0.10f;

    struct rusage usage_start;
    struct timeval tv_start, tv_end;
    getrusage(RUSAGE_SELF, &usage_start);
    gettimeofday(&tv_start, NULL);

    /* Lecture PNG ; fallback PPM si l'extension n'est pas .png */
    Image *img = NULL;
    const char *ext = strrchr(argv[1], '.');
    if (ext && strcmp(ext, ".ppm") == 0)
        img = read_ppm(argv[1]);
    else
        img = read_png(argv[1]);

    if (!img) {
        fprintf(stderr, "Erreur lecture image : %s\n", argv[1]);
        return 1;
    }

    int bw = (img->width  + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bh = (img->height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned char *roi_mask  = (unsigned char*)calloc((size_t)bw * bh, 1);
    unsigned char *otsu_full = NULL;

    detect_roi_blocks(img, roi_mask, roi_threshold, &otsu_full);

    /* Masques sauvegardés en PNG (remplace PPM) */
    char outfile[512];
    snprintf(outfile, 512, "%s/otsu_mask.png", argv[2]);
    {
        /* otsu_full est déjà un tableau uint8 de 0/255 -- on le reformate en RGB */
        int size = img->width * img->height;
        unsigned char *rgb3 = (unsigned char*)malloc((size_t)size * 3);
        for (int i = 0; i < size; i++) {
            rgb3[i*3+0] = otsu_full[i];
            rgb3[i*3+1] = otsu_full[i];
            rgb3[i*3+2] = otsu_full[i];
        }
        size_t dummy;
        unsigned char *png_b = png_compress_memory(rgb3, img->width, img->height, &dummy);
        free(rgb3);
        FILE *fp = fopen(outfile, "wb");
        if (fp) { fwrite(png_b, 1, dummy, fp); fclose(fp); }
        free(png_b);
    }

    snprintf(outfile, 512, "%s/roi_mask.png", argv[2]);
    save_mask_png(outfile, roi_mask, img->width, img->height, bw, bh);

    /* Compression */
    unsigned char *roi_canvas;
    build_roi_canvas(img, roi_mask, bw, bh, q_roi, &roi_canvas);

    unsigned char *bg_buf; int bg_w, bg_h;
    build_background_buffer(img, q_bg, subsample, &bg_buf, &bg_w, &bg_h);

    size_t roi_png_size = 0, bg_png_size = 0;
    unsigned char *roi_png = png_compress_memory(roi_canvas, img->width, img->height, &roi_png_size);
    unsigned char *bg_png  = png_compress_memory(bg_buf, bg_w, bg_h, &bg_png_size);
    free(roi_canvas);
    free(bg_buf);

    snprintf(outfile, 512, "%s/compressed.p2", argv[2]);
    save_compressed(outfile, img->width, img->height, q_roi, q_bg, subsample,
                    roi_mask, bw, bh, roi_png, roi_png_size, bg_png, bg_png_size, bg_w, bg_h);

    printf("ROI PNG: %zu bytes (%dx%d), Fond PNG: %zu bytes (%dx%d)\n",
           roi_png_size, img->width, img->height, bg_png_size, bg_w, bg_h);

    free(roi_png);
    free(bg_png);

    /* Métriques (énergie NON rapportée — INA219 non fonctionnel au boot) */
    gettimeofday(&tv_end, NULL);
    struct rusage usage_end;
    getrusage(RUSAGE_SELF, &usage_end);

    double cpu_time = (tv_end.tv_sec  - tv_start.tv_sec)  * 1000.0 +
                      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
    long mem_kb = usage_end.ru_maxrss;

    snprintf(outfile, 512, "%s/compressed.p2", argv[2]);
    FILE *f_stat = fopen(outfile, "rb");
    fseek(f_stat, 0, SEEK_END);
    long compressed_size = ftell(f_stat);
    fclose(f_stat);

    long original_size = (long)img->width * img->height * 3;
    double ratio = (double)original_size / compressed_size;

    snprintf(outfile, 512, "%s/node_metrics.txt", argv[2]);
    FILE *fm = fopen(outfile, "w");
    fprintf(fm, "cpu_time_ms,%.3f\n", cpu_time);
    fprintf(fm, "memory_kb,%ld\n",    mem_kb);
    fprintf(fm, "compressed_bytes,%ld\n", compressed_size);
    fprintf(fm, "compression_ratio,%.2f\n", ratio);
    fprintf(fm, "roi_png_bytes,%zu\n", roi_png_size);
    fprintf(fm, "bg_png_bytes,%zu\n",  bg_png_size);
    fclose(fm);

    printf("Compression: %.2fx, Taille: %ld bytes\n", ratio, compressed_size);

    free(img->r); free(img->g); free(img->b); free(img);
    free(roi_mask);
    free(otsu_full);
    return 0;
}
