/*
 * WZ-OSEG encoder (corrected v2)
 * ---------------------------------------------------------------
 * Fixes reviewer concern III.1 / Table 4 gap: the original prototype
 * stored the 21 retained low-frequency DCT coefficients per 8x8 block
 * with NO entropy coding, which is why the measured ratio (2.6x) fell
 * far short of the algorithmic Wyner-Ziv benchmark (12x).
 *
 * This version adds:
 *   1) Run-length encoding (RLE) of zero-valued coefficients.
 *   2) A canonical Huffman code built on the resulting token stream.
 *
 * The coefficient VALUES themselves (post-quantization) are unchanged,
 * so reconstruction quality is identical to v1 -- only the serialization
 * is more compact. Expected gain: ~3-4x over v1, per the corrective
 * roadmap already stated in the paper (Sec. 5.2).
 *
 * Build:
 *   gcc -O2 -Wall -o encode encode.cpp -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>

#define BLOCK_SIZE 8
#define PI 3.14159265358979323846
#define ESCAPE_MARKER ((int16_t)-32768)
#define FREQ_TABLE_SIZE 131071  /* prime, open-addressing hash table */

typedef struct {
    int width;
    int height;
    unsigned char *r, *g, *b;
} Image;

/* ---------------------------------------------------------------- */
/* I/O                                                                */
/* ---------------------------------------------------------------- */

Image* read_ppm(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    char magic[3];
    fscanf(f, "%2s", magic);
    if (strcmp(magic, "P6") != 0) {
        fclose(f);
        return NULL;
    }

    Image *img = (Image*)malloc(sizeof(Image));
    fscanf(f, "%d %d", &img->width, &img->height);
    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f);

    int size = img->width * img->height;
    img->r = (unsigned char*)malloc(size);
    img->g = (unsigned char*)malloc(size);
    img->b = (unsigned char*)malloc(size);

    for (int i = 0; i < size; i++) {
        img->r[i] = fgetc(f);
        img->g[i] = fgetc(f);
        img->b[i] = fgetc(f);
    }

    fclose(f);
    return img;
}

void save_ppm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        fputc(data[i], f);
        fputc(data[i], f);
        fputc(data[i], f);
    }
    fclose(f);
}

/* ---------------------------------------------------------------- */
/* Segmentation (unchanged from v1)                                  */
/* ---------------------------------------------------------------- */

unsigned char* rgb_to_gray(Image *img) {
    int size = img->width * img->height;
    unsigned char *gray = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++) {
        gray[i] = (unsigned char)(0.299 * img->r[i] + 0.587 * img->g[i] + 0.114 * img->b[i]);
    }
    return gray;
}

int otsu_threshold(unsigned char *gray, int size) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[gray[i]]++;

    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];

    float sumB = 0;
    int wB = 0, wF = 0;
    float maxVar = 0;
    int threshold = 0;

    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        wF = size - wB;
        if (wF == 0) break;

        sumB += (float)(t * hist[t]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        if (varBetween > maxVar) {
            maxVar = varBetween;
            threshold = t;
        }
    }
    return threshold;
}

unsigned char* hue_segmentation(Image *img) {
    int size = img->width * img->height;
    unsigned char *mask = (unsigned char*)calloc(size, 1);

    for (int i = 0; i < size; i++) {
        float r = img->r[i] / 255.0f;
        float g = img->g[i] / 255.0f;
        float b = img->b[i] / 255.0f;

        float cmax = fmaxf(r, fmaxf(g, b));
        float cmin = fminf(r, fminf(g, b));
        float delta = cmax - cmin;

        float h = 0;
        if (delta > 0.001f && cmax == g) {
            h = 60.0f * (fmodf((b - r) / delta, 6.0f) + 2.0f);
        }

        if (h >= 60.0f && h <= 180.0f) mask[i] = 255;
    }
    return mask;
}

/* ---------------------------------------------------------------- */
/* DCT (unchanged from v1)                                           */
/* ---------------------------------------------------------------- */

void dct_1d(float *data, int n) {
    float *temp = (float*)malloc(n * sizeof(float));
    for (int k = 0; k < n; k++) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i] * cosf(PI * k * (2 * i + 1) / (2.0f * n));
        }
        float scale = (k == 0) ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
        temp[k] = scale * sum;
    }
    memcpy(data, temp, n * sizeof(float));
    free(temp);
}

int dct_compress_block(unsigned char *channel, int width, int height, short **dct_out, int *total_coeffs) {
    int bw = width / BLOCK_SIZE;
    int bh = height / BLOCK_SIZE;
    int total_blocks = bw * bh;

    int coeffs_per_block = 0;
    for (int y = 0; y < BLOCK_SIZE; y++)
        for (int x = 0; x < BLOCK_SIZE; x++)
            if (x + y < 6) coeffs_per_block++;

    *total_coeffs = total_blocks * coeffs_per_block;
    *dct_out = (short*)calloc(*total_coeffs, sizeof(short));
    int coeff_idx = 0;

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            float block[BLOCK_SIZE][BLOCK_SIZE];

            for (int y = 0; y < BLOCK_SIZE; y++)
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    int idx = (by * BLOCK_SIZE + y) * width + (bx * BLOCK_SIZE + x);
                    block[y][x] = (float)channel[idx] - 128.0f;
                }

            for (int y = 0; y < BLOCK_SIZE; y++) dct_1d(block[y], BLOCK_SIZE);
            float col[BLOCK_SIZE];
            for (int x = 0; x < BLOCK_SIZE; x++) {
                for (int y = 0; y < BLOCK_SIZE; y++) col[y] = block[y][x];
                dct_1d(col, BLOCK_SIZE);
                for (int y = 0; y < BLOCK_SIZE; y++) block[y][x] = col[y];
            }

            for (int y = 0; y < BLOCK_SIZE; y++) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    if (x + y < 6) {
                        int q = (x + y < 3) ? 8 : 16;
                        long qv = lroundf(block[y][x] / q);
                        /* Defensive clamp: keep well clear of ESCAPE_MARKER (-32768)
                           and int16 range; quantized DCT values never approach this
                           in practice, this is a pure safety margin. */
                        if (qv < -30000) qv = -30000;
                        if (qv > 30000) qv = 30000;
                        (*dct_out)[coeff_idx++] = (short)qv;
                    }
                }
            }
        }
    }

    return coeff_idx;
}

/* ---------------------------------------------------------------- */
/* RLE                                                                */
/* ---------------------------------------------------------------- */

/* Token stream: nonzero coefficients pass through unchanged; runs of
   zeros (any length >=1) become [ESCAPE_MARKER, run_length]. */
static int16_t* rle_encode(short *coeffs, int n, int *out_count) {
    int16_t *tokens = (int16_t*)malloc(sizeof(int16_t) * (2 * n + 4));
    int t = 0, i = 0;
    while (i < n) {
        if (coeffs[i] == 0) {
            int run = 0;
            while (i < n && coeffs[i] == 0 && run < 32000) { run++; i++; }
            tokens[t++] = ESCAPE_MARKER;
            tokens[t++] = (int16_t)run;
        } else {
            tokens[t++] = coeffs[i];
            i++;
        }
    }
    *out_count = t;
    return tokens;
}

/* ---------------------------------------------------------------- */
/* Frequency hash table                                              */
/* ---------------------------------------------------------------- */

typedef struct { int16_t symbol; int count; int used; } HashSlot;

static unsigned hash_symbol(int16_t s) {
    unsigned u = (unsigned short)s;
    u = (u ^ (u >> 8)) * 2654435761u;
    return u;
}

static void hash_increment(HashSlot *table, int table_size, int16_t sym) {
    unsigned idx = hash_symbol(sym) % table_size;
    while (table[idx].used && table[idx].symbol != sym) idx = (idx + 1) % table_size;
    if (!table[idx].used) {
        table[idx].used = 1;
        table[idx].symbol = sym;
        table[idx].count = 0;
    }
    table[idx].count++;
}

/* ---------------------------------------------------------------- */
/* Huffman tree                                                      */
/* ---------------------------------------------------------------- */

typedef struct HNode {
    long freq;
    int16_t symbol;
    int is_leaf;
    struct HNode *left, *right;
} HNode;

typedef struct { HNode **data; int size, cap; } MinHeap;

static MinHeap* heap_create(int cap) {
    MinHeap *h = (MinHeap*)malloc(sizeof(MinHeap));
    h->data = (HNode**)malloc(sizeof(HNode*) * cap);
    h->size = 0; h->cap = cap;
    return h;
}
static void heap_push(MinHeap *h, HNode *n) {
    int i = h->size++;
    h->data[i] = n;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p]->freq <= h->data[i]->freq) break;
        HNode *tmp = h->data[p]; h->data[p] = h->data[i]; h->data[i] = tmp;
        i = p;
    }
}
static HNode* heap_pop(MinHeap *h) {
    HNode *top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < h->size && h->data[l]->freq < h->data[smallest]->freq) smallest = l;
        if (r < h->size && h->data[r]->freq < h->data[smallest]->freq) smallest = r;
        if (smallest == i) break;
        HNode *tmp = h->data[i]; h->data[i] = h->data[smallest]; h->data[smallest] = tmp;
        i = smallest;
    }
    return top;
}

/* Builds Huffman code lengths (not codes yet -- canonicalized later). */
static void build_huffman(HashSlot *table, int table_size, int16_t **out_symbols, uint8_t **out_lengths, int *out_nsym) {
    int nsym = 0;
    for (int i = 0; i < table_size; i++) if (table[i].used) nsym++;

    if (nsym == 0) { *out_nsym = 0; return; }

    int16_t *sym_out = (int16_t*)malloc(sizeof(int16_t) * nsym);
    uint8_t *len_out = (uint8_t*)malloc(sizeof(uint8_t) * nsym);

    if (nsym == 1) {
        for (int i = 0; i < table_size; i++) {
            if (table[i].used) { sym_out[0] = table[i].symbol; break; }
        }
        len_out[0] = 1;
        *out_symbols = sym_out;
        *out_lengths = len_out;
        *out_nsym = 1;
        return;
    }

    MinHeap *h = heap_create(nsym * 2);
    for (int i = 0; i < table_size; i++) {
        if (table[i].used) {
            HNode *leaf = (HNode*)malloc(sizeof(HNode));
            leaf->freq = table[i].count;
            leaf->symbol = table[i].symbol;
            leaf->is_leaf = 1;
            leaf->left = leaf->right = NULL;
            heap_push(h, leaf);
        }
    }

    while (h->size > 1) {
        HNode *a = heap_pop(h);
        HNode *b = heap_pop(h);
        HNode *parent = (HNode*)malloc(sizeof(HNode));
        parent->freq = a->freq + b->freq;
        parent->is_leaf = 0;
        parent->left = a;
        parent->right = b;
        heap_push(h, parent);
    }
    HNode *root = heap_pop(h);

    typedef struct { HNode *node; int depth; } StackItem;
    StackItem *stack = (StackItem*)malloc(sizeof(StackItem) * (nsym * 2 + 2));
    int sp = 0;
    stack[sp].node = root; stack[sp].depth = 0; sp++;
    int count = 0;
    while (sp > 0) {
        StackItem cur = stack[--sp];
        if (cur.node->is_leaf) {
            sym_out[count] = cur.node->symbol;
            len_out[count] = (uint8_t)(cur.depth == 0 ? 1 : cur.depth);
            count++;
        } else {
            stack[sp].node = cur.node->left;  stack[sp].depth = cur.depth + 1; sp++;
            stack[sp].node = cur.node->right; stack[sp].depth = cur.depth + 1; sp++;
        }
    }
    free(stack);

    *out_symbols = sym_out;
    *out_lengths = len_out;
    *out_nsym = count;
}

/* ---------------------------------------------------------------- */
/* Canonical Huffman codes                                           */
/* ---------------------------------------------------------------- */

typedef struct { int16_t symbol; uint8_t length; uint32_t code; } CanonEntry;

static int cmp_canon(const void *a, const void *b) {
    const CanonEntry *ea = (const CanonEntry*)a, *eb = (const CanonEntry*)b;
    if (ea->length != eb->length) return (int)ea->length - (int)eb->length;
    return (int)ea->symbol - (int)eb->symbol;
}

static CanonEntry* build_canonical(int16_t *symbols, uint8_t *lengths, int nsym) {
    CanonEntry *entries = (CanonEntry*)malloc(sizeof(CanonEntry) * nsym);
    for (int i = 0; i < nsym; i++) {
        entries[i].symbol = symbols[i];
        entries[i].length = lengths[i];
    }
    qsort(entries, nsym, sizeof(CanonEntry), cmp_canon);

    uint32_t code = 0;
    for (int i = 0; i < nsym; i++) {
        entries[i].code = code;
        code++;
        if (i + 1 < nsym && entries[i+1].length > entries[i].length) {
            code <<= (entries[i+1].length - entries[i].length);
        }
    }
    return entries;
}

typedef struct { int16_t symbol; uint32_t code; uint8_t length; int used; } CodeSlot;

static void code_table_insert(CodeSlot *table, int table_size, int16_t sym, uint32_t code, uint8_t length) {
    unsigned idx = hash_symbol(sym) % table_size;
    while (table[idx].used && table[idx].symbol != sym) idx = (idx + 1) % table_size;
    table[idx].used = 1;
    table[idx].symbol = sym;
    table[idx].code = code;
    table[idx].length = length;
}
static CodeSlot* code_table_find(CodeSlot *table, int table_size, int16_t sym) {
    unsigned idx = hash_symbol(sym) % table_size;
    while (table[idx].used) {
        if (table[idx].symbol == sym) return &table[idx];
        idx = (idx + 1) % table_size;
    }
    return NULL;
}

/* ---------------------------------------------------------------- */
/* Bit writer (MSB first)                                            */
/* ---------------------------------------------------------------- */

typedef struct { uint8_t *buf; int cap, byte_pos, bit_pos; } BitWriter;

static BitWriter* bw_create(int cap) {
    BitWriter *bw = (BitWriter*)malloc(sizeof(BitWriter));
    bw->buf = (uint8_t*)calloc(cap > 0 ? cap : 1, 1);
    bw->cap = cap > 0 ? cap : 1;
    bw->byte_pos = 0;
    bw->bit_pos = 0;
    return bw;
}
static void bw_ensure(BitWriter *bw, int extra_bytes) {
    if (bw->byte_pos + extra_bytes >= bw->cap) {
        bw->cap = (bw->cap + extra_bytes) * 2;
        bw->buf = (uint8_t*)realloc(bw->buf, bw->cap);
    }
}
static void bw_write_bits(BitWriter *bw, uint32_t code, int length) {
    bw_ensure(bw, (length / 8) + 2);
    for (int i = length - 1; i >= 0; i--) {
        int bit = (code >> i) & 1;
        bw->buf[bw->byte_pos] |= (bit << (7 - bw->bit_pos));
        bw->bit_pos++;
        if (bw->bit_pos == 8) { bw->bit_pos = 0; bw->byte_pos++; }
    }
}
static int bw_total_bits(BitWriter *bw)  { return bw->byte_pos * 8 + bw->bit_pos; }
static int bw_total_bytes(BitWriter *bw) { return bw->byte_pos + (bw->bit_pos > 0 ? 1 : 0); }

/* ---------------------------------------------------------------- */
/* File output                                                       */
/* ---------------------------------------------------------------- */

void save_compressed(const char *filename, Image *img, unsigned char *otsu_mask,
                     unsigned char *hue_mask, int bw_blocks, int bh_blocks, int coeff_count,
                     CanonEntry *canon, int hnsym, int token_count, int total_bits,
                     uint8_t *packed_bits, int packed_bytes) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "PROTO1_WZv2\n");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d %d\n", bw_blocks, bh_blocks);
    fprintf(f, "%d\n", coeff_count);
    fprintf(f, "%d\n", token_count);
    fprintf(f, "%d\n", hnsym);
    fprintf(f, "%d\n", total_bits);
    fprintf(f, "HUFFTABLE\n");
    for (int i = 0; i < hnsym; i++) {
        int16_t s = canon[i].symbol;
        uint8_t l = canon[i].length;
        fwrite(&s, sizeof(int16_t), 1, f);
        fwrite(&l, sizeof(uint8_t), 1, f);
    }
    fprintf(f, "DATA\n");

    int size = img->width * img->height;
    for (int i = 0; i < size; i += 4) fputc(otsu_mask[i], f);
    for (int i = 0; i < size; i += 4) fputc(hue_mask[i], f);

    fwrite(packed_bits, 1, packed_bytes, f);

    fclose(f);
}

/* ---------------------------------------------------------------- */
/* Main                                                               */
/* ---------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input.ppm> <output_dir>\n", argv[0]);
        return 1;
    }

    struct rusage usage_start, usage_end;
    struct timeval tv_start, tv_end;

    getrusage(RUSAGE_SELF, &usage_start);
    gettimeofday(&tv_start, NULL);

    Image *img = read_ppm(argv[1]);
    if (!img) {
        fprintf(stderr, "Erreur lecture image\n");
        return 1;
    }

    int size = img->width * img->height;

    /* Segmentation (unchanged) */
    unsigned char *gray = rgb_to_gray(img);
    int threshold = otsu_threshold(gray, size);
    unsigned char *otsu_mask = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++) otsu_mask[i] = (gray[i] > threshold) ? 255 : 0;
    unsigned char *hue_mask = hue_segmentation(img);

    char outfile[512];
    snprintf(outfile, 512, "%s/otsu_mask.ppm", argv[2]);
    save_ppm(outfile, otsu_mask, img->width, img->height);
    snprintf(outfile, 512, "%s/hue_mask.ppm", argv[2]);
    save_ppm(outfile, hue_mask, img->width, img->height);

    /* DCT (unchanged) */
    int bw = img->width / BLOCK_SIZE;
    int bh = img->height / BLOCK_SIZE;
    short *dct_y;
    int coeff_count;
    dct_compress_block(gray, img->width, img->height, &dct_y, &coeff_count);
    free(gray);

    /* --- NEW: RLE + canonical Huffman entropy coding --- */
    int token_count;
    int16_t *tokens = rle_encode(dct_y, coeff_count, &token_count);
    free(dct_y);

    HashSlot *freq_table = (HashSlot*)calloc(FREQ_TABLE_SIZE, sizeof(HashSlot));
    for (int i = 0; i < token_count; i++) hash_increment(freq_table, FREQ_TABLE_SIZE, tokens[i]);

    int16_t *hsymbols = NULL; uint8_t *hlengths = NULL; int hnsym = 0;
    build_huffman(freq_table, FREQ_TABLE_SIZE, &hsymbols, &hlengths, &hnsym);
    free(freq_table);

    CanonEntry *canon = build_canonical(hsymbols, hlengths, hnsym);
    free(hsymbols); free(hlengths);

    int code_table_size = hnsym * 4 + 17;
    CodeSlot *code_table = (CodeSlot*)calloc(code_table_size, sizeof(CodeSlot));
    for (int i = 0; i < hnsym; i++)
        code_table_insert(code_table, code_table_size, canon[i].symbol, canon[i].code, canon[i].length);

    BitWriter *bwr = bw_create(token_count + 16);
    for (int i = 0; i < token_count; i++) {
        CodeSlot *cs = code_table_find(code_table, code_table_size, tokens[i]);
        bw_write_bits(bwr, cs->code, cs->length);
    }
    free(code_table);
    free(tokens);

    int total_bits = bw_total_bits(bwr);
    int packed_bytes = bw_total_bytes(bwr);

    /* Save */
    snprintf(outfile, 512, "%s/compressed.p1", argv[2]);
    save_compressed(outfile, img, otsu_mask, hue_mask, bw, bh, coeff_count,
                     canon, hnsym, token_count, total_bits, bwr->buf, packed_bytes);

    free(canon);
    free(bwr->buf);
    free(bwr);

    /* Metrics (energy_mj deliberately NOT reported -- see README) */
    gettimeofday(&tv_end, NULL);
    getrusage(RUSAGE_SELF, &usage_end);

    double cpu_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 +
                      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
    long mem_kb = usage_end.ru_maxrss;

    FILE *f_stat = fopen(outfile, "rb");
    fseek(f_stat, 0, SEEK_END);
    long compressed_size = ftell(f_stat);
    fclose(f_stat);

    long original_size = size * 3;
    double ratio = (double)original_size / compressed_size;

    snprintf(outfile, 512, "%s/node_metrics.txt", argv[2]);
    FILE *fm = fopen(outfile, "w");
    fprintf(fm, "cpu_time_ms,%.3f\n", cpu_time);
    fprintf(fm, "memory_kb,%ld\n", mem_kb);
    fprintf(fm, "compressed_bytes,%ld\n", compressed_size);
    fprintf(fm, "compression_ratio,%.2f\n", ratio);
    fprintf(fm, "huffman_symbols,%d\n", hnsym);
    fclose(fm);

    printf("Compression: %.2fx, Taille: %ld bytes (%d symboles Huffman)\n", ratio, compressed_size, hnsym);

    free(img->r); free(img->g); free(img->b); free(img);
    free(otsu_mask);
    free(hue_mask);

    return 0;
}
