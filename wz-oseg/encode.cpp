/*
 * WZ-OSEG encoder (v3)
 * ---------------------------------------------------------------
 * Changements par rapport à v2 :
 *   - Lecture PNG en entrée (via libpng) au lieu de PPM,
 *     pour correspondre au pipeline_test.py qui travaille en PNG.
 *   - Sauvegarde des masques (otsu_mask, hue_mask) en PNG au lieu
 *     de PPM, directement lisibles par Pillow côté serveur.
 *   - Fallback PPM conservé si l'extension du fichier est .ppm.
 *   - Logique RLE + Huffman inchangée (v2).
 *   - Énergie NON rapportée (INA219 saturait à 3.2 A au boot).
 *
 * Build :
 *   gcc -O2 -Wall -o encode encode.cpp -lm -lpng
 *
 * Usage :
 *   ./encode <input.png> <output_dir>
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

#define BLOCK_SIZE     8
#define PI             3.14159265358979323846
#define ESCAPE_MARKER  ((int16_t)-32768)
#define FREQ_TABLE_SIZE 131071

typedef struct {
    int width, height;
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
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL); fclose(f); return NULL;
    }

    png_init_io(png_ptr, f);
    png_read_info(png_ptr, info_ptr);

    int width  = (int)png_get_image_width(png_ptr,  info_ptr);
    int height = (int)png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr,  info_ptr);

    /* Normalisation → RGB 8 bits */
    if (bit_depth == 16)                                          png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)                     png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)       png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))          png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGBA ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)                   png_set_strip_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)                   png_set_gray_to_rgb(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    png_bytep *rows = (png_bytep*)malloc(sizeof(png_bytep) * height);
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    for (int y = 0; y < height; y++)
        rows[y] = (png_bytep)malloc(row_bytes);
    png_read_image(png_ptr, rows);

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width; img->height = height;
    int size = width * height;
    img->r = (unsigned char*)malloc(size);
    img->g = (unsigned char*)malloc(size);
    img->b = (unsigned char*)malloc(size);

    for (int y = 0; y < height; y++) {
        png_bytep row = rows[y];
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            img->r[idx] = row[x * 3 + 0];
            img->g[idx] = row[x * 3 + 1];
            img->b[idx] = row[x * 3 + 2];
        }
        free(rows[y]);
    }
    free(rows);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(f);
    return img;
}

/* Conservé pour compatibilité tests unitaires */
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
/* Sauvegarde masque en PNG niveaux de gris                          */
/* ---------------------------------------------------------------- */

typedef struct { unsigned char *buf; size_t len, cap; } PngMemBuf;

static void png_mem_write(png_structp p, png_bytep data, png_size_t len) {
    PngMemBuf *m = (PngMemBuf*)png_get_io_ptr(p);
    if (m->len + len > m->cap) { while (m->len + len > m->cap) m->cap *= 2; m->buf = (unsigned char*)realloc(m->buf, m->cap); }
    memcpy(m->buf + m->len, data, len); m->len += len;
}
static void png_mem_flush(png_structp p) { (void)p; }

static void save_mask_png(const char *filename, unsigned char *mask, int width, int height) {
    /* mask[] : valeurs 0 ou 255, un octet par pixel */
    png_structp pw = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop   pi = png_create_info_struct(pw);
    if (setjmp(png_jmpbuf(pw))) { fprintf(stderr, "Erreur PNG mask\n"); return; }

    PngMemBuf mem;
    mem.cap = (size_t)width * height / 2 + 1024;
    mem.buf = (unsigned char*)malloc(mem.cap);
    mem.len = 0;
    png_set_write_fn(pw, &mem, png_mem_write, png_mem_flush);
    png_set_IHDR(pw, pi, width, height, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(pw, 9);
    png_write_info(pw, pi);

    png_bytep *rows = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) rows[y] = mask + y * width;
    png_write_image(pw, rows);
    png_write_end(pw, NULL);
    free(rows);
    png_destroy_write_struct(&pw, &pi);

    FILE *fp = fopen(filename, "wb");
    if (fp) { fwrite(mem.buf, 1, mem.len, fp); fclose(fp); }
    free(mem.buf);
}

/* ---------------------------------------------------------------- */
/* Segmentation (inchangée depuis v1)                                */
/* ---------------------------------------------------------------- */

unsigned char* rgb_to_gray(Image *img) {
    int size = img->width * img->height;
    unsigned char *gray = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++)
        gray[i] = (unsigned char)(0.299*img->r[i] + 0.587*img->g[i] + 0.114*img->b[i]);
    return gray;
}

int otsu_threshold(unsigned char *gray, int size) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[gray[i]]++;
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];
    float sumB = 0; int wB = 0, wF = 0; float maxVar = 0; int threshold = 0;
    for (int t = 0; t < 256; t++) {
        wB += hist[t]; if (wB == 0) continue;
        wF = size - wB; if (wF == 0) break;
        sumB += (float)(t * hist[t]);
        float mB = sumB / wB, mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; threshold = t; }
    }
    return threshold;
}

unsigned char* hue_segmentation(Image *img) {
    int size = img->width * img->height;
    unsigned char *mask = (unsigned char*)calloc(size, 1);
    for (int i = 0; i < size; i++) {
        float r = img->r[i]/255.0f, g = img->g[i]/255.0f, b = img->b[i]/255.0f;
        float cmax = fmaxf(r, fmaxf(g, b));
        float cmin = fminf(r, fminf(g, b));
        float delta = cmax - cmin;
        float h = 0;
        if (delta > 0.001f && cmax == g)
            h = 60.0f * (fmodf((b - r) / delta, 6.0f) + 2.0f);
        if (h >= 60.0f && h <= 180.0f) mask[i] = 255;
    }
    return mask;
}

/* ---------------------------------------------------------------- */
/* DCT (inchangée depuis v1)                                         */
/* ---------------------------------------------------------------- */

void dct_1d(float *data, int n) {
    float *temp = (float*)malloc(n * sizeof(float));
    for (int k = 0; k < n; k++) {
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += data[i] * cosf(PI * k * (2*i+1) / (2.0f*n));
        float scale = (k == 0) ? sqrtf(1.0f/n) : sqrtf(2.0f/n);
        temp[k] = scale * sum;
    }
    memcpy(data, temp, n * sizeof(float));
    free(temp);
}

int dct_compress_block(unsigned char *channel, int width, int height,
                       short **dct_out, int *total_coeffs) {
    int bw = width / BLOCK_SIZE, bh = height / BLOCK_SIZE;
    int coeffs_per_block = 0;
    for (int y = 0; y < BLOCK_SIZE; y++)
        for (int x = 0; x < BLOCK_SIZE; x++)
            if (x + y < 6) coeffs_per_block++;

    *total_coeffs = bw * bh * coeffs_per_block;
    *dct_out = (short*)calloc(*total_coeffs, sizeof(short));
    int coeff_idx = 0;

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            float block[BLOCK_SIZE][BLOCK_SIZE];
            for (int y = 0; y < BLOCK_SIZE; y++)
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    int idx = (by*BLOCK_SIZE+y)*width + (bx*BLOCK_SIZE+x);
                    block[y][x] = (float)channel[idx] - 128.0f;
                }
            for (int y = 0; y < BLOCK_SIZE; y++) dct_1d(block[y], BLOCK_SIZE);
            float col[BLOCK_SIZE];
            for (int x = 0; x < BLOCK_SIZE; x++) {
                for (int y = 0; y < BLOCK_SIZE; y++) col[y] = block[y][x];
                dct_1d(col, BLOCK_SIZE);
                for (int y = 0; y < BLOCK_SIZE; y++) block[y][x] = col[y];
            }
            for (int y = 0; y < BLOCK_SIZE; y++)
                for (int x = 0; x < BLOCK_SIZE; x++)
                    if (x + y < 6) {
                        int q = (x + y < 3) ? 8 : 16;
                        long qv = lroundf(block[y][x] / q);
                        if (qv < -30000) qv = -30000;
                        if (qv >  30000) qv =  30000;
                        (*dct_out)[coeff_idx++] = (short)qv;
                    }
        }
    }
    return coeff_idx;
}

/* ---------------------------------------------------------------- */
/* RLE + Huffman (inchangés depuis v2)                               */
/* ---------------------------------------------------------------- */

static int16_t* rle_encode(short *coeffs, int n, int *out_count) {
    int16_t *tokens = (int16_t*)malloc(sizeof(int16_t) * (2*n+4));
    int t = 0, i = 0;
    while (i < n) {
        if (coeffs[i] == 0) {
            int run = 0;
            while (i < n && coeffs[i] == 0 && run < 32000) { run++; i++; }
            tokens[t++] = ESCAPE_MARKER;
            tokens[t++] = (int16_t)run;
        } else { tokens[t++] = coeffs[i]; i++; }
    }
    *out_count = t;
    return tokens;
}

typedef struct { int16_t symbol; int count; int used; } HashSlot;
static unsigned hash_symbol(int16_t s) {
    unsigned u = (unsigned short)s;
    u = (u ^ (u >> 8)) * 2654435761u;
    return u;
}
static void hash_increment(HashSlot *table, int sz, int16_t sym) {
    unsigned idx = hash_symbol(sym) % sz;
    while (table[idx].used && table[idx].symbol != sym) idx = (idx+1) % sz;
    if (!table[idx].used) { table[idx].used=1; table[idx].symbol=sym; table[idx].count=0; }
    table[idx].count++;
}

typedef struct HNode { long freq; int16_t symbol; int is_leaf; struct HNode *left, *right; } HNode;
typedef struct { HNode **data; int size, cap; } MinHeap;
static MinHeap* heap_create(int cap) {
    MinHeap *h = (MinHeap*)malloc(sizeof(MinHeap));
    h->data = (HNode**)malloc(sizeof(HNode*)*cap); h->size=0; h->cap=cap; return h;
}
static void heap_push(MinHeap *h, HNode *n) {
    int i = h->size++; h->data[i] = n;
    while (i > 0) {
        int p = (i-1)/2;
        if (h->data[p]->freq <= h->data[i]->freq) break;
        HNode *tmp = h->data[p]; h->data[p]=h->data[i]; h->data[i]=tmp; i=p;
    }
}
static HNode* heap_pop(MinHeap *h) {
    HNode *top = h->data[0]; h->data[0] = h->data[--h->size];
    int i = 0;
    while (1) {
        int l=2*i+1, r=2*i+2, s=i;
        if (l<h->size && h->data[l]->freq < h->data[s]->freq) s=l;
        if (r<h->size && h->data[r]->freq < h->data[s]->freq) s=r;
        if (s==i) break;
        HNode *tmp=h->data[i]; h->data[i]=h->data[s]; h->data[s]=tmp; i=s;
    }
    return top;
}

static void build_huffman(HashSlot *table, int tsz,
                          int16_t **out_sym, uint8_t **out_len, int *out_n) {
    int nsym = 0;
    for (int i = 0; i < tsz; i++) if (table[i].used) nsym++;
    if (nsym == 0) { *out_n = 0; return; }
    int16_t *sym_out = (int16_t*)malloc(sizeof(int16_t)*nsym);
    uint8_t *len_out = (uint8_t*)malloc(sizeof(uint8_t)*nsym);
    if (nsym == 1) {
        for (int i = 0; i < tsz; i++) if (table[i].used) { sym_out[0]=table[i].symbol; break; }
        len_out[0]=1; *out_sym=sym_out; *out_len=len_out; *out_n=1; return;
    }
    MinHeap *h = heap_create(nsym*2);
    for (int i = 0; i < tsz; i++) if (table[i].used) {
        HNode *leaf=(HNode*)malloc(sizeof(HNode));
        leaf->freq=table[i].count; leaf->symbol=table[i].symbol;
        leaf->is_leaf=1; leaf->left=leaf->right=NULL; heap_push(h,leaf);
    }
    while (h->size > 1) {
        HNode *a=heap_pop(h), *b=heap_pop(h);
        HNode *p=(HNode*)malloc(sizeof(HNode));
        p->freq=a->freq+b->freq; p->is_leaf=0; p->left=a; p->right=b; heap_push(h,p);
    }
    HNode *root = heap_pop(h);
    typedef struct { HNode *node; int depth; } SI;
    SI *stack=(SI*)malloc(sizeof(SI)*(nsym*2+2));
    int sp=0, count=0;
    stack[sp].node=root; stack[sp].depth=0; sp++;
    while (sp > 0) {
        SI cur = stack[--sp];
        if (cur.node->is_leaf) {
            sym_out[count]=cur.node->symbol;
            len_out[count]=(uint8_t)(cur.depth==0?1:cur.depth);
            count++;
        } else {
            stack[sp].node=cur.node->left;  stack[sp].depth=cur.depth+1; sp++;
            stack[sp].node=cur.node->right; stack[sp].depth=cur.depth+1; sp++;
        }
    }
    free(stack);
    *out_sym=sym_out; *out_len=len_out; *out_n=count;
}

typedef struct { int16_t symbol; uint8_t length; uint32_t code; } CanonEntry;
static int cmp_canon(const void *a, const void *b) {
    const CanonEntry *ea=(const CanonEntry*)a, *eb=(const CanonEntry*)b;
    if (ea->length != eb->length) return (int)ea->length - (int)eb->length;
    return (int)ea->symbol - (int)eb->symbol;
}
static CanonEntry* build_canonical(int16_t *sym, uint8_t *len, int n) {
    CanonEntry *e=(CanonEntry*)malloc(sizeof(CanonEntry)*n);
    for (int i=0;i<n;i++){e[i].symbol=sym[i];e[i].length=len[i];}
    qsort(e,n,sizeof(CanonEntry),cmp_canon);
    uint32_t code=0;
    for (int i=0;i<n;i++){
        e[i].code=code; code++;
        if (i+1<n && e[i+1].length>e[i].length) code<<=(e[i+1].length-e[i].length);
    }
    return e;
}

typedef struct { int16_t symbol; uint32_t code; uint8_t length; int used; } CodeSlot;
static void code_insert(CodeSlot *t,int sz,int16_t sym,uint32_t code,uint8_t len){
    unsigned idx=hash_symbol(sym)%sz;
    while(t[idx].used&&t[idx].symbol!=sym) idx=(idx+1)%sz;
    t[idx].used=1;t[idx].symbol=sym;t[idx].code=code;t[idx].length=len;
}
static CodeSlot* code_find(CodeSlot *t,int sz,int16_t sym){
    unsigned idx=hash_symbol(sym)%sz;
    while(t[idx].used){if(t[idx].symbol==sym)return &t[idx];idx=(idx+1)%sz;}
    return NULL;
}

typedef struct { uint8_t *buf; int cap,byte_pos,bit_pos; } BitWriter;
static BitWriter* bw_create(int cap){
    BitWriter *bw=(BitWriter*)malloc(sizeof(BitWriter));
    bw->buf=(uint8_t*)calloc(cap>0?cap:1,1);bw->cap=cap>0?cap:1;
    bw->byte_pos=0;bw->bit_pos=0;return bw;
}
static void bw_ensure(BitWriter *bw,int extra){
    if(bw->byte_pos+extra>=bw->cap){bw->cap=(bw->cap+extra)*2;bw->buf=(uint8_t*)realloc(bw->buf,bw->cap);}
}
static void bw_write(BitWriter *bw,uint32_t code,int len){
    bw_ensure(bw,(len/8)+2);
    for(int i=len-1;i>=0;i--){
        int bit=(code>>i)&1;
        bw->buf[bw->byte_pos]|=(bit<<(7-bw->bit_pos));
        if(++bw->bit_pos==8){bw->bit_pos=0;bw->byte_pos++;}
    }
}
static int bw_bits(BitWriter *bw){return bw->byte_pos*8+bw->bit_pos;}
static int bw_bytes(BitWriter *bw){return bw->byte_pos+(bw->bit_pos>0?1:0);}

/* ---------------------------------------------------------------- */
/* Sortie fichier compressé (inchangée)                              */
/* ---------------------------------------------------------------- */

void save_compressed(const char *filename, Image *img,
                     unsigned char *otsu, unsigned char *hue,
                     int bw, int bh, int coeff_count,
                     CanonEntry *canon, int hnsym, int token_count,
                     int total_bits, uint8_t *packed, int packed_bytes) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "PROTO1_WZv2\n");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d %d\n", bw, bh);
    fprintf(f, "%d\n", coeff_count);
    fprintf(f, "%d\n", token_count);
    fprintf(f, "%d\n", hnsym);
    fprintf(f, "%d\n", total_bits);
    fprintf(f, "HUFFTABLE\n");
    for (int i = 0; i < hnsym; i++) {
        int16_t s = canon[i].symbol;
        uint8_t l = canon[i].length;
        fwrite(&s, sizeof(int16_t), 1, f);
        fwrite(&l, sizeof(uint8_t),  1, f);
    }
    fprintf(f, "DATA\n");
    int size = img->width * img->height;
    for (int i = 0; i < size; i += 4) fputc(otsu[i], f);
    for (int i = 0; i < size; i += 4) fputc(hue[i],  f);
    fwrite(packed, 1, packed_bytes, f);
    fclose(f);
}

/* ---------------------------------------------------------------- */
/* Main                                                               */
/* ---------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input.png> <output_dir>\n", argv[0]);
        return 1;
    }

    struct rusage usage_end;
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    /* Lecture PNG ; fallback PPM si extension .ppm */
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

    int size = img->width * img->height;

    /* Segmentation */
    unsigned char *gray       = rgb_to_gray(img);
    int            threshold  = otsu_threshold(gray, size);
    unsigned char *otsu_mask  = (unsigned char*)malloc(size);
    for (int i = 0; i < size; i++)
        otsu_mask[i] = (gray[i] > threshold) ? 255 : 0;
    unsigned char *hue_mask = hue_segmentation(img);

    /* Masques sauvegardés en PNG (remplace PPM) */
    char outfile[512];
    snprintf(outfile, 512, "%s/otsu_mask.png", argv[2]);
    save_mask_png(outfile, otsu_mask, img->width, img->height);
    snprintf(outfile, 512, "%s/hue_mask.png", argv[2]);
    save_mask_png(outfile, hue_mask, img->width, img->height);

    /* DCT */
    int bw = img->width / BLOCK_SIZE;
    int bh = img->height / BLOCK_SIZE;
    short *dct_y; int coeff_count;
    dct_compress_block(gray, img->width, img->height, &dct_y, &coeff_count);
    free(gray);

    /* RLE + Huffman */
    int token_count;
    int16_t *tokens = rle_encode(dct_y, coeff_count, &token_count);
    free(dct_y);

    HashSlot *freq_table = (HashSlot*)calloc(FREQ_TABLE_SIZE, sizeof(HashSlot));
    for (int i = 0; i < token_count; i++)
        hash_increment(freq_table, FREQ_TABLE_SIZE, tokens[i]);

    int16_t *hsym = NULL; uint8_t *hlen = NULL; int hnsym = 0;
    build_huffman(freq_table, FREQ_TABLE_SIZE, &hsym, &hlen, &hnsym);
    free(freq_table);

    CanonEntry *canon = build_canonical(hsym, hlen, hnsym);
    free(hsym); free(hlen);

    int ctsz = hnsym * 4 + 17;
    CodeSlot *ctable = (CodeSlot*)calloc(ctsz, sizeof(CodeSlot));
    for (int i = 0; i < hnsym; i++)
        code_insert(ctable, ctsz, canon[i].symbol, canon[i].code, canon[i].length);

    BitWriter *bwr = bw_create(token_count + 16);
    for (int i = 0; i < token_count; i++) {
        CodeSlot *cs = code_find(ctable, ctsz, tokens[i]);
        bw_write(bwr, cs->code, cs->length);
    }
    free(ctable); free(tokens);

    int total_bits   = bw_bits(bwr);
    int packed_bytes = bw_bytes(bwr);

    snprintf(outfile, 512, "%s/compressed.p1", argv[2]);
    save_compressed(outfile, img, otsu_mask, hue_mask, bw, bh, coeff_count,
                    canon, hnsym, token_count, total_bits, bwr->buf, packed_bytes);

    free(canon); free(bwr->buf); free(bwr);

    /* Métriques (énergie NON rapportée) */
    gettimeofday(&tv_end, NULL);
    getrusage(RUSAGE_SELF, &usage_end);
    double cpu_time = (tv_end.tv_sec  - tv_start.tv_sec)  * 1000.0 +
                      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
    long mem_kb = usage_end.ru_maxrss;

    FILE *fs = fopen(outfile, "rb");
    fseek(fs, 0, SEEK_END);
    long compressed_size = ftell(fs);
    fclose(fs);

    long original_size = (long)size * 3;
    double ratio = (double)original_size / compressed_size;

    snprintf(outfile, 512, "%s/node_metrics.txt", argv[2]);
    FILE *fm = fopen(outfile, "w");
    fprintf(fm, "cpu_time_ms,%.3f\n",       cpu_time);
    fprintf(fm, "memory_kb,%ld\n",           mem_kb);
    fprintf(fm, "compressed_bytes,%ld\n",    compressed_size);
    fprintf(fm, "compression_ratio,%.2f\n",  ratio);
    fprintf(fm, "huffman_symbols,%d\n",      hnsym);
    fclose(fm);

    printf("Compression: %.2fx, Taille: %ld bytes (%d symboles Huffman)\n",
           ratio, compressed_size, hnsym);

    free(img->r); free(img->g); free(img->b); free(img);
    free(otsu_mask); free(hue_mask);
    return 0;
}
