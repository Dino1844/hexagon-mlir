/* Test-only protocol driver: runs the REAL HMXLayout.c on the host.
 *
 * Consumed by: test_hmx_layout_permutation_contract.py, which compiles this
 * file with the stub SDK headers in this directory first on the include path.
 * The production file bin/runtime/hmx/src/HMXLayout.c is pulled in verbatim
 * below -- its offset tables, its control flow, its intrinsic call sites are
 * the code under test; this driver contains no layout knowledge whatsoever.
 *
 * It is a pure executor: Python generates every case's buffers and expected
 * bytes from an independent closed-form model, streams the case over stdin,
 * and reads the destination buffer back. The wire protocol (one command or
 * reply per line):
 *
 *   > case <id> <poison-hex>     poison-fill all three buffers
 *   > fill S|R <off> <hex>       write hex bytes at offset (even length)
 *   > run <leaf> <args...>       call the leaf; buffer args are OFFSETS
 *   > dump D <off> <len>         reply: D <off> <hex>
 *   > quit
 *
 * Replies: one banner line first, then exactly one dump line per executed
 * dump, in order. Any protocol or bounds error prints "ERR <message>" and
 * exits 2 so the Python side fails loudly.
 *
 * Why the buffers are mapped below 2 GB: every leaf takes buffer arguments
 * as plain `unsigned` (32-bit) addresses, exactly like the device, and does
 * its address arithmetic in 32 bits. On the host that is only faithful if
 * the real addresses fit in 32 bits, so the driver maps its three buffers
 * with MAP_32BIT and refuses to start otherwise. The addresses are printed
 * in the banner so the Python side can verify them too.
 *
 * HMX_EMU_LAYOUT_C (optional): compile the driver against a REPLACEMENT copy
 * of HMXLayout.c instead of the real one. This exists only so the test's
 * negative controls (injecting a known-wrong table into a scratch copy and
 * asserting the battery goes red) can share this driver verbatim.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0 /* pre-4.17: plain hint; the range check rejects it */
#endif

#ifdef HMX_EMU_LAYOUT_C
#include HMX_EMU_LAYOUT_C
#else
#include "../../hmx/src/HMXLayout.c"
#endif

/* The stub headers must have won the include race, not a real SDK header. */
#if !defined(HMX_HOST_EMU_HEXAGON_TYPES_STUB)
#error "HMXLayout.c was compiled against a non-stub <hexagon_types.h>"
#endif

#define BUF_SIZE (4u << 20) /* 4 MiB per buffer; every case fits well inside */

static unsigned char *bufS, *bufD, *bufR;

static unsigned char *map_low(size_t n) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (p == MAP_FAILED) {
        /* Fallback for kernels where MAP_32BIT is unavailable: claim a
         * fixed low address that a PIE test binary does not use. */
        p = mmap((void *)0x10000000, n, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    }
    if (p == MAP_FAILED)
        return NULL;
    if ((uintptr_t)p >= (1ull << 31) || (uintptr_t)p + n >= (1ull << 31))
        return NULL;
    return (unsigned char *)p;
}

/* ------------------------------------------------------------------ */
/* leaf dispatch: signature specs mirror HMXAPI.h's declarations.        */
/*   D/S/R = offset into the destination/source/residual buffer         */
/*   -     = plain integer argument                                     */

typedef void (*leaf_fn)(unsigned *a);

static void w_pack_act_f16(unsigned *a) { hmx_pack_act_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_weight_f16(unsigned *a) { hmx_pack_weight_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_act_f16_bulk(unsigned *a) { hmx_pack_act_f16_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_weight_f16_bulk(unsigned *a) { hmx_pack_weight_f16_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_act_f32(unsigned *a) { hmx_pack_act_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_weight_f32(unsigned *a) { hmx_pack_weight_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_act_f32_bulk(unsigned *a) { hmx_pack_act_f32_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_weight_f32_bulk(unsigned *a) { hmx_pack_weight_f32_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_weight_f16_T(unsigned *a) { hmx_pack_weight_f16_T(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_weight_f16_T_bulk(unsigned *a) { hmx_pack_weight_f16_T_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_weight_f32_T(unsigned *a) { hmx_pack_weight_f32_T(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_pack_weight_f32_T_bulk(unsigned *a) { hmx_pack_weight_f32_T_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }
static void w_pack_act_tail_f16(unsigned *a) { hmx_pack_act_tail_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_pack_weight_tail_f16(unsigned *a) { hmx_pack_weight_tail_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_pack_act_tail_f32(unsigned *a) { hmx_pack_act_tail_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_pack_weight_tail_f32(unsigned *a) { hmx_pack_weight_tail_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_pack_weight_tail_f16_T(unsigned *a) { hmx_pack_weight_tail_f16_T(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_pack_weight_tail_f32_T(unsigned *a) { hmx_pack_weight_tail_f32_T(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_unpack_acc_f16(unsigned *a) { hmx_unpack_acc_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_unpack_acc_f16_bulk(unsigned *a) { hmx_unpack_acc_f16_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
static void w_unpack_acc_f32(unsigned *a) { hmx_unpack_acc_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); }
static void w_unpack_acc_f32_bulk(unsigned *a) { hmx_unpack_acc_f32_bulk(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); }
static void w_unpack_acc_tail_f16(unsigned *a) { hmx_unpack_acc_tail_f16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
static void w_unpack_acc_tail_f32(unsigned *a) { hmx_unpack_acc_tail_f32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); }

static const struct {
    const char *name;
    const char *spec;
    leaf_fn fn;
} LEAVES[] = {
    {"pack_act_f16", "DS-----", w_pack_act_f16},
    {"pack_weight_f16", "DS-----", w_pack_weight_f16},
    {"pack_act_f16_bulk", "DS------", w_pack_act_f16_bulk},
    {"pack_weight_f16_bulk", "DS------", w_pack_weight_f16_bulk},
    {"pack_act_f32", "DS-----", w_pack_act_f32},
    {"pack_weight_f32", "DS-----", w_pack_weight_f32},
    {"pack_act_f32_bulk", "DS------", w_pack_act_f32_bulk},
    {"pack_weight_f32_bulk", "DS------", w_pack_weight_f32_bulk},
    {"pack_weight_f16_T", "DS-----", w_pack_weight_f16_T},
    {"pack_weight_f16_T_bulk", "DS------", w_pack_weight_f16_T_bulk},
    {"pack_weight_f32_T", "DS-----", w_pack_weight_f32_T},
    {"pack_weight_f32_T_bulk", "DS------", w_pack_weight_f32_T_bulk},
    {"pack_act_tail_f16", "DS-------", w_pack_act_tail_f16},
    {"pack_weight_tail_f16", "DS-------", w_pack_weight_tail_f16},
    {"pack_act_tail_f32", "DS-------", w_pack_act_tail_f32},
    {"pack_weight_tail_f32", "DS-------", w_pack_weight_tail_f32},
    {"pack_weight_tail_f16_T", "DS-------", w_pack_weight_tail_f16_T},
    {"pack_weight_tail_f32_T", "DS-------", w_pack_weight_tail_f32_T},
    {"unpack_acc_f16", "DS-----", w_unpack_acc_f16},
    {"unpack_acc_f16_bulk", "DS-----", w_unpack_acc_f16_bulk},
    {"unpack_acc_f32", "DR-S------", w_unpack_acc_f32},
    {"unpack_acc_f32_bulk", "DR-S------", w_unpack_acc_f32_bulk},
    {"unpack_acc_tail_f16", "DS-------", w_unpack_acc_tail_f16},
    {"unpack_acc_tail_f32", "DR-S--------", w_unpack_acc_tail_f32},
};

#define MAX_ARGS 12
#define N_LEAVES ((int)(sizeof LEAVES / sizeof LEAVES[0]))

static void fail(const char *msg) {
    printf("ERR %s\n", msg);
    fflush(stdout);
    exit(2);
}

static unsigned char *buf_of(char c) {
    switch (c) {
    case 'S': return bufS;
    case 'D': return bufD;
    case 'R': return bufR;
    default: fail("bad buffer name"); return NULL;
    }
}

static int hexval(int ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

/* Longest line: a fill command carrying ~1 MiB of hex. */
static char line[2 << 20];
/* Dump encode buffer: 4 MiB of bytes -> 8 MiB of hex. */
static char hexout[8 + (BUF_SIZE * 2) + 4];

/* All leaf execution runs on a worker thread whose STACK is mapped below
 * 2 GB. Why: the bounds-safe tail leaves stage their scratch on the C stack
 * and pass its address to the inner leaf as a plain `unsigned` (32-bit),
 * which is lossless on Hexagon (a 32-bit architecture) but would truncate a
 * normal host stack address (0x7f...) into a wild pointer. Mapping the
 * worker stack low makes every address the production code handles --
 * buffer or stack -- genuinely 32-bit, exactly like the device. */
static void *worker(void *unused) {
    (void)unused;
    bufS = map_low(BUF_SIZE);
    bufD = map_low(BUF_SIZE);
    bufR = map_low(BUF_SIZE);
    if (!bufS || !bufD || !bufR)
        fail("could not map buffers below 2 GB");
    printf("emu-driver ready S=%u D=%u R=%u\n",
           (unsigned)(uintptr_t)bufS, (unsigned)(uintptr_t)bufD,
           (unsigned)(uintptr_t)bufR);
    fflush(stdout);

    while (fgets(line, sizeof line, stdin)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;

        if (strncmp(line, "case ", 5) == 0) {
            unsigned id, poison;
            if (sscanf(line + 5, "%u %x", &id, &poison) != 2)
                fail("bad case command");
            unsigned char p = (unsigned char)poison;
            memset(bufS, p, BUF_SIZE);
            memset(bufD, p, BUF_SIZE);
            memset(bufR, p, BUF_SIZE);
        } else if (strncmp(line, "fill ", 5) == 0) {
            char bname;
            unsigned off;
            int consumed = 0;
            if (sscanf(line + 5, "%c %u %n", &bname, &off, &consumed) != 2)
                fail("bad fill command");
            const char *hex = line + 5 + consumed;
            size_t nhex = strlen(hex);
            if (nhex & 1)
                fail("odd hex length");
            if (off > BUF_SIZE || (size_t)off + nhex / 2 > BUF_SIZE)
                fail("fill out of bounds");
            unsigned char *buf = buf_of(bname);
            for (size_t i = 0; i < nhex / 2; i++) {
                int hi = hexval((unsigned char)hex[2 * i]);
                int lo = hexval((unsigned char)hex[2 * i + 1]);
                if (hi < 0 || lo < 0)
                    fail("bad hex digit");
                buf[off + i] = (unsigned char)((hi << 4) | lo);
            }
        } else if (strncmp(line, "run ", 4) == 0) {
            char name[64];
            int consumed = 0;
            if (sscanf(line + 4, "%63s %n", name, &consumed) != 1)
                fail("bad run command");
            int idx = -1;
            for (int i = 0; i < N_LEAVES; i++)
                if (strcmp(LEAVES[i].name, name) == 0) { idx = i; break; }
            if (idx < 0)
                fail("unknown leaf name");
            const char *spec = LEAVES[idx].spec;
            int nargs = (int)strlen(spec);
            unsigned args[MAX_ARGS];
            if (nargs > MAX_ARGS)
                fail("spec too long");
            const char *p = line + 4 + consumed;
            for (int i = 0; i < nargs; i++) {
                int used = 0;
                unsigned v;
                if (sscanf(p, "%u %n", &v, &used) != 1)
                    fail("too few arguments");
                p += used;
                if (spec[i] == '-') {
                    args[i] = v;
                } else {
                    if (v >= BUF_SIZE)
                        fail("buffer offset out of bounds");
                    args[i] = (unsigned)((uintptr_t)buf_of(spec[i]) + v);
                }
            }
            /* no trailing junk */
            while (*p == ' ') p++;
            if (*p)
                fail("too many arguments");
            LEAVES[idx].fn(args);
        } else if (strncmp(line, "dump D ", 7) == 0) {
            unsigned off, n;
            if (sscanf(line + 7, "%u %u", &off, &n) != 2)
                fail("bad dump command");
            if (off > BUF_SIZE || n > BUF_SIZE - off)
                fail("dump out of bounds");
            int o = snprintf(hexout, sizeof hexout, "D %u ", off);
            static const char digits[] = "0123456789abcdef";
            for (unsigned i = 0; i < n; i++) {
                hexout[o + 2 * i] = digits[bufD[off + i] >> 4];
                hexout[o + 2 * i + 1] = digits[bufD[off + i] & 15];
            }
            hexout[o + 2 * n] = '\n';
            hexout[o + 2 * n + 1] = 0;
            fputs(hexout, stdout);
            fflush(stdout);
        } else if (strcmp(line, "quit") == 0) {
            break;
        } else if (len == 0) {
            /* ignore blank lines */
        } else {
            fail("unknown command");
        }
    }
    return 0;
}

int main(void) {
    size_t stk = 16u << 20;
    void *stkmem = mmap(NULL, stk, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (stkmem == MAP_FAILED)
        stkmem = mmap((void *)0x20000000, stk, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (stkmem == MAP_FAILED || (uintptr_t)stkmem + stk >= (1ull << 31)) {
        fprintf(stderr, "emu driver: could not map a low worker stack\n");
        return 2;
    }
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, stkmem, stk);
    pthread_t t;
    if (pthread_create(&t, &attr, worker, NULL) != 0) {
        fprintf(stderr, "emu driver: worker thread creation failed\n");
        return 2;
    }
    pthread_join(t, NULL);
    return 0;
}
