/* Test-only host stub of the SDK's <hvx_hexagon_protos.h>.
 *
 * Consumed by: test_hmx_layout_permutation_contract.py, which compiles the
 * REAL bin/runtime/hmx/src/HMXLayout.c on the host by putting this directory
 * first on the include path. Every intrinsic below is a scalar emulation of
 * the 15 HVX builtins HMXLayout.c uses -- nothing else -- and each one
 * documents where its semantics come from.
 *
 * Where the semantics are known from outside this repository
 * ------------------------------------------------------------------
 * The strongest evidence is that llama.cpp (an independent implementation,
 * read-only reference) uses two of these idioms in the same operand order and
 * states the same intent:
 *
 *   * ggml-hexagon/htp/hvx-base.h hvx_vec_f32_to_f16_shuff(v0, v1) is
 *     token-for-token the same idiom as HMXLayout.c's hmx__f32_pair_to_block
 *     (Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(conv(v1), conv(v0)))), and the
 *     tree-external layout oracle (exp/hmx/oracle/layout_oracle.h, itself
 *     derived from the HMX PRM figures and llama.cpp's producers) transcribes
 *     its output as "halfword 2c = row 2j, 2c+1 = row 2j+1".
 *   * ggml-hexagon/htp/flash-attn-ops.c decodes croutons with
 *     Q6_W_vdeal_VVR(second_tile, first_tile, -2) and reads row r0 from .lo,
 *     row r1 from .hi -- the same operands, order and intent as
 *     hmx__unpack_chunk.
 *
 * How this is kept honest
 * -----------------------
 * These emulations feed the production code's own tables and control flow
 * into an end-to-end battery whose expected bytes come from an independent
 * closed-form model (the test's Python side). If any emulation below were
 * wrong for the way HMXLayout.c uses it, the battery goes red; it cannot
 * quietly agree with a wrong production edit, because the expected bytes are
 * not derived from this file. What this cannot catch is documented in the
 * test's module docstring (the residual risk: emulator and production code
 * wrong in exactly compensating ways -- which the device-side numerical
 * tests would then also have to miss).
 *
 * A new HVX intrinsic appearing in HMXLayout.c fails to compile against this
 * header (undeclared), which is the intended loud failure: extending the
 * emulation is a decision that needs its own semantics write-up here.
 */
#ifndef HMX_HOST_EMU_HVX_HEXAGON_PROTOS_H
#define HMX_HOST_EMU_HVX_HEXAGON_PROTOS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The HVX types come from the companion <hexagon_types.h> stub (the real
 * SDK's hvx_hexagon_protos.h likewise builds on hexagon_types.h; HMXLayout.c
 * includes them in that order). */
#include "hexagon_types.h"

/* ------------------------------------------------------------------ */

/* All-zeros vector. */
static inline HVX_Vector Q6_V_vzero(void) {
    HVX_Vector v;
    memset(&v, 0, sizeof v);
    return v;
}

/* Predicate with the first n BYTES set (byte-granular, from byte 0).
 * Evidence: HMXLayout.c masks "the first `ncols` halfwords of each row" with
 * vsetq(2*ncols) for f16 and vsetq(4*ncols) for f32 -- two bytes per f16,
 * four per f32 -- so the unit is bytes and the origin is lane 0. Every use
 * in this file has n <= 128; larger n is refused loudly rather than wrapped
 * silently. */
static inline HVX_VectorPred Q6_Q_vsetq_R(int n) {
    HVX_VectorPred q;
    memset(&q, 0, sizeof q);
    if (n < 0 || n > 128) {
        fprintf(stderr, "hmx host emu: Q6_Q_vsetq_R(%d) outside 0..128\n", n);
        abort();
    }
    for (int i = 0; i < n; i++)
        q.lane[i] = 1;
    return q;
}

/* Per-byte select: result byte i = Q[i] ? a[i] : b[i].
 * Evidence: hmx__pack_pair keeps v0's low 64 bytes with vmux(vsetq(64), v0,
 * ror(v1, 64)) to build [row0.lo64 | row1.lo64]; the masking sites select a
 * value's first 2*ncols bytes against zero with the same operand order. */
static inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q, HVX_Vector a, HVX_Vector b) {
    unsigned char A[128], B[128], O[128];
    memcpy(A, &a, 128);
    memcpy(B, &b, 128);
    for (int i = 0; i < 128; i++)
        O[i] = q.lane[i] ? A[i] : B[i];
    HVX_Vector v;
    memcpy(&v, O, 128);
    return v;
}

/* Bitwise rotate-right of the 128-byte vector by n bytes: out[i] = in[(i+n)
 * mod 128] (the byte-string generalization of ROR: with x=0x01, k=1 bit,
 * ROR=0x80, i.e. out bit j = in bit (j+k) mod N).
 * HMXLayout.c only ever calls this with n = 64, where left and right rotation
 * coincide (the halves swap), so the direction convention is moot for this
 * file; the composed intent (ror(v1, 64).high64 == v1.low64) is what
 * hmx__pack_pair needs and what the end-to-end battery checks. */
static inline HVX_Vector Q6_V_vror_VR(HVX_Vector v, int n) {
    unsigned char A[128], O[128];
    memcpy(A, &v, 128);
    int r = ((n % 128) + 128) % 128;
    for (int i = 0; i < 128; i++)
        O[i] = A[(i + r) % 128];
    HVX_Vector o;
    memcpy(&o, O, 128);
    return o;
}

/* Halfword scatter: for each of the 64 halfword lanes i, store vals.h[i] at
 * base + offs.h[i] if the offset lies inside the region mask (a store is
 * performed iff (offset & region) == offset; region is 2^k-1 in every use,
 * bounding the scatter window). Offsets are BYTES: HMXLayout.c's tables are
 * documented in bytes ("element n ... at byte 4n") and its regions are byte
 * masks (127 = one 128 B block, 2047 = one crouton, 4095 = two croutons).
 * Every table entry in this file is non-negative and even; a negative or odd
 * target is refused loudly rather than emulated by guesswork. */
static inline void Q6_vscatter_RMVhV(size_t base, unsigned region, HVX_Vector offs, HVX_Vector vals) {
    int16_t O[64];
    uint16_t V[64];
    memcpy(O, &offs, 128);
    memcpy(V, &vals, 128);
    for (int i = 0; i < 64; i++) {
        int off = O[i];
        if (off < 0 || (off & 1)) {
            fprintf(stderr, "hmx host emu: scatter offset %d (lane %d) is negative or odd\n", off, i);
            abort();
        }
        if (((unsigned)off & region) != (unsigned)off)
            continue; /* lane outside the region: suppressed */
        memcpy((unsigned char *)(uintptr_t)base + off, &V[i], 2);
    }
}

/* Combine two vectors into a pair: SECOND operand is the low half.
 * Evidence: llama.cpp's hvx_vec_f16_to_f32_shuff recombines a pair p with
 * Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p)) -- an identity under this
 * convention -- and its hvx_vec_f32_to_f16_shuff(v0, v1) passes
 * vcombine(conv(v1), conv(v0)) (the same argument order as HMXLayout.c's
 * hmx__f32_pair_to_block) to produce halfword 2c from v0, which requires
 * conv(v0) to land in .lo. */
static inline HVX_VectorPair Q6_W_vcombine_VV(HVX_Vector hi, HVX_Vector lo) {
    HVX_VectorPair w;
    memcpy(&w, &lo, 128);
    memcpy((unsigned char *)&w + 128, &hi, 128);
    return w;
}

static inline HVX_Vector Q6_V_lo_W(HVX_VectorPair w) {
    HVX_Vector v;
    memcpy(&v, &w, 128);
    return v;
}

static inline HVX_Vector Q6_V_hi_W(HVX_VectorPair w) {
    HVX_Vector v;
    memcpy(&v, (const unsigned char *)&w + 128, 128);
    return v;
}

/* qf32 add of two sf32 vectors, per 32 lanes. Used in HMXLayout.c both as
 * the v79 sf32->qf32 convert (adding zero, documented there as exact) and as
 * the fused tail's residual add. Emulated as plain float addition; the
 * battery only feeds values (small integers and quarters) that are exact in
 * fp32 and well inside any plausible qf32 significand, so emulation
 * precision cannot mask a permutation error. */
static inline HVX_Vector Q6_Vqf32_vadd_VsfVsf(HVX_Vector a, HVX_Vector b) {
    float A[32], B[32], O[32];
    memcpy(A, &a, 128);
    memcpy(B, &b, 128);
    for (int i = 0; i < 32; i++)
        O[i] = A[i] + B[i];
    HVX_Vector v;
    memcpy(&v, O, 128);
    return v;
}

/* qf32 pair -> halfword vector: .lo's 32 values become the EVEN halfwords,
 * .hi's the ODD ones (the inverse of the Wqf32 widening, which places even
 * input halfwords in .lo -- as transcribed in the tree-external layout
 * oracle from llama.cpp's producers). The f32->f16 rounding is the host
 * compiler's (_Float16 cast, round-to-nearest-even); the battery's values
 * are exact in fp16 so the rounding mode is not load-bearing. */
static inline HVX_Vector Q6_Vhf_equals_Wqf32(HVX_VectorPair w) {
    float L[32], H[32];
    memcpy(L, &w, 128);
    memcpy(H, (const unsigned char *)&w + 128, 128);
    _Float16 o[64];
    for (int i = 0; i < 32; i++) {
        o[2 * i] = (_Float16)L[i];
        o[2 * i + 1] = (_Float16)H[i];
    }
    HVX_Vector v;
    memcpy(&v, o, 128);
    return v;
}

/* Halfword widening multiply: EVEN input halfwords (times b's even) become
 * .lo's 32 values, ODD become .hi. Used in HMXLayout.c as widen-by-1.0
 * (0x3C00 = fp16 1.0), exact for every finite fp16. */
static inline HVX_VectorPair Q6_Wqf32_vmpy_VhfVhf(HVX_Vector a, HVX_Vector b) {
    _Float16 A[64], B[64];
    memcpy(A, &a, 128);
    memcpy(B, &b, 128);
    float lo[32], hi[32];
    for (int i = 0; i < 32; i++) {
        lo[i] = (float)A[2 * i] * (float)B[2 * i];
        hi[i] = (float)A[2 * i + 1] * (float)B[2 * i + 1];
    }
    HVX_VectorPair w;
    memcpy(&w, lo, 128);
    memcpy((unsigned char *)&w + 128, hi, 128);
    return w;
}

/* qf32 -> sf32 narrow convert. In this emulation qf32 lanes are held as
 * plain float32 (see Q6_Vqf32_vadd_VsfVsf), so this is the identity; the
 * battery's exactness discipline makes that sound. */
static inline HVX_Vector Q6_Vsf_equals_Vqf32(HVX_Vector q) {
    return q;
}

/* Splat a 16-bit pattern to all 64 halfwords. Used only with 0x3C00 (fp16
 * 1.0) in HMXLayout.c. */
static inline HVX_Vector Q6_Vh_vsplat_R(int x) {
    uint16_t p = (uint16_t)(unsigned)x;
    uint16_t o[64];
    for (int i = 0; i < 64; i++)
        o[i] = p;
    HVX_Vector v;
    memcpy(&v, o, 128);
    return v;
}

/* Halfword deal: EVEN input halfwords gather to the low 64 bytes, ODD to the
 * high 64 (de-interleave at halfword granularity).
 * Evidence: llama.cpp's hvx_vec_f32_to_f16 is vdeal(shuff(v0, v1)) -- it
 * turns the interleaved shuff output into the concatenated [v0 | v1] form --
 * and HMXLayout.c's 32-column unpack tail uses it to split one interleaved
 * crouton block into scratch rows [r0 | r1]. */
static inline HVX_Vector Q6_Vh_vdeal_Vh(HVX_Vector v) {
    uint16_t A[64], O[64];
    memcpy(A, &v, 128);
    for (int i = 0; i < 32; i++) {
        O[i] = A[2 * i];
        O[32 + i] = A[2 * i + 1];
    }
    HVX_Vector o;
    memcpy(&o, O, 128);
    return o;
}

/* Two-vector deal with granularity -2 (halfwords): treat the 256-byte
 * concatenation [second operand | first operand] (second operand at the low
 * 128 bytes, matching vcombine's argument order) and de-interleave its
 * halfwords: .lo = even halfwords, .hi = odd halfwords.
 * Evidence: both independent users (HMXLayout.c's hmx__unpack_chunk and
 * llama.cpp's flash-attn-ops.c crouton decode, identical operand order)
 * pass (second_crouton_block, first_crouton_block, -2) and read row r0 from
 * .lo with its 64 columns in order -- which is exactly this composition.
 * Any granularity other than -2 is refused loudly: this file has no other
 * use, and silently guessing a different deal width would be the one way
 * this emulation could lie. */
static inline HVX_VectorPair Q6_W_vdeal_VVR(HVX_Vector a, HVX_Vector b, int granularity) {
    if (granularity != -2) {
        fprintf(stderr, "hmx host emu: Q6_W_vdeal_VVR granularity %d != -2 (not derived, refusing)\n", granularity);
        abort();
    }
    uint16_t concat[128];
    memcpy(concat, &b, 128);         /* second operand: low 128 B */
    memcpy(concat + 64, &a, 128);    /* first operand: high 128 B */
    uint16_t lo[64], hi[64];
    for (int i = 0; i < 64; i++) {
        lo[i] = concat[2 * i];
        hi[i] = concat[2 * i + 1];
    }
    HVX_VectorPair w;
    memcpy(&w, lo, 128);
    memcpy((unsigned char *)&w + 128, hi, 128);
    return w;
}

/* Predicated vector store: stores exactly the bytes whose predicate lanes
 * are set. Deliberately byte-wise through an unsigned-char pointer: the
 * production call sites cast possibly-unaligned destination addresses to
 * HVX_Vector * (aligned type), which is fine for the hardware's predicated
 * store but would let a host compiler emit an aligned move if this
 * emulation dereferenced the typed pointer. */
static inline void Q6_vmem_QRIV(HVX_VectorPred q, HVX_Vector *p, HVX_Vector v) {
    unsigned char *dst = (unsigned char *)p;
    unsigned char V[128];
    memcpy(V, &v, 128);
    for (int i = 0; i < 128; i++)
        if (q.lane[i])
            dst[i] = V[i];
}

#endif /* HMX_HOST_EMU_HVX_HEXAGON_PROTOS_H */
