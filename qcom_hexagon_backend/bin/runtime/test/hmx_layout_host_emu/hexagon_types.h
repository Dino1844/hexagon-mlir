/* Test-only host stub of the SDK's <hexagon_types.h>.
 *
 * Consumed by: test_hmx_layout_permutation_contract.py, which compiles the
 * REAL bin/runtime/hmx/src/HMXLayout.c on the host by putting this directory
 * first on the include path (HMXLayout.c includes <hexagon_types.h> itself).
 * It never enters any device build: nothing in bin/runtime/test/CMakeLists.txt
 * references it, and that list is literal (no globs).
 *
 * The types are the SDK's shapes, re-expressed with GCC/Clang vector
 * extensions so that the production file's own statements -- assignments
 * between HVX_UVector and HVX_Vector, dereferences of HVX_UVector pointers at
 * addresses that are only 2-byte aligned -- compile and run unchanged:
 *
 *   typedef long HVX_Vector     vector_size(128), aligned(128);
 *   typedef long HVX_UVector    vector_size(128), aligned(2);
 *   typedef long HVX_VectorPair vector_size(256), aligned(256);
 *
 * Both compilers accept the pair as compatible types (probed on gcc 13 and
 * clang 18; the unaligned dereference compiles to an unaligned-safe 128-byte
 * access, which the same test exercises at runtime through misaligned
 * sources). Little-endian is assumed, same as Hexagon and x86.
 *
 * HVX_VectorPred is opaque here: only Q6_Q_vsetq_R produces one and only
 * Q6_V_vmux_QVV / Q6_vmem_QRIV consume it, all inside the companion stub
 * hvx_hexagon_protos.h, so its representation is this emulation's own
 * business. It is a byte-lane flag array (one flag per byte of a 128 B
 * vector), which is the granularity the intrinsics' documented behavior
 * needs.
 */
#ifndef HMX_HOST_EMU_HEXAGON_TYPES_H
#define HMX_HOST_EMU_HEXAGON_TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef long HVX_Vector __attribute__((vector_size(128), aligned(128)));
typedef long HVX_UVector __attribute__((vector_size(128), aligned(2)));
typedef long HVX_VectorPair __attribute__((vector_size(256), aligned(256)));

typedef struct {
    unsigned char lane[128];
} HVX_VectorPred;

/* Marker asserted by the driver after including HMXLayout.c: if a real SDK
 * hexagon_types.h ever won the include-path race instead of this stub, the
 * driver must fail loudly rather than silently compile against different
 * types. */
#define HMX_HOST_EMU_HEXAGON_TYPES_STUB 1

#endif /* HMX_HOST_EMU_HEXAGON_TYPES_H */
