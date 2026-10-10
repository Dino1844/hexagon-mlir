//===- HmxVectorExecutor.h - Run HVX read-out on a second thread ------------===//
//
// WHY THIS EXISTS
// --------------
// A lowered HMX matmul issues its read-out (`hmx.unpack_acc`) on the same thread
// as the matrix engine, so the two Hexagon engines take turns. For
// 1024x512x64 at pipeline-depth=2 the read-out is 39.79% of the kernel -- 24.3 us
// of 61 us -- and overlaps nothing.
//
// Moving it to a second thread looks hopeless at first and is not, because of how
// the work is batched. Measured on this device (exp/hmx/s2_handoff/probe2.cpp):
// a 32-byte descriptor handoff with a `qurt_signal` wake costs ~0.9 us end to
// end, which is MORE than the 760 ns of read-out one m-tile carries. Per tile it
// therefore loses. But the accumulator array holds every m-row at once
// (post-partition IR: `%alloc_2` is allocated before the m-loop and released after
// it), so the read-out of rows [m, m+G) may be deferred as a group with no extra
// VTCM and no double buffering. Cost per tile then falls as 900/G ns:
//
//     G=1 -> 28.8 us of handoff, loses.   G=2 -> 14.4 us, 1.19x.
//     G=4 ->  7.2 us, 1.39x.               G=8 -> 3.6 us, 1.51x.
//
// THIS IS THE SHAPE llama.cpp USES, NOT THE OBVIOUS ONE
// ---------------------------------------------------
// ggml/src/ggml-hexagon/htp/hmx-queue.h is a lock-free SPSC ring plus a dedicated
// resident thread that owns the HMX unit; matmul-ops.c:2601-2623 pushes chunk i
// and pops chunk i-1, so the wake latency lands on a generation that is already
// done. Two details worth copying, and one worth NOT copying:
//
//   * copy: bounded spin then block. hmx-queue.c:70-86 spins HMX_QUEUE_POLL_COUNT
//     times before qurt_futex_wait. 2000 on parts above v79, 1 on v79 and below.
//     An unbounded spin is what made the first attempt at measuring this hang the
//     DSP for 13 minutes.
//   * copy: give the HVX context back while idle (hmx-queue.c:76-77,
//     qurt_hvx_unlock). Our engine thread is the caller's own thread, so the
//     idle side is this executor.
//   * do not copy: their persistent thread sits on the HMX unit because their
//     matmul is dispatched from a host thread. Ours is entered directly on the
//     DSP, so the calling thread already holds the engine and the HMX lock
//     (hexagon_runtime_hmx_ensure_dsp). Only the VECTOR side needs a thread,
//     which is why this file is a vector executor and not a symmetric one.
//
// THE HVX DEADLOCK QUESTION, SETTLED
// ---------------------------------
// The executor thread needs an HVX context for `unpack_acc`. The SDK says
// qurt_hvx_lock() blocks "if the current HVX mode is different from the requested
// mode" (qurt_hvx.h:141-145), and if this part exposed only ONE 128B unit while
// the engine thread held it, that call would block forever while the engine
// thread blocked forever in drain() -- the kernel would hang holding the device.
//
// Measured on v79 with qurt_hvx_try_lock, which by contract cannot block
// (exp/hmx/s2_handoff/hvx_units.cpp, 2026-10-03):
//
//     single_try_lock      rc=0  1519 pcyc (0.7 us to acquire a 128B unit)
//     two_thread_try_lock  holder_rc=0 contender_rc=0
//     VERDICT              AT_LEAST_TWO_128B_HVX_UNITS
//
// So two 128B units exist and the executor thread is safe. Acquiring one costs
// ~0.7 us, paid once at thread start, not per batch.
//
// WHAT STILL DOES NOT SETTLE
// --------------------------
// Whether two threads reading and writing the same VTCM banks contend on ports
// is a hardware question this interface cannot answer. `unpack_acc(row m)` and
// `hmx.acc_read(m', n)` touch byte-disjoint rows, so there is no aliasing in the
// IR; port contention is separate and remains unmeasured. The first real
// measurement must treat a null speedup as ambiguous rather than as proof that
// the split cannot work.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HMX_VECTOR_EXECUTOR_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HMX_VECTOR_EXECUTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Version of this ABI, so a kernel built against an older header fails loudly at
/// init instead of calling a function whose descriptor it does not fill in.
#define HEXMLIR_HMX_VECTOR_EXECUTOR_ABI 1

/// One group of accumulator rows ready to be read out.
///
/// SIX 32-BIT WORDS -- 24 bytes on the target, where pointers are 32-bit. The
/// handoff cost was measured with a 32-byte descriptor (exp/hmx/s2_handoff/
/// probe2.cpp, 2026-10-02), so the real cost of shipping this struct is a little
/// BELOW the measured figure and the batch-size table derived from it is
/// conservative rather than optimistic.
///
/// Both sides of this ABI must agree on the layout, and they agree by
/// construction rather than by convention: the runtime declares the ring as
/// `HmxReadoutBatch slots[16]` and so inherits sizeof, while the compiler emits
/// exactly six i32 words. If a field is ever added, BOTH change together -- a
/// mismatch here is not a compile error, it is the executor reading garbage
/// pointers out of the middle of a tile descriptor. The static_assert below
/// makes the runtime half of that pair a loud build break (this header is
/// compiled only for the 32-bit device target, so 24 bytes is the pinned
/// size); the compiler half is pinned by kHmxReadoutBatchWords in
/// HmxReadoutHandoff.h.
typedef struct {
  uint32_t rowStart;   ///< first m-row in `ar`
  uint32_t rowCount;   ///< how many rows, i.e. the batch size G
  uint32_t validRows;  ///< rows valid in the LAST row of the batch (tail tile)
  uint32_t nCroutons;  ///< acc_read's `count`: n-croutons per row
  void    *ar;         ///< accumulator array in VTCM (all rows live, any subset)
  void    *dst;        ///< row-major destination, already offset to rowStart
} HmxReadoutBatch;
static_assert(sizeof(HmxReadoutBatch) == 24,
              "the read-out descriptor ABI changed size on the device target; "
              "every side agrees on the layout through this header, so this "
              "break is loud by construction -- the compiler emission must "
              "carry the new field count via kHmxReadoutBatchWords too");

/// Type of the read-out the executor runs on the vector thread. The compiler
/// emits one of these per kernel; it is the function that used to be inline.
typedef void (*HmxReadoutFn)(const HmxReadoutBatch *batch, uint32_t count);

/// Return codes from `configure`. A negative return means the executor is NOT
/// usable and the caller must run the read-out inline, because publishing to a
/// dead executor would drop batches and hand the caller an unwritten output.
#define HEXMLIR_HMX_EXEC_OK             0
#define HEXMLIR_HMX_EXEC_ERR_NULL_FN (-1)  ///< fn was null
#define HEXMLIR_HMX_EXEC_ERR_THREADS (-2)  ///< numThreads clamped to 1
#define HEXMLIR_HMX_EXEC_ERR_CREATE  (-3)  ///< the resident thread would not start

/// Declare which function the vector thread runs. Idempotent: calling it again
/// with the same function replaces the previous one (recompiles reuse the same
/// resident thread), a different function waits for the current read-out to
/// finish first. Must be called before the first `publish` for a kernel.
int32_t hexagon_runtime_hmx_exec_configure(HmxReadoutFn fn, int32_t numThreads);

/// Hand `count` batches to the vector thread. Returns the number accepted, which
/// is less than `count` only if the ring is full -- a full ring means the vector
/// side cannot keep up, and dropping is better than blocking the engine thread
/// because the alternative is losing the overlap for everything behind it.
/// Never blocks.
///
/// THE CALLER MUST CHECK THIS. A dropped batch means that destination region is
/// never written, so the kernel returns its own uninitialized output -- a wrong
/// answer, not a slower one. There is deliberately no silent inline fallback
/// inside the runtime: a caller that ignores the count gets whatever happened to
/// be in the destination buffer.
uint32_t hexagon_runtime_hmx_exec_publish(const HmxReadoutBatch *batches,
                                          uint32_t count);

/// Block until every published batch has been read out. The kernel calls this
/// before returning, because the destination buffer is the caller's memory.
void hexagon_runtime_hmx_exec_drain(void);

/// Release the vector thread. Called at session teardown, not per launch: the
/// thread is deliberately resident, because creating one costs ~18 us measured on
/// this device, which is most of the 24.3 us the split is trying to recover.
void hexagon_runtime_hmx_exec_shutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HMX_VECTOR_EXECUTOR_H