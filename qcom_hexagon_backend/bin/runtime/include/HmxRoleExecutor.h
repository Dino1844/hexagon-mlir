//===- HmxRoleExecutor.h - Bind a resident thread to one role --------------===//
//
// WHY THIS EXISTS (ROADMAP1001 §2, §4.1)
// --------------------------------------
// A lowered HMX kernel issues its engine leaves (mma / bias_load / acc_read)
// on the same thread as its vector work (pack_act / unpack_acc), so the two
// Hexagon engines take turns. The thread-role split moves the engine section
// of a dual-role kernel onto ONE dedicated resident thread -- T_HMX -- and
// hands tile groups to it through the SPSC ring in HmxSpscRing.h:
//
//     launching thread (producer):  pack_act(i+1) ... unpack(i)
//     T_HMX (consumer):                          mma(i+1) / bias_load / acc_read
//
// The shape is the IREE `deferred_work_queue` bind_to_thread abstraction:
// work is SUBMITTED from the launching thread but EXECUTED on the thread the
// executor bound it to. It is also deliberately the same shape as the
// read-out split already in this tree (multithreading/HmxVectorExecutor.cpp,
// configure/publish/drain/shutdown): that file is the vector-side precedent
// and this one follows it rather than inventing a second style.
//
// THE FIVE INTERFACES
// -------------------
//   bind(fn, depth)   declare which section the bound thread runs, and how
//                     deep the ring is. Idempotent; a rebind waits for all
//                     in-flight work first (drain-before-swap, below).
//   submit(groups,n)  hand n tile-group descriptors to the bound thread.
//                     Never blocks; returns how many were accepted.
//   wait_retired(t)   block until the retire counter reaches t: the sections
//                     of the first t groups have all RETURNED. The granular
//                     form of drain (see the entry's own comment for the
//                     read-out coexistence it exists for).
//   drain()           block until every accepted group has been executed.
//   join()            release the bound thread. Session teardown only.
//
// DESIGN DECISIONS, EACH WITH ITS PROVENANCE
// ------------------------------------------
// * THE BOUND THREAD IS RESIDENT, AND ITS LIFETIME IS THE PROCESS, NOT THE
//   LAUNCH. Creating a thread measured ~18 us on this device while the whole
//   read-out the vector split hides is 24.3 us (exp/hmx/s2_handoff/
//   probe2.cpp; HmxVectorExecutor.cpp startVectorThread). Worse, the
//   launching wrapper's ThreadManager creates FRESH qurt threads on every
//   launch and joins them at the end (test/utils/dsp/include/
//   multithreading.h:57 keeps its own "TODO: Keep the thread pool alive"),
//   and a per-launch thread identity is exactly what sank
//   enableWorkspaceResident's first attempt: the resident map keyed on a
//   thread serial that rotated every launch and grew without bound
//   (AGENTS.md, workspace-resident entry). T_HMX therefore escapes the
//   per-launch lifecycle BY CONSTRUCTION: it is created at the first bind
//   and destroyed only by join(), which is a session-teardown call, never a
//   per-launch one. Nothing in the launch path owns or touches it.
//
// * T_HMX TAKES THE HMX LOCK ONCE AND HOLDS IT FOR ITS LIFETIME (the first
//   half of the lock-ownership migration, ROADMAP1001 §1.2/§4.2). The bound
//   thread calls hexagon_runtime_hmx_ensure_dsp exactly once, at thread
//   start, before it can run any section; it never unlocks. This is the
//   decision recorded in ROADMAP1001 §4.2 ("默认不释放--单实例部署你们已
//   拍板"). The per-kernel ensure/unlock pairing the compiler emits today is
//   the LEGACY path and is deliberately untouched: it serves every kernel
//   that exists today, and this executor is inert until something binds it
//   (nothing does until the S3 dispatch channel lands).
//
//   ⚠️ THE HAZARD THAT MAKES "LAZY" LOAD-BEARING: a lock on a held HMX unit
//   BLOCKS in the resource manager until the holder releases it
//   (bin/runtime/src/HexagonCAPI.cpp, hexagon_runtime_hmx_ensure_dsp's
//   threading note). Once T_HMX exists and holds the lock for life, ANY
//   other thread's per-kernel ensure blocks forever. Binding is therefore
//   only correct in a process whose HMX work is routed through this
//   executor -- which is the S3 dispatch channel's contract, not this
//   file's. Until that channel exists, nothing may call bind.
//
// * SUBMIT NEVER BLOCKS AND A SHORT RETURN IS THE CALLER'S PROBLEM. A full
//   ring means the consumer cannot keep up; blocking the producer would
//   stall the engine-side pipeline for every group behind the waited one,
//   so the tail is dropped instead and the return value says how many were
//   taken. A dropped group is work that never runs -- a wrong answer, not a
//   slower one -- exactly as for HmxVectorExecutor's publish
//   (include/HmxVectorExecutor.h:124-134). The S3 compiler side must check
//   the count.
//
// * SINGLE PRODUCER, BY CONSTRUCTION. One launch is one producer thread; a
//   second concurrent producer would need an MPSC ring and is not supported
//   (same contract as HmxVectorExecutor). bind() re-establishes the producer
//   role for a new launch by draining first, so the producer may legally be
//   a different thread on every launch -- just never two at once. grid>1
//   launches are outside this contract (the kernel-side ring protocol is
//   grid=1, matching the existing executor's stated boundary).
//
// * THE RING DEPTH IS A PARAMETER, NOT A CONSTANT. The compiler side derives
//   it from the tile-ring geometry of the kernel (ROADMAP1001 §2 choice 4);
//   this ABI refuses depth 0 rather than inventing a default. Capacity is
//   grow-only across rebinds: a smaller rebind keeps the larger storage (a
//   deeper-than-needed ring is harmless), a larger rebind reallocates under
//   the drain-before-swap window (HmxSpscRing.h, reset()).
//
// WHAT IS DELIBERATELY NOT HERE
// -----------------------------
// * No HVX context handling: T_HMX never touches HVX (that is the point of
//   the split -- the engine thread needs no HVX context, upstream PR
//   #222340's TTI note). The vector-side executor is the one that manages an
//   HVX context; this one must not grow one.
// * No host->device dispatch channel, no dlsym probing, no manifest reading:
//   that is the next slice (ROADMAP1001 §4.6). This file is the runtime
//   floor the channel will stand on.
// * No diagnostics channel in v1: the readout executor's accounting-file
//   detour measured 684 us per append and never came back from a
//   FastRPC-launched kernel anyway (HmxVectorExecutor.cpp,
//   kReadoutAccountingPath). Counters and dumps, if S3 needs them, go in
//   with their own measured justification.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_EXECUTOR_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_EXECUTOR_H

#include <stdint.h>

#include "HmxSpscRing.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Version of this ABI, so a kernel built against an older header fails
/// loudly at bind instead of calling an entry whose descriptor it mis-fills.
#define HEXMLIR_HMX_ROLE_EXECUTOR_ABI 1

/// The section the bound thread runs: one call per tile-group descriptor,
/// with `count == 1` -- the same per-item calling discipline as
/// HmxReadoutFn (include/HmxVectorExecutor.h:106-108; a run-coalesced call
/// would make the meaning of `count` depend on ring geometry, a contract the
/// compiler pass would have to reproduce exactly).
typedef void (*HmxSectionFn)(const HmxTileGroupDesc *groups, uint32_t count);

/// Return codes from bind. Negative means the executor is NOT usable and the
/// caller must run the section inline on the launching thread.
#define HEXMLIR_HMX_ROLE_OK          0
#define HEXMLIR_HMX_ROLE_ERR_NULL_FN (-1) ///< fn was null
#define HEXMLIR_HMX_ROLE_ERR_DEPTH   (-2) ///< depth was 0 (depth is a parameter; there is no default)
#define HEXMLIR_HMX_ROLE_ERR_CREATE  (-3) ///< ring storage or the thread could not be created

/// Declare which section the bound thread runs and how deep the ring is.
///
/// Idempotent. A rebind with the same section just re-stores it; a rebind
/// with a different section (a recompile in the same process) first waits
/// for every in-flight group from the previous epoch (the drain barrier), so
/// the bound thread never runs a previous kernel's section against this
/// kernel's descriptors. A larger depth reallocates the ring in that same
/// quiesced window; a smaller one keeps the existing (larger) ring.
///
/// The first successful bind creates T_HMX: it takes the HMX lock at thread
/// start and holds it for its lifetime (see the hazard note in the file
/// header -- nothing may bind until the process routes its HMX work through
/// this executor).
int32_t hexagon_runtime_hmx_role_bind(HmxSectionFn fn, uint32_t depth);

/// Hand `count` tile-group descriptors to the bound thread; returns how many
/// were accepted, which is less than `count` only when the ring filled.
/// Never blocks. THE CALLER OWNS THE DROPPED GROUPS: a dropped group never
/// executes, so ignoring a short return is silent corruption, not a slowdown
/// (same contract as hexagon_runtime_hmx_exec_publish).
uint32_t hexagon_runtime_hmx_role_submit(const HmxTileGroupDesc *groups,
                                         uint32_t count);

/// Block until every accepted group has been executed (the bound section has
/// RETURNED for each -- the barrier is on the ring's retire counter, which
/// advances only after the section returns). The kernel must call this
/// before returning, because the destinations the sections write are the
/// caller's memory.
void hexagon_runtime_hmx_role_drain(void);

/// Wait until the retire counter has reached `target`: every group the ring
/// retired before the target-th retire has had its section RETURN, so the
/// destinations those sections wrote (the caller's memory) are visible to the
/// caller on return. `target <= retired` returns immediately.
///
/// THE GRANULAR-BARRIER ENTRY (added 2026-10-09, R2; the review that proposed
/// it is docs/architecture/
/// hmx-coscheduling-architecture-review-2026-10-09.md section 4f). drain is
/// the ALL-groups barrier and the only one the first form needed: it runs once
/// at kernel exit, where "everything" is the only granularity that exists.
/// The read-out coexistence needs a finer one -- the producer publishes tile
/// i's read-out to the vector executor INSIDE the tile loop, and that publish
/// may only happen once tile i's engine section has returned. `tail` already
/// advances per group (HmxSpscRing.h, retire), and the producer already
/// acquire-loads it (publish's full check), so the fact this entry exposes is
/// not new state -- only a read of it, with a wait.
///
/// The safety argument this entry carries, stated once because it replaces an
/// older one: the first form moved every in-loop read-out past the exit drain
/// because "after drain, every section has returned" was the only proof the
/// `ar` rows were complete. This entry upgrades that proof to "after the
/// retire of group i, group i's section has returned" -- semantically the same
/// statement (retire happens only after the section returns,
/// HmxSpscRing.h's ownership model) at per-group granularity, which is exactly
/// the upgrade "same semantics, finer grain" the review asked for. The
/// producer-side order the compiler emits is submit(batch k+1) ...
/// wait_retired(batch k's last row + 1) ... publish(batch k): the wait
/// is the proof, the publish is what it unblocks, and the publish LAGS
/// the submit frontier by one batch. The first emission waited for the
/// batch it had just submitted (lag zero), which serialized the producer
/// behind the engine -- measured +10 us on S1 (r2r3-ab-2026-10-09) -- so
/// the lag is load-bearing: waiting for already-submitted, almost-
/// certainly-retired tiles costs nothing, while waiting for the batch
/// just handed over parks the only thread that could keep the engine fed.
///
/// PARK POLICY: identical to drain -- a bounded spin (kPollCount), then
/// qurt_futex_wait on the ring's `tail` word with a re-check before the park
/// (the lost-wakeup window drain's comment documents). The consumer's retire
/// wakes a parked waiter (it futex-wakes `tail` once per group). No unbounded
/// spin exists on any path -- including an UNSATISFIABLE one: a target beyond
/// the published head can never be retired (the caller is the producer, so
/// nothing new is published while it waits), and waiting for it forever is the
/// 13-minute-DSP-hang failure mode this file's design constraint 3 exists to
/// prevent. Such a target is a caller bug; this entry names it in the log and
/// RETURNS instead of parking, so the failure is loud and recoverable rather
/// than silent and fatal. A target of 0 returns immediately (nothing has ever
/// been published that 0 would wait for).
///
/// ADDITIVE, AND THE ABI VERSION STAYS 1 -- DELIBERATELY. The macro above
/// exists so "a kernel built against an older header fails loudly at bind
/// instead of calling an entry whose descriptor it mis-fills". This entry
/// mis-fills nothing: it takes one uint32 whose meaning is self-contained, it
/// is a NEW symbol no kernel built against an older header references, and a
/// kernel that does reference it pulls the same archive member that defines
/// submit/drain (all three live in HmxRoleExecutor.cpp), so no kernel can call
/// it against a runtime that lacks it. A version bump here would fail old
/// kernels at bind for no defect they can exhibit -- the opposite of the
/// macro's purpose.
void hexagon_runtime_hmx_role_wait_retired(uint32_t target);

/// Release the bound thread. Session teardown only -- NEVER per launch: the
/// thread is the process's HMX-role thread, and paying its ~18 us creation
/// cost (plus an HMX re-ensure) per launch is exactly the per-launch
/// lifecycle this executor exists to escape. Drains before stopping, so it
/// cannot race a section that is writing into caller memory.
///
/// OUT OF CONTRACT: binding again after join. The thread exits WITHOUT
/// releasing the HMX lock (decision 2 -- no unlock, ever), so whether a
/// freshly created bound thread can even take the lock afterwards is a
/// device property, not a guarantee of this ABI; it is what the
/// lock-lifetime device probe (exp/hmx/s2_runtime_probes/lock_lifetime)
/// exists to answer. Until it does, a process gets ONE bound epoch: bind,
/// work, join, end.
void hexagon_runtime_hmx_role_join(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_EXECUTOR_H
