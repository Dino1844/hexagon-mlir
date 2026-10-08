//===- HmxRoleChannel.h - per-kernel symbol channel for the role split -----===//
//
// THE HOST->DEVICE DISPATCH CHANNEL (ROADMAP1001 section 4, item 6)
// ------------------------------------------------------------------
// The device runtime does not read the manifest: bin/runtime/src/HexagonAPI.cpp
// and HexagonCAPI.cpp have zero occurrences of "manifest" or "topology" (the
// manifest reaches the HOST launcher only). A dual-role kernel -- one whose
// engine section runs on the bound thread of HmxRoleExecutor.h -- therefore
// has to be recognisable in the loaded module itself, or the launch side
// cannot know to bind before the kernel's first submit.
//
// The channel is a PER-KERNEL SYMBOL scheme (the direction ROADMAP1001
// section 4.6 records as decided):
//
//   * the compiler emits the section ENTRY POINT as a default-visibility
//     function named   <kernel entry> + HMX_ROLE_SECTION_SUFFIX
//   * the compiler emits the ring depth as a default-visibility uint32_t
//     object named    <kernel entry> + HMX_ROLE_DEPTH_SUFFIX
//   * the launch side probes the loaded module for both (dlsym) and binds
//     what it finds; a module with neither symbol is a legacy kernel and
//     keeps today's inline path, byte for byte.
//
// WHY PER-KERNEL SYMBOLS (ROADMAP1001 section 4.6's recorded rationale):
//   * less intrusive than a launch parameter: the launch ABI does not change.
//     A parameter would have to ride the FastRPC call chain, which is shared
//     with every non-HMX kernel in existence.
//   * more composable than an environment variable: the decision is per
//     kernel, spelled in the kernel's own module, and cannot be flipped for
//     one kernel by a process-wide variable another kernel set.
//
// WHY THE DEPTH IS A COMPANION DATA OBJECT AND NOT PART OF THE NAME
// ------------------------------------------------------------------
// The section name stays a pure identity -- the probe CONCATENATES, it never
// parses -- and the depth stays a typed fact (uint32_t), for three reasons:
//
//   1. Encoding the depth in the name ("<entry>__hmx_section_d2") would make
//      the name a parser contract: every producer and every probe would have
//      to agree on the number's spelling, and a mismatch would be a silently
//      wrong depth rather than a failed lookup.
//   2. The depth is a compile-time fact the compiler derives from the tile
//      ring geometry (HmxSpscRing.h: "the compiler side derives it from the
//      tile-ring geometry of the kernel"). A data object carries that fact
//      with its type; a name suffix carries it as prose.
//   3. Half-presence is detectable and refused: a module with the section
//      but not the depth (or vice versa) is a half-emitted channel, and the
//      probe returns an error so the launch fails loudly instead of binding
//      a guessed depth. A name-encoded depth cannot offer that check.
//
// WHERE THE PROBE'S ANSWER GOES
// ------------------------------------------------------------------
// Found: the launch side calls hexagon_runtime_hmx_role_bind (the executor
// ABI in HmxRoleExecutor.h, which is FROZEN and is not edited by this
// channel) with the section pointer and the depth. Not found: nothing
// happens -- the legacy kernel's own body carries today's inline engine path
// with the per-kernel ensure/unlock pair, and this channel leaves it alone.
//
// THE HAZARD THIS CHANNEL IS THE CONTRACT FOR (HmxRoleExecutor.h's header):
// once the bound thread exists it holds the HMX lock for its lifetime, so
// any OTHER thread's per-kernel ensure blocks forever. Binding is therefore
// only correct in a process whose HMX work is routed through the executor.
// A process that loads one dual-role kernel must not later run a legacy HMX
// kernel in it; this channel detects per kernel which kind it is holding,
// and refuses a half-emitted one, but it cannot undo a bind -- process
// composition is the deployer's contract, recorded here so it is written
// down exactly once.
//
// BOTH SIDES COMPILE THIS HEADER
// ------------------------------------------------------------------
// The compiler emission (lib/Conversion/HmxToLLVM/HmxToLLVMPass.cpp) and the
// runtime probe (bin/runtime/multithreading/HmxRoleExecutor.cpp) both
// include this file, so the suffix spellings agree by construction rather
// than by convention -- the same discipline HmxSpscRing.h uses for the
// descriptor layout. A suffix that drifts on one side is a failed dlsym,
// which this channel reports as "legacy" for a dual-role kernel; that case
// is why the probe ALSO refuses half-presence loudly instead of guessing.
// The header is plain C on purpose: it must compile inside the DSP runtime
// (C-ish C++) and inside the MLIR compiler (C++17) without either side's
// dependencies.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_CHANNEL_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_CHANNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Suffix of the outlined section ENTRY POINT of a dual-role kernel: the
/// entry of kernel `<name>` is the default-visibility function
/// `<name>__hmx_section`, with the C signature HmxSectionFn declares in
/// HmxRoleExecutor.h (void(const HmxTileGroupDesc*, uint32_t)).
///
/// The ENTRY is the adapter, not the outlined section body itself: the body
/// keeps its memref parameters and the entry unpacks the descriptor onto
/// them, exactly the arrangement the vector read-out uses
/// (HmxReadoutHandoff.h's kHmxReadoutEntrySuffix precedent).
#define HMX_ROLE_SECTION_SUFFIX "__hmx_section"

/// Suffix of the companion depth object of a dual-role kernel: the ring
/// depth of kernel `<name>` is the default-visibility uint32_t object
/// `<name>__hmx_role_depth`, holding the bind() depth the compiler derived
/// from the kernel's tile-ring geometry. See the file header for why this
/// is a data object and not part of the section name.
#define HMX_ROLE_DEPTH_SUFFIX "__hmx_role_depth"

/// Return codes of hexagon_runtime_hmx_role_channel_launch.
///
/// Positive: the kernel is dual-role and the executor is bound for it.
/// Zero: the kernel is legacy -- no channel symbols, nothing bound, and the
/// caller must run it exactly the way it runs every kernel today.
/// Negative: the channel exists but is broken (half-emitted symbols, a zero
/// depth, or a bind failure). THE CALLER MUST NOT RUN THE KERNEL: a
/// dual-role kernel whose bind failed would submit into an unbound
/// executor, and submit-before-bind drops every group (HmxRoleExecutor.cpp)
/// -- a silent wrong answer, which is strictly worse than a refused launch.
#define HMX_ROLE_CHANNEL_BOUND 1
#define HMX_ROLE_CHANNEL_LEGACY 0
#define HMX_ROLE_CHANNEL_ERR_ARG (-1)  ///< null kernel-entry name
#define HMX_ROLE_CHANNEL_ERR_NAME (-2) ///< entry name + suffix overflows the buffer
#define HMX_ROLE_CHANNEL_ERR_HALF (-3) ///< one channel symbol present, the other absent
#define HMX_ROLE_CHANNEL_ERR_BIND (-4) ///< bind refused (its own return code is logged)

/// The launch-side probe: is `handle`'s module a dual-role kernel, and if so
/// bind it.
///
/// `handle` is the dlsym search handle: on the device the launch side passes
/// RTLD_SELF (the channel entry lives in the same .so as the section
/// symbols, because the kernel's strong submit/drain references pull the
/// archive member that defines this entry); a host contract test passes the
/// dlopen handle of the module under test. The entry takes it as a parameter
/// precisely so it never has to spell a handle constant that differs between
/// the two platforms.
///
/// `kernel_entry` is the kernel's own entry symbol name -- the name the
/// launch side is about to call -- NOT the section name; the section and
/// depth names are built here, from the suffixes above, so the spelling
/// exists in exactly one place.
///
/// Idempotent per process to the extent bind is (HmxRoleExecutor.h): a
/// rebind with the same section re-stores it, a rebind with a different
/// section drains first. The launch side calls this once per launch, before
/// the first kernel invocation.
int32_t hexagon_runtime_hmx_role_channel_launch(void *handle,
                                                const char *kernel_entry);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HMX_ROLE_CHANNEL_H
