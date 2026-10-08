/* Test-only host emulation of the HAP FARF logging macro. NOT for the device
 * build: there the real SDK header (incs/HAP_farf.h) wins the include race.
 *
 * The SDK's FARF(level, fmt, ...) prints a leveled log line. The emulation
 * prints the same shape to stderr, which is exactly what the host test needs:
 * the ERROR paths the production code logs (submit-before-bind, ring full)
 * must be audible in the driver session, and stderr is captured by the
 * Python harness so a test can assert on it where that matters.
 *
 * This stub is clean under -pedantic (the SDK header's `, ##__VA_ARGS__` is
 * what forces the pragma dance in the production TU; not needed here).
 */
#ifndef HMX_ROLE_HOST_EMU_HAP_FARF_H
#define HMX_ROLE_HOST_EMU_HAP_FARF_H

#include <cstdio>

#define FARF(level, ...)                                                      \
  do {                                                                        \
    std::fprintf(stderr, "FARF " #level ": " __VA_ARGS__);                    \
    std::fprintf(stderr, "\n");                                               \
  } while (0)

#endif /* HMX_ROLE_HOST_EMU_HAP_FARF_H */
