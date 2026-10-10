# ===- launch_intent.py -----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""What the caller intends to do with this launch's outputs.

The device tunnel is priced per byte in both directions: a launch that writes
a 4 MB output pays ~4 s/MB to pull it back (measured 2026-10-10, see
``docs/env/hexmlir-testbed.md`` "测量发射协议"), and a launch whose result
nobody reads pays that for nothing.  Steady-state benchmarks are exactly that
case: they run the same kernel many times and consume only the ``Perf`` line,
which comes from ``perf.txt``, not from the output tensors.

So the launcher accepts a *declaration* of intent, scoped to a block of
launches:

.. code-block:: python

    from triton.backends.qcom_hexagon_backend.launch_intent import (
        pull_outputs,
    )

    with pull_outputs(False):
        kernel[(1,)](*args)      # Perf-only: no output pull, no copy-back

Why a context manager instead of a launch kwarg or an environment variable:

* A launch kwarg cannot reach here.  ``CompiledKernel.run`` is called with the
  bound arguments only; extra ``**kwargs`` from ``kernel[grid](*args, **kw)``
  are consumed by the binder and the compiler, never forwarded to the launcher.
* An environment variable is process-global, so one forgotten ``export`` in a
  session silently turns off the data every ``rel`` check depends on.  That is
  the failure mode this feature must not have.

Why this lives in the launcher and not in :class:`HexagonExecutor`: the
executor has no idea who wants the bytes; the caller does.

Semantics
---------
``pull_outputs(False)`` skips exactly two things: the ``adb ... pull`` of the
output tensors, and reading those local files back into host tensors.  It does
**not** touch

* the device-side dump (the wrapper still writes every output file, so the
  next launch still overwrites this run's product and a later pull is never
  stale),
* ``perf.txt`` (``Perf`` / ``PerfPcycles`` are still pulled, so the timing
  itself is unaffected),
* the launch itself (the kernel runs, the input push happens, the exit status
  is still checked).

The default is ``True``: a caller that declares nothing gets today's behaviour.
When outputs are skipped the launcher returns an **empty** result list, so a
caller that *did* want its tensors back sees emptiness rather than stale bytes
copied into its inputs -- the loud failure, not the quiet one.

Caveat for measurement scripts
------------------------------
An output buffer must still be deterministically initialised (``torch.zeros``)
for ``HEXAGON_FAST_LAUNCH``'s md5 push-skip to hit: the launcher pushes the
output buffer's initial bytes as an input file, so ``torch.empty`` defeats the
skip on every round.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

__all__ = ["pull_outputs", "pull_outputs_enabled"]

# One variable, not one per option, so that "the launch intent" stays a single
# fact.  A ContextVar rather than a module global: a benchmark thread (or a
# nested scope) must not leak its intent into another launch's decision.
_PULL_OUTPUTS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hexagon_pull_outputs", default=True
)


def pull_outputs_enabled() -> bool:
    """Whether this launch's output tensors should be pulled back."""
    return bool(_PULL_OUTPUTS.get())


@contextlib.contextmanager
def pull_outputs(enabled: bool) -> Iterator[None]:
    """Declare, for the launches inside this block, whether outputs are wanted.

    ``enabled=False`` is a promise that nobody reads the output tensors of the
    launches in scope -- not a request to make the kernel stop writing them.
    """
    token = _PULL_OUTPUTS.set(bool(enabled))
    try:
        yield
    finally:
        _PULL_OUTPUTS.reset(token)
