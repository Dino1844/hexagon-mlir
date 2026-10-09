# ===- conftest.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===---------------------------------------------------------------------------==

"""Shared pytest setup for the on-device test trees under this directory.

Every Triton test module used to repeat this pair:

    from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
    triton.runtime.driver.set_active(HexagonDriver())

Both lines are import-time bookkeeping, not device access:

* ``HexagonDriver.__init__`` only builds a launcher class and records a couple of
  strings (``backend/driver.py``, ``backend/triton_hexagon_launcher.py``);
* ``DriverConfig.set_active`` only stores the driver in a field
  (``triton/runtime/driver.py``).

Nothing in this file opens a socket or an ssh session. The DSP is first touched
when a kernel is *launched*, from a test body, and that still happens through
``tools/run_tests.sh`` under the device lock.

Doing it here means the driver is installed once, before any test module is
imported (pytest loads a directory's conftest before the modules in it), so the
test modules keep only the imports whose names they actually use -- ``torch``,
``triton``, ``tl``, ``libdevice`` -- and not the driver pair. A new test file
that omits the pair therefore runs on the DSP instead of dying with "no active
driver".
"""

import triton
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver


def activate_hexagon_driver() -> None:
    """Make the Hexagon backend Triton's active driver. No device access."""
    triton.runtime.driver.set_active(HexagonDriver())


activate_hexagon_driver()
