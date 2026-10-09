# ===- conftest.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===---------------------------------------------------------------------------==

"""Shared pytest setup for the MLIR kernel tests under this directory.

Every test directory under here used to open with the same prologue:

    import os
    import sys

    # Add parent directory to path to import mlir_test_utils
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mlir_test_utils import run_mlir_kernel_test

The shim existed only because these test directories are flat (no
``__init__.py``) while the shared helper lives one level up, so a plain
``from mlir_test_utils import ...`` could not resolve on its own.

pytest imports this conftest before it imports any module in this subtree, so
the shim runs once here and the test files keep an ordinary import. Pure path
bookkeeping: nothing here touches the DSP.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
