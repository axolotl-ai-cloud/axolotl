# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

import sys
from importlib.metadata import version

import pytest
from packaging.version import Version

from axolotl.integrations.kernels.autotune_collector import _KERNEL_MODULE_HINTS


@pytest.fixture
def xfail_fp8_scalar_scale(request):
    if getattr(request.node, "callspec", None) is not None:
        if request.node.callspec.params.get("fmt", "block_fp8") != "block_fp8":
            return
    if Version(version("torch")).release[:2] != (2, 15) or Version(
        version("triton")
    ).release[:2] != (3, 9):
        return
    from triton.compiler.errors import CompilationError

    request.node.add_marker(
        pytest.mark.xfail(
            strict=True,
            raises=pytest.RaisesExc(
                CompilationError,
                match="too many indices for tensor of rank 0",
                check=lambda error: "b_s[None, :]" in str(error),
            ),
            reason="finegrained-fp8 v4 grouped matmul indexes a scalar scale under Triton 3.9",
        )
    )


@pytest.fixture
def isolated_kernel_modules(monkeypatch):
    """Hide already-imported real kernel modules so autotuner scans see only the test's fakes."""
    for name in [n for n in sys.modules if any(h in n for h in _KERNEL_MODULE_HINTS)]:
        monkeypatch.delitem(sys.modules, name)
