# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

import sys

import pytest

from axolotl.integrations.kernels.autotune_collector import _KERNEL_MODULE_HINTS


@pytest.fixture
def isolated_kernel_modules(monkeypatch):
    """Hide already-imported real kernel modules so autotuner scans see only the test's fakes."""
    for name in [n for n in sys.modules if any(h in n for h in _KERNEL_MODULE_HINTS)]:
        monkeypatch.delitem(sys.modules, name)
