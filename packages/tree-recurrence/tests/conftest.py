# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import sys

import pytest

# The suite runs on CPU against transformers' PyTorch references; where the CUDA-only
# kernel packages are installed, transformers would bind them on import instead.
for _package in ("causal_conv1d", "fla", "mamba_ssm"):
    sys.modules.setdefault(_package, None)  # type: ignore[arg-type]

from tree_recurrence import uninstall  # noqa: E402


@pytest.fixture(autouse=True)
def _uninstalled():
    """Every test starts and ends without hooks."""
    uninstall()
    yield
    uninstall()
