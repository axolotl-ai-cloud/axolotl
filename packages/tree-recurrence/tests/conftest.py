# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import pytest
from tree_recurrence import uninstall


@pytest.fixture(autouse=True)
def _uninstalled():
    """Every test starts and ends without hooks."""
    uninstall()
    yield
    uninstall()
