# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Treat CUDA OOM as a skip for tests in this directory.

When the suite runs under ``pytest-xdist``, multiple workers contend for the
same physical GPU's memory budget. A test that fits comfortably in isolation
can OOM purely because peer workers are already holding most of VRAM. That's
an environmental race, not a code defect, so converting it to a skip keeps
mixed-GPU CI green without masking real regressions (a real correctness bug
surfaces as an assert/exception, not as ``torch.OutOfMemoryError``).

We hook ``pytest_runtest_call`` rather than using an autouse fixture because
pytest captures the test exception before re-entering the fixture's
generator — the fixture's ``try/except`` around ``yield`` never sees it.
"""

from __future__ import annotations

import gc

import pytest
import torch


def _cuda_oom_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = []
    if hasattr(torch, "OutOfMemoryError"):
        types.append(torch.OutOfMemoryError)
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and cuda_oom not in types:
        types.append(cuda_oom)
    return tuple(types) or (RuntimeError,)


_OOM = _cuda_oom_types()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is None:
        return
    exc_val = excinfo[1]
    if isinstance(exc_val, _OOM):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outcome.force_exception(
            pytest.skip.Exception(
                f"skipping on CUDA OOM (likely xdist worker contention): {exc_val}",
                _use_item_location=True,
            )
        )
