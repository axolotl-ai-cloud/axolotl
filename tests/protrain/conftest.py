"""Shared fixtures for ProTrain plugin tests.

Test-suite isolation quirk
--------------------------
The slow integration tests (most notably :mod:`test_integration_7b` and
:mod:`test_multi_gpu_7b`) construct a 7B-class model and drive a full
ProTrain forward+backward+step on GPU. Even after the test body
completes, the CUDA context retains fragmented allocator state, a loaded
DeepSpeed CPU-Adam extension, and per-chunk pinned-host buffers that can
linger into the next test's setup and cause spurious OOMs or device
contention.

Recommended invocation:

* Default CI: ``pytest tests/protrain/`` — slow tests are deselected by
  the ``-m 'not slow'`` addopts, so no cross-test contamination is
  possible.
* Slow suite: ``pytest tests/protrain/ -m 'slow or not slow' -p no:xdist``
  — run sequentially (no xdist) and prefer running the 7B-class tests in
  their own process (``pytest ... --forked`` or as a separate invocation).

The ``reset_cuda_state_between_tests`` fixture below is ``autouse`` for
tests marked ``slow`` so that back-to-back slow tests at least start
from a cleared allocator cache / gc cycle. It does *not* fully rebuild
the CUDA context — that still requires process isolation — but is
sufficient for the unit-scale slow tests implemented in
:mod:`test_chunk_manager` and :mod:`test_block_manager`.
"""

from __future__ import annotations

import gc
import os
from typing import Iterator

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Auto-skip ``@pytest.mark.gpu`` tests on hosts without CUDA.

    Mirrors the import / availability guards used by ``set_seed`` and
    ``reset_cuda_state_between_tests`` so the marker actually enforces
    a skip instead of merely labelling tests.
    """
    if item.get_closest_marker("gpu") is None:
        return
    try:
        import torch
    except ImportError:
        pytest.skip("gpu test requires torch")
    if not torch.cuda.is_available():
        pytest.skip("gpu test requires CUDA")


@pytest.fixture
def gpu_device() -> int:
    """Resolve the GPU ordinal tests should use.

    Honors ``CUDA_VISIBLE_DEVICES`` when set — the first listed device maps to
    logical ordinal 0 under PyTorch's device masking. Falls back to 0.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return 0  # logical ordinal under CUDA_VISIBLE_DEVICES masking
    return 0


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Deterministic seed for every test in this package."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def reset_cuda_state_between_tests(request: pytest.FixtureRequest) -> Iterator[None]:
    """Empty the CUDA allocator cache + run gc between slow tests.

    Applied automatically to any test carrying the ``slow`` marker. Runs
    before and after the test so a slow test can't leak fragmented
    allocator state into the next test (at least within the limits of a
    single CUDA context — full isolation still requires process forking).

    No-op on CPU-only hosts or for non-slow tests, keeping the fast
    unit-test lane cost-free.
    """
    is_slow = request.node.get_closest_marker("slow") is not None
    if not is_slow:
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
