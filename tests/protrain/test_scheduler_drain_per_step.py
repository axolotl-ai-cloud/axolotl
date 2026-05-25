"""Tests for per-step ``Scheduler.drain()`` invocation from ``_ProTrainOptimizer.step()``.

The Scheduler.drain() method synchronizes the three side-streams
(_prefetch_stream, _swap_stream, _offload_stream), flushes deferred
offloads, and waits on CPU-Adam. It is the documented step-boundary
finalize hook (see Scheduler.__init__ comment "Step boundaries are
inferred from drain() calls"). These tests assert that the optimizer
step actually invokes it via the ChunkManager -> Scheduler back-ref.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from axolotl.integrations.protrain.api.optim_wrapper import _ProTrainOptimizer


class _FakeScheduler:
    def __init__(self) -> None:
        self.drain_calls = 0

    def drain(self) -> None:
        self.drain_calls += 1


class _FakeChunkManager:
    """Minimal stand-in exposing the surface _ProTrainOptimizer.step() touches."""

    def __init__(self, scheduler: _FakeScheduler | None = None) -> None:
        self.zero3_shard = False
        self._non_persistent_ids: list[Any] = []
        self.rank = 0
        self.world_size = 1
        self.wait_cpu_optim_all_calls = 0
        if scheduler is not None:
            self._scheduler_ref = scheduler

    def wait_cpu_optim_all(self) -> None:
        self.wait_cpu_optim_all_calls += 1


def _build_optimizer(chunk_manager: _FakeChunkManager) -> _ProTrainOptimizer:
    """Build a _ProTrainOptimizer wired to the fake chunk manager (no GPU/CPU adapters)."""
    params = [torch.nn.Parameter(torch.zeros(1))]
    return _ProTrainOptimizer(
        gpu_optim=None,
        cpu_optim=None,
        params=params,
        defaults={"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        chunk_manager=chunk_manager,
    )


# ---------------------------------------------------------------------------
# Test 1: optimizer.step() invokes scheduler.drain() exactly once per call.
# ---------------------------------------------------------------------------


def test_step_calls_scheduler_drain_once_per_step() -> None:
    scheduler = _FakeScheduler()
    mgr = _FakeChunkManager(scheduler=scheduler)
    optim = _build_optimizer(mgr)

    optim.step()
    assert scheduler.drain_calls == 1, (
        f"Expected exactly one drain() call after step(); got {scheduler.drain_calls}"
    )
    assert mgr.wait_cpu_optim_all_calls == 1

    optim.step()
    optim.step()
    assert scheduler.drain_calls == 3, (
        "drain() must fire on every step (not just the first)"
    )
    assert mgr.wait_cpu_optim_all_calls == 3


def test_step_without_scheduler_ref_is_noop() -> None:
    """Pre-attach / detach path: missing back-ref is OK (no crash, no drain)."""
    mgr = _FakeChunkManager(scheduler=None)
    assert not hasattr(mgr, "_scheduler_ref")
    optim = _build_optimizer(mgr)

    optim.step()
    assert mgr.wait_cpu_optim_all_calls == 1


def test_step_drain_runs_after_wait_cpu_optim_all() -> None:
    """drain() must run AFTER wait_cpu_optim_all() — the order matters for in-flight CPU-Adam."""
    call_order: list[str] = []

    class _OrderedFakeChunkManager(_FakeChunkManager):
        def wait_cpu_optim_all(self) -> None:
            call_order.append("wait_cpu_optim_all")
            super().wait_cpu_optim_all()

    class _OrderedFakeScheduler(_FakeScheduler):
        def drain(self) -> None:
            call_order.append("drain")
            super().drain()

    scheduler = _OrderedFakeScheduler()
    mgr = _OrderedFakeChunkManager(scheduler=scheduler)
    optim = _build_optimizer(mgr)

    optim.step()
    assert call_order == ["wait_cpu_optim_all", "drain"], (
        f"Unexpected call order: {call_order}"
    )


# ---------------------------------------------------------------------------
# Test 2: Scheduler.drain() semantics — synchronizes all three side streams.
# ---------------------------------------------------------------------------


class _StubStream:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def _make_scheduler_with_stub_streams(*, has_cuda: bool, with_offload: bool) -> Any:
    """Build a Scheduler with stub streams + chunk_manager, bypassing _init_streams.

    Avoids touching real CUDA so the test runs on CPU CI.
    """
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched._has_cuda = has_cuda
    sched._is_inert = False
    sched._prefetch_stream = _StubStream() if has_cuda else None
    sched._swap_stream = _StubStream() if has_cuda else None
    sched._offload_stream = _StubStream() if (has_cuda and with_offload) else None
    sched._first_iter_trace_enabled = False
    sched._step_timing_enabled = False
    sched._step_timing_step_idx = 0
    sched._step_timing_emit_every = 1
    sched._rank = 0

    drain_deferred_calls = {"n": 0}
    wait_cpu_optim_calls = {"n": 0}

    class _StubChunkManager:
        def drain_deferred_offloads(self) -> None:
            drain_deferred_calls["n"] += 1

        def wait_cpu_optim(self) -> None:
            wait_cpu_optim_calls["n"] += 1

    sched.chunk_manager = _StubChunkManager()  # type: ignore[assignment]
    return sched, drain_deferred_calls, wait_cpu_optim_calls


def test_drain_synchronizes_all_three_side_streams() -> None:
    sched, drain_deferred_calls, wait_cpu_optim_calls = (
        _make_scheduler_with_stub_streams(has_cuda=True, with_offload=True)
    )

    sched.drain()

    assert sched._prefetch_stream.synchronize_calls == 1
    assert sched._swap_stream.synchronize_calls == 1
    assert sched._offload_stream.synchronize_calls == 1
    assert drain_deferred_calls["n"] == 1
    assert wait_cpu_optim_calls["n"] == 1


def test_drain_without_offload_stream_skips_third_sync() -> None:
    """Mode B (n_offload=0) leaves _offload_stream=None; drain() must not raise."""
    sched, _, _ = _make_scheduler_with_stub_streams(has_cuda=True, with_offload=False)

    sched.drain()

    assert sched._prefetch_stream.synchronize_calls == 1
    assert sched._swap_stream.synchronize_calls == 1
    assert sched._offload_stream is None


def test_drain_on_cpu_only_path_is_safe() -> None:
    """No-CUDA host: drain() must skip stream syncs but still drain deferreds + cpu_optim."""
    sched, drain_deferred_calls, wait_cpu_optim_calls = (
        _make_scheduler_with_stub_streams(has_cuda=False, with_offload=False)
    )

    sched.drain()

    assert sched._prefetch_stream is None
    assert sched._swap_stream is None
    assert sched._offload_stream is None
    assert drain_deferred_calls["n"] == 1
    assert wait_cpu_optim_calls["n"] == 1


# ---------------------------------------------------------------------------
# Test 3: chunk_manager._scheduler_ref is wired at runtime construction.
# ---------------------------------------------------------------------------


def test_model_wrapper_wires_scheduler_ref_onto_chunk_manager() -> None:
    """The construction site in api/model_wrapper.py must set chunk_manager._scheduler_ref.

    This is a static read of the source so we can run it without spinning up
    a full ProTrain runtime. If the assignment is ever moved or deleted,
    this test catches it.
    """
    import inspect

    from axolotl.integrations.protrain.api import model_wrapper

    src = inspect.getsource(model_wrapper._construct_runtime)
    assert "chunk_manager._scheduler_ref = scheduler" in src, (
        "Expected `chunk_manager._scheduler_ref = scheduler` wiring in "
        "_construct_runtime — drain() at step() boundary depends on this back-ref."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
