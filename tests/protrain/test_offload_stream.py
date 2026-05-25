"""Unit tests for the dedicated OFFLOAD backward re-gather stream.

PR #17(b) adds ``Scheduler._offload_stream`` so per-chunk H2D + NCCL
``all_gather_into_tensor`` invoked from backward hooks runs on its own
CUDA stream, overlapping with backward compute on the default stream.
On non-NVLink topology this closes the bs=2 + n_offload > 0 hang
verified in v71/v72-redux.

These tests cover the structural invariants:

* the stream is only allocated when at least one block is in OFFLOAD mode
  (no per-step overhead on inert / no-offload / all-SWAP configs),
* the stream is exposed via the ``offload_stream`` property,
* ``ChunkManager.gather()`` honors the optional ``stream=`` kwarg and
  wraps the body in ``torch.cuda.stream(stream)``,
* ``ChunkManager.gather_for_backward()`` threads the kwarg through,
* the scheduler's backward path issues
  ``compute.wait_stream(_offload_stream)`` so backward compute observes
  the re-gather's writes,
* inert / CPU paths leave ``_offload_stream`` as ``None`` and never
  trip a ``wait_stream`` call.

CUDA is required for the real-stream tests; CPU-only paths fall through
to assertion of ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    ParamId,
)

if TYPE_CHECKING:
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler


class _RecordingChunkManager:
    """Minimal ChunkManager stand-in (mirrors test_scheduler.py)."""

    def __init__(self) -> None:
        self.gather_calls: list[tuple[ChunkId, str]] = []
        self.buffer_pool: object | None = None

    def gather(
        self,
        chunk_id: ChunkId,
        phase: str = "forward_regather",
        stream: object | None = None,
    ) -> None:
        self.gather_calls.append((chunk_id, phase))

    def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
        pass


def _two_block_layout(
    mode_a: BlockMode = BlockMode.NONE,
    mode_b: BlockMode = BlockMode.NONE,
) -> tuple[ChunkLayout, BlockStrategyMap]:
    p_b0 = cast(ParamId, "transformer.h.0.weight")
    p_b1 = cast(ParamId, "transformer.h.1.weight")
    layout = ChunkLayout(
        S_chunk=1 << 20,
        N_chunk=2,
        chunks=((p_b0,), (p_b1,)),
        param_to_chunk={p_b0: cast(ChunkId, 0), p_b1: cast(ChunkId, 1)},
        block_to_chunks={
            cast(BlockId, 0): (cast(ChunkId, 0),),
            cast(BlockId, 1): (cast(ChunkId, 1),),
        },
    )
    block_map: BlockStrategyMap = {
        cast(BlockId, 0): mode_a,
        cast(BlockId, 1): mode_b,
    }
    return layout, block_map


def _make_scheduler(
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
) -> "Scheduler":
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

    chunk_manager = _RecordingChunkManager()
    return Scheduler(
        chunk_manager=cast("object", chunk_manager),  # type: ignore[arg-type]
        block_map=block_map,
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )


# ---------- Scheduler-side allocation invariants ----------


def test_offload_stream_none_when_no_offload_blocks() -> None:
    """No OFFLOAD blocks -> _offload_stream stays None (no per-step overhead)."""
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)
    assert sched._offload_stream is None
    assert sched.offload_stream is None


def test_offload_stream_none_for_ckpt_only_layout() -> None:
    """All-CKPT layout (no OFFLOAD) -> _offload_stream stays None."""
    layout, block_map = _two_block_layout(BlockMode.CKPT, BlockMode.CKPT)
    sched = _make_scheduler(layout, block_map)
    assert sched._offload_stream is None


def test_offload_stream_none_for_swap_only_layout() -> None:
    """SWAP-only is a separate path; _offload_stream is NOT shared with _swap_stream."""
    layout, block_map = _two_block_layout(BlockMode.SWAP, BlockMode.SWAP)
    sched = _make_scheduler(layout, block_map)
    assert sched._offload_stream is None


@pytest.mark.gpu
def test_offload_stream_created_when_offload_block_present() -> None:
    """At least one OFFLOAD block -> dedicated _offload_stream is allocated and != _swap_stream."""
    import torch

    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.OFFLOAD)
    sched = _make_scheduler(layout, block_map)
    assert sched._offload_stream is not None
    assert isinstance(sched._offload_stream, torch.cuda.Stream)
    # Critical: must not collide with the swap stream — SWAP and OFFLOAD
    # are independent paths and must run in parallel.
    assert sched._offload_stream is not sched._swap_stream
    assert sched._offload_stream is not sched._prefetch_stream
    # Public accessor must expose the same handle.
    assert sched.offload_stream is sched._offload_stream


@pytest.mark.gpu
def test_offload_stream_not_torch_default_stream() -> None:
    """The OFFLOAD stream must be a non-default stream (it's the whole point)."""
    import torch

    layout, block_map = _two_block_layout(BlockMode.OFFLOAD, BlockMode.OFFLOAD)
    sched = _make_scheduler(layout, block_map)
    assert sched._offload_stream is not None
    default = torch.cuda.default_stream(device=sched._offload_stream.device)
    assert sched._offload_stream != default


# ---------- ChunkManager API: stream= kwarg ----------


def test_chunk_manager_gather_accepts_stream_kwarg() -> None:
    """ChunkManager.gather must accept the new ``stream=`` kwarg (signature compatibility)."""
    import inspect

    from axolotl.integrations.protrain.chunk import manager as manager_mod

    sig = inspect.signature(manager_mod.ChunkManager.gather)
    assert "stream" in sig.parameters, (
        "ChunkManager.gather must expose a stream= kwarg for the dedicated "
        "_offload_stream wiring"
    )
    # Default must be None so existing call sites are unaffected.
    assert sig.parameters["stream"].default is None


def test_chunk_manager_gather_for_backward_accepts_stream_kwarg() -> None:
    """gather_for_backward must forward stream= through to gather()."""
    import inspect

    from axolotl.integrations.protrain.chunk import manager as manager_mod

    sig = inspect.signature(manager_mod.ChunkManager.gather_for_backward)
    assert "stream" in sig.parameters
    assert sig.parameters["stream"].default is None


@pytest.mark.gpu
def test_gather_stream_kwarg_wraps_torch_cuda_stream_context() -> None:
    """When stream= is provided and CUDA is available, gather wraps the body in torch.cuda.stream(stream)."""
    import torch

    from axolotl.integrations.protrain.chunk import manager as manager_mod

    streams_entered: list[object] = []

    def _capture_body(self, chunk_id):
        # Record whichever stream is current when the body executes; this is
        # how we observe that gather() honored the stream= kwarg.
        streams_entered.append(torch.cuda.current_stream())

    real_stream = torch.cuda.Stream()
    cid = cast(ChunkId, 7)

    with patch.object(manager_mod.ChunkManager, "_gather_impl_body", _capture_body):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", 0.0):
                mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                mgr._persistent_ids = set()
                mgr._cpu_slots = {cid: []}
                mgr.gather(cid, stream=real_stream)

    assert streams_entered, (
        "expected _gather_impl_body to fire under the stream context"
    )
    assert streams_entered[0] == real_stream, (
        f"gather(stream=X) must run body on X; got {streams_entered[0]!r} "
        f"(expected {real_stream!r})"
    )


# ---------- Backward synchronization points ----------


@pytest.mark.gpu
def test_pre_block_backward_syncs_compute_with_offload_stream() -> None:
    """pre_block_backward must call compute.wait_stream(_offload_stream) before backward reads."""
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.OFFLOAD)
    sched = _make_scheduler(layout, block_map)

    sync_calls: list[str] = []
    orig_sync_off = sched._sync_offload_with_compute

    def _record_off():
        sync_calls.append("offload")
        return orig_sync_off()

    sched._sync_offload_with_compute = _record_off  # type: ignore[method-assign]

    # Block 1 is OFFLOAD. Triggering its pre-backward must issue at least
    # one offload-stream sync barrier (the entry sync) plus possibly a
    # second one after re-gathering misses; both go through the same
    # method so the count is >= 1.
    # buffer_pool is None on this stub which makes pre_block_backward bail
    # before reaching the offload-stream sync; for the structural assert
    # we install a minimal pool stub.
    class _StubPool:
        def lookup_resident(self, _cid):
            return None

    sched.chunk_manager.buffer_pool = _StubPool()  # type: ignore[attr-defined,assignment]

    sched.pre_block_backward(cast(BlockId, 1))
    assert "offload" in sync_calls, (
        "pre_block_backward on an OFFLOAD block must invoke "
        "_sync_offload_with_compute so backward compute waits for in-flight "
        "OFFLOAD re-gather"
    )


@pytest.mark.gpu
def test_inert_scheduler_does_not_create_offload_stream() -> None:
    """When _is_inert flips True, _offload_stream stays None (the inert path bypasses hook installation, but the constructor still ran — assert no allocation when block_map has no OFFLOAD)."""
    # The inert short-circuit triggers when n_persist == N_chunk AND no
    # OFFLOAD blocks; the latter alone already prevents stream creation.
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.CKPT)
    sched = _make_scheduler(layout, block_map)
    sched._is_inert = True
    # ensure_block_resident with _is_inert=True must early-return and not
    # touch any stream.
    sched.ensure_block_resident(cast(BlockId, 0))
    assert sched._offload_stream is None


# ---------- Defensive unpack-time stream threading ----------


def test_offload_block_unpack_threads_scheduler_offload_stream() -> None:
    """OffloadedBlock._unpack must pass the scheduler's _offload_stream to gather_for_backward."""
    import inspect

    from axolotl.integrations.protrain.block import offload as offload_mod

    src = inspect.getsource(offload_mod.OffloadedBlock._unpack)
    # Structural assert: the unpack path must reference _offload_stream
    # (via the scheduler) and forward it through gather_for_backward.
    assert "_offload_stream" in src, (
        "OffloadedBlock._unpack must look up the scheduler's _offload_stream "
        "so backward re-gather overlaps backward compute"
    )
    assert "stream=offload_stream" in src or "stream = offload_stream" in src, (
        "OffloadedBlock._unpack must thread the offload stream into "
        "gather_for_backward / gather"
    )


# ---------- Edge cases: NVLink / single-rank / inert ----------


def test_single_rank_replicated_layout_still_creates_offload_stream() -> None:
    """Even without ZeRO-3 sharding, OFFLOAD H2D benefits from the dedicated stream."""
    # NCCL all_gather isn't on the path for replicated layouts, but the
    # H2D copy in _rebind_params_to_buffer still benefits from running
    # off the compute stream — the gating is "any OFFLOAD block", not
    # "any sharded chunk".
    layout, block_map = _two_block_layout(BlockMode.OFFLOAD, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)
    # CPU-only stub: _has_cuda will be False so the stream is None and the
    # scheduler degrades to synchronous transfers (correctness, not perf).
    if not sched._has_cuda:
        assert sched._offload_stream is None
    else:
        assert sched._offload_stream is not None


__all__: list[str] = []
