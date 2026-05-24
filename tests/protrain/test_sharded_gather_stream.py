"""Unit tests for routing Mode C forward sharded all_gather on the prefetch stream.

v73 hardware verification confirmed PR #17(b) closes the Mode B bs=2 hang
(rc=0 434s sps 4.942/rank) but exposed a separate Mode C bs=2 hang at
forward block=10. SLOW_OFFLOAD_REGATHER watchdog stayed silent and Mode C
searcher picked ``n_offload=4`` (only 4 chunks offloaded — too few for
OFFLOAD re-gather to dominate). Root cause traced to
``Scheduler.ensure_chunks_resident`` (LoRA-container fan-out path)
issuing the NCCL ``all_gather_into_tensor`` on the compute stream so
block N's compute serializes with block N-1's reconstruction.

These tests cover the structural invariants of the fix:

* ``ensure_chunks_resident`` routes the gather onto ``_prefetch_stream``
  when CUDA is available so the sharded all_gather overlaps compute;
* compute waits on the prefetch stream AFTER the gather so reads observe
  the writes;
* the CPU / no-prefetch-stream lane falls back to synchronous gather
  (correctness over perf when streams are unavailable);
* the inert short-circuit still bypasses everything;
* the SLOW_SHARDED_GATHER watchdog env var + threshold reader behaves;
* ``_gather_sharded`` accepts and threads the ``phase`` kwarg used by
  the watchdog and by upstream callers.

Mode B (``force_replicated_cpu_offload=True``, no shard) is structurally
untouched by this change — it does not go through ``_gather_sharded`` at
all (replicated chunks H2D-copy directly into the buffer), so the
correctness of PR #17(b)'s ``_offload_stream`` wiring is preserved.
"""

from __future__ import annotations

import inspect
import os
from typing import cast
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


class _RecordingChunkManager:
    """Minimal ChunkManager stand-in capturing gather() invocations."""

    def __init__(self) -> None:
        self.gather_calls: list[tuple[ChunkId, str, object | None]] = []
        self.buffer_pool: object | None = None

    def gather(
        self,
        chunk_id: ChunkId,
        phase: str = "forward_regather",
        stream: object | None = None,
    ) -> None:
        self.gather_calls.append((chunk_id, phase, stream))

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


def _make_scheduler(layout: ChunkLayout, block_map: BlockStrategyMap):
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

    chunk_manager = _RecordingChunkManager()
    return Scheduler(
        chunk_manager=cast("object", chunk_manager),  # type: ignore[arg-type]
        block_map=block_map,
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )


# ---------- ensure_chunks_resident streaming ----------


@pytest.mark.gpu
def test_ensure_chunks_resident_routes_gather_on_prefetch_stream() -> None:
    """ensure_chunks_resident must place the gather() call under torch.cuda.stream(_prefetch_stream)."""
    import torch

    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)

    if sched._prefetch_stream is None:
        pytest.skip("prefetch stream unavailable on this host (no CUDA)")

    captured_streams: list[torch.cuda.Stream] = []

    def _capture_gather(self, chunk_id, phase="forward_regather", stream=None):
        # Record whichever stream is current when gather() is invoked — this
        # is how we observe that ensure_chunks_resident actually placed us
        # under torch.cuda.stream(_prefetch_stream).
        captured_streams.append(torch.cuda.current_stream())

    # Patch the RecordingChunkManager's gather so it captures the stream
    # context the scheduler placed the call under.
    sched.chunk_manager.gather = _capture_gather.__get__(  # type: ignore[method-assign]
        sched.chunk_manager, type(sched.chunk_manager)
    )

    sched.ensure_chunks_resident((cast(ChunkId, 0), cast(ChunkId, 1)))

    assert len(captured_streams) == 2, (
        f"expected 2 gather() invocations, got {len(captured_streams)}"
    )
    for s in captured_streams:
        assert s == sched._prefetch_stream, (
            "ensure_chunks_resident must route gather() under "
            "torch.cuda.stream(_prefetch_stream); "
            f"observed current_stream={s!r} != _prefetch_stream={sched._prefetch_stream!r}"
        )


@pytest.mark.gpu
def test_ensure_chunks_resident_syncs_compute_after_prefetch() -> None:
    """Compute stream must wait_stream(prefetch) so reads observe the gather's writes."""
    import torch

    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)
    if sched._prefetch_stream is None:
        pytest.skip("prefetch stream unavailable")

    wait_calls: list[torch.cuda.Stream] = []
    orig_wait = torch.cuda.Stream.wait_stream

    def _record_wait(self, other):
        wait_calls.append(other)
        return orig_wait(self, other)

    with patch.object(torch.cuda.Stream, "wait_stream", _record_wait):
        sched.ensure_chunks_resident((cast(ChunkId, 0),))

    # The compute (current) stream must have waited on _prefetch_stream after
    # the gather was queued; this is the contract that lets the LoRA-container
    # post-hooks observe rebuilt chunk data.
    assert sched._prefetch_stream in wait_calls, (
        "compute stream must call wait_stream(_prefetch_stream) AFTER gathering "
        f"so block N reads see block N-1's writes. wait_stream calls: {wait_calls!r}"
    )


def test_ensure_chunks_resident_inert_short_circuit() -> None:
    """When _is_inert is True the method must early-return WITHOUT touching any stream / gather."""
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)
    sched._is_inert = True

    sched.ensure_chunks_resident((cast(ChunkId, 0), cast(ChunkId, 1)))
    # No gather() calls; inert configs must remain truly inert.
    assert sched.chunk_manager.gather_calls == []  # type: ignore[attr-defined]


def test_ensure_chunks_resident_empty_chunk_ids_no_op() -> None:
    """Empty chunk_ids must short-circuit before any stream work (hot-path invariant)."""
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)

    sched.ensure_chunks_resident(())
    assert sched.chunk_manager.gather_calls == []  # type: ignore[attr-defined]


def test_ensure_chunks_resident_cpu_fallback_invokes_synchronous_gather() -> None:
    """On CPU (no _prefetch_stream) the method must still gather synchronously."""
    layout, block_map = _two_block_layout(BlockMode.NONE, BlockMode.NONE)
    sched = _make_scheduler(layout, block_map)
    # Force the CPU lane regardless of host: _prefetch_stream None ⇒ no stream context.
    sched._has_cuda = False
    sched._prefetch_stream = None

    sched.ensure_chunks_resident((cast(ChunkId, 0), cast(ChunkId, 1)))
    cids = [c[0] for c in sched.chunk_manager.gather_calls]  # type: ignore[attr-defined]
    assert cids == [cast(ChunkId, 0), cast(ChunkId, 1)], (
        f"CPU fallback must gather both chunks synchronously; got {cids!r}"
    )


# ---------- Mode B (replicated) preserves behavior ----------


def test_mode_b_replicated_path_does_not_use_gather_sharded() -> None:
    """Mode B (force_replicated_cpu_offload) chunks never enter _gather_sharded.

    Replicated chunks (no _chunk_shards entry) skip the all_gather path and
    H2D-copy directly into the buffer in _rebind_params_to_buffer. PR #17(b)'s
    _offload_stream wiring remains the only stream-route for Mode B; this
    sharded-gather change must not affect that flow.
    """
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    src = inspect.getsource(manager_mod.ChunkManager._gather_impl_body)
    # Structural assert: the body branches on shard_state and ONLY enters
    # _gather_sharded when shard_state is not None.
    assert "if shard_state is not None:" in src, (
        "_gather_impl_body must keep the shard_state guard so replicated "
        "(Mode B) chunks bypass _gather_sharded"
    )
    assert "self._gather_sharded(" in src
    # The replicated lane (Mode B) must still rebind directly (needs_copy=True).
    assert "needs_copy=True" in src, (
        "replicated lane must still call _rebind_params_to_buffer with "
        "needs_copy=True (Mode B path)"
    )


# ---------- _gather_sharded phase threading ----------


def test_gather_sharded_signature_accepts_phase_kwarg() -> None:
    """_gather_sharded must expose a ``phase`` kwarg for the SLOW_SHARDED_GATHER watchdog."""
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    sig = inspect.signature(manager_mod.ChunkManager._gather_sharded)
    assert "phase" in sig.parameters, (
        "_gather_sharded must accept phase= so the watchdog can tag "
        "forward vs backward gathers"
    )
    # Default must keep prior call sites compatible.
    assert sig.parameters["phase"].default == "forward_regather"


def test_gather_impl_threads_phase_into_gather_sharded() -> None:
    """_gather_impl(_body) must forward the phase kwarg into _gather_sharded."""
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    body_src = inspect.getsource(manager_mod.ChunkManager._gather_impl_body)
    assert "phase=phase" in body_src, (
        "_gather_impl_body must thread phase= into the _gather_sharded call"
    )
    impl_src = inspect.getsource(manager_mod.ChunkManager._gather_impl)
    assert "phase=phase" in impl_src, (
        "_gather_impl must thread phase= into _gather_impl_body"
    )


# ---------- SLOW_SHARDED_GATHER watchdog ----------


def test_slow_sharded_gather_threshold_default() -> None:
    """Default threshold is 5.0s, matching the SLOW_OFFLOAD_REGATHER convention."""
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    # Re-read with no env var set so we observe the documented default.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S", None)
        assert manager_mod._slow_sharded_gather_threshold_s() == pytest.approx(5.0)


def test_slow_sharded_gather_threshold_env_override() -> None:
    """PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S overrides the default; bad values fall back to 5.0."""
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    with patch.dict(os.environ, {"PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S": "0.5"}):
        assert manager_mod._slow_sharded_gather_threshold_s() == pytest.approx(0.5)
    with patch.dict(os.environ, {"PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S": "0"}):
        assert manager_mod._slow_sharded_gather_threshold_s() == pytest.approx(0.0)
    with patch.dict(os.environ, {"PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S": "garbage"}):
        assert manager_mod._slow_sharded_gather_threshold_s() == pytest.approx(5.0)
    with patch.dict(os.environ, {"PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S": "-1.0"}):
        # Negative clamped to 0 (disable).
        assert manager_mod._slow_sharded_gather_threshold_s() == pytest.approx(0.0)


def test_slow_sharded_gather_module_constant_present() -> None:
    """Module-level _SLOW_SHARDED_GATHER_S constant exists and is readable."""
    from axolotl.integrations.protrain.chunk import manager as manager_mod

    assert hasattr(manager_mod, "_SLOW_SHARDED_GATHER_S")
    assert isinstance(manager_mod._SLOW_SHARDED_GATHER_S, float)


# ---------- Edge cases ----------


def test_ensure_chunks_resident_uses_prefetch_not_compute_for_streamed_gather() -> None:
    """Structural assert: the streamed lane references _prefetch_stream, not compute."""
    from axolotl.integrations.protrain.runtime import scheduler as sched_mod

    src = inspect.getsource(sched_mod.Scheduler.ensure_chunks_resident)
    # The fix marker: gather must run under torch.cuda.stream(_prefetch_stream).
    assert "torch.cuda.stream(self._prefetch_stream)" in src or (
        "_torch.cuda.stream(self._prefetch_stream)" in src
    ), (
        "ensure_chunks_resident must wrap gather() in "
        "torch.cuda.stream(self._prefetch_stream) so sharded all_gather "
        "overlaps compute instead of serializing with it"
    )
    # Compute must wait on prefetch AFTER (post-gather sync).
    assert "compute.wait_stream(self._prefetch_stream)" in src, (
        "compute stream must wait_stream(_prefetch_stream) after the gather "
        "is queued so block N reads see the writes"
    )


def test_ensure_chunks_resident_does_not_wait_on_offload_or_swap_on_compute() -> None:
    """The fix moves swap/offload wait_stream barriers off the compute stream onto the prefetch stream.

    Pre-fix code called compute.wait_stream(swap/prefetch/offload) before
    running the gather on compute. Post-fix code calls
    prefetch.wait_stream(swap/offload) so prefetch sequences correctly
    behind swap/offload, then issues gather, then compute.wait_stream(prefetch).
    """
    from axolotl.integrations.protrain.runtime import scheduler as sched_mod

    src = inspect.getsource(sched_mod.Scheduler.ensure_chunks_resident)
    # Post-fix: prefetch.wait_stream(swap) and prefetch.wait_stream(offload).
    assert "self._prefetch_stream.wait_stream(self._swap_stream)" in src, (
        "ensure_chunks_resident must sequence prefetch behind swap "
        "(prefetch.wait_stream(swap)) so pool buffers are coherent"
    )
    assert "self._prefetch_stream.wait_stream(self._offload_stream)" in src, (
        "ensure_chunks_resident must sequence prefetch behind offload "
        "(prefetch.wait_stream(offload)) so in-flight backward re-gather "
        "drains before forward LoRA-container fan-out issues its gather"
    )


__all__: list[str] = []
