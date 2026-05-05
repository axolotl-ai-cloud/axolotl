"""Unit tests for :class:`axolotl.integrations.protrain.runtime.scheduler.Scheduler`.

Focused on scheduler-internal invariants that don't require CUDA — in
particular the block-contiguity ownership rule for chunks shared
between adjacent transformer blocks (§3.1.1). When a single chunk
holds params for more than one consecutive block, only the EARLIEST
forward-order owner (= LAST block visited in backward) may finalize
the chunk's grads via ``reduce_grads_and_offload`` — otherwise the
LATER-forward block visits first in backward and finalizes before the
earlier block has produced its grads.
"""

from __future__ import annotations

from typing import cast

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    ParamId,
)


class _RecordingChunkManager:
    """Minimal stand-in for ``ChunkManager`` capturing the calls the test cares about.

    The scheduler's ``post_block_backward`` only invokes
    ``reduce_grads_and_offload`` on this object; everything else
    exercised here is satisfied by the layout. A real ``ChunkManager``
    is far heavier (buffer pool, pinned host memory, optimizer plumbing)
    and not needed to validate the gating.
    """

    def __init__(self) -> None:
        self.reduce_calls: list[ChunkId] = []
        # ``post_block_backward`` doesn't touch ``buffer_pool`` directly,
        # but the attribute is read elsewhere in the scheduler — keep it
        # exposed so accidental future couplings fail loudly instead of
        # silently no-oping on ``None``.
        self.buffer_pool: object | None = None

    def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
        self.reduce_calls.append(chunk_id)


def _two_blocks_one_chunk_layout() -> ChunkLayout:
    """Construct a layout where blocks 0 and 1 share chunk 0.

    Mirrors the block-contiguity packing rule in
    ``chunk.layout.build_layout``: when the params of two consecutive
    blocks fit inside a single ``S_chunk``, both block ids point at the
    same chunk id in ``block_to_chunks``. We construct the layout
    directly here rather than re-running ``build_layout`` because the
    test only needs the ``block_to_chunks`` shape that triggers the
    ownership-overlap path.
    """
    p_b0 = cast(ParamId, "transformer.h.0.weight")
    p_b1 = cast(ParamId, "transformer.h.1.weight")
    return ChunkLayout(
        S_chunk=1 << 20,
        N_chunk=1,
        chunks=((p_b0, p_b1),),
        param_to_chunk={p_b0: cast(ChunkId, 0), p_b1: cast(ChunkId, 0)},
        block_to_chunks={
            cast(BlockId, 0): (cast(ChunkId, 0),),
            cast(BlockId, 1): (cast(ChunkId, 0),),
        },
    )


def test_post_block_backward_only_finalizes_at_last_owner() -> None:
    """Shared chunks must be finalized exactly once, by the EARLIEST forward owner.

    Setup
    -----
    Layout where block 0 and block 1 each list chunk 0 in
    ``block_to_chunks`` (the block-contiguity rule packs them together
    when they fit). Backward order is reverse-forward: block 1 first,
    then block 0.

    Expectation
    -----------
    ``post_block_backward(1)`` must NOT call
    ``reduce_grads_and_offload(0)`` (block 0 hasn't produced grads
    yet). ``post_block_backward(0)`` MUST call
    ``reduce_grads_and_offload(0)`` exactly once. Net: a single
    finalization call, fired at the right time.

    Regression
    ----------
    Before the fix, ``post_block_backward`` finalized every chunk in
    ``self._chunks_for(block_id)`` unconditionally — block 1's
    backward would finalize chunk 0 prematurely (best case: an extra
    regather/offload cycle; worst case: double-finalize of the
    chunk's CPU-optim state).
    """
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

    layout = _two_blocks_one_chunk_layout()
    block_map: BlockStrategyMap = {
        cast(BlockId, 0): BlockMode.NONE,
        cast(BlockId, 1): BlockMode.NONE,
    }
    chunk_manager = _RecordingChunkManager()
    scheduler = Scheduler(
        chunk_manager=cast("object", chunk_manager),  # type: ignore[arg-type]
        block_map=block_map,
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )

    # Sanity: the precomputed map should pin chunk 0 to block 0 (its
    # earliest forward owner = its last backward owner).
    assert scheduler._chunk_last_bwd_owner == {cast(ChunkId, 0): cast(BlockId, 0)}

    # Backward order is reverse-forward: block 1 finalizes first.
    scheduler.post_block_backward(cast(BlockId, 1))
    assert chunk_manager.reduce_calls == [], (
        "post_block_backward(1) must NOT finalize chunk 0 — block 0 "
        "still owes grads. Got: "
        f"{chunk_manager.reduce_calls!r}"
    )

    # Now the EARLIEST forward owner runs; it owns the finalize.
    scheduler.post_block_backward(cast(BlockId, 0))
    assert chunk_manager.reduce_calls == [cast(ChunkId, 0)], (
        "post_block_backward(0) must finalize chunk 0 exactly once. Got: "
        f"{chunk_manager.reduce_calls!r}"
    )


def test_post_block_backward_unshared_chunks_finalize_normally() -> None:
    """Each block finalizes its own chunk when no chunk is shared.

    Negative control for the gating: if ``block_to_chunks`` has a
    one-to-one shape (no overlap), every block must still finalize its
    own chunks on its own ``post_block_backward`` call. The fix must
    not regress the common case.
    """
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

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
        cast(BlockId, 0): BlockMode.NONE,
        cast(BlockId, 1): BlockMode.NONE,
    }
    chunk_manager = _RecordingChunkManager()
    scheduler = Scheduler(
        chunk_manager=cast("object", chunk_manager),  # type: ignore[arg-type]
        block_map=block_map,
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )

    assert scheduler._chunk_last_bwd_owner == {
        cast(ChunkId, 0): cast(BlockId, 0),
        cast(ChunkId, 1): cast(BlockId, 1),
    }

    scheduler.post_block_backward(cast(BlockId, 1))
    scheduler.post_block_backward(cast(BlockId, 0))
    assert chunk_manager.reduce_calls == [cast(ChunkId, 1), cast(ChunkId, 0)], (
        "Each block should finalize its own (unshared) chunk on its own "
        f"post_block_backward call. Got: {chunk_manager.reduce_calls!r}"
    )
