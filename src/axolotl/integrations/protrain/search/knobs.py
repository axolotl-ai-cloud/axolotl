"""Bound derivation for the ProTrain knob search."""

from __future__ import annotations

from collections import Counter

from axolotl.integrations.protrain.types import (
    Bounds,
    ChunkLayout,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def derive_bounds(trace: ProfilerTrace, layout: ChunkLayout) -> Bounds:
    """Derive (N_chunk, N_block, N_interval) bounds from trace + layout."""
    n_chunk = int(layout.N_chunk)
    n_block = len(trace.activation_sizes)

    # N_interval = forward ops/block; degenerate → 1.
    if n_block <= 0:
        n_interval = 1
    else:
        per_block: Counter[int] = Counter()
        for op in trace.op_order:
            if op.is_forward and op.block_id is not None:
                per_block[int(op.block_id)] += 1
        if per_block:
            # Mean (not min) so single-hot-op blocks don't tank the bound.
            n_interval = max(1, sum(per_block.values()) // max(1, n_block))
        else:
            # No block_id mapping; fall back to flat ratio.
            forward_op_count = sum(1 for op in trace.op_order if op.is_forward)
            n_interval = max(1, forward_op_count // max(1, n_block))

    LOG.debug(
        "derive_bounds: N_chunk=%d N_block=%d N_interval=%d",
        n_chunk,
        n_block,
        n_interval,
    )
    return Bounds(N_chunk=n_chunk, N_block=n_block, N_interval=n_interval)


__all__ = ["derive_bounds"]
