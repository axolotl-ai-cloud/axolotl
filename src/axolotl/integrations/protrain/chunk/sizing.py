"""S_chunk grid search minimising fragmentation under build_layout's exact packing rules."""

from __future__ import annotations

from typing import Mapping, Sequence

from axolotl.integrations.protrain.chunk.layout import (
    _build_packing_steps,
    _pack_chunks_with_block_rules,
)
from axolotl.integrations.protrain.types import BlockId, ParamId
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Paper-specified grid; also duplicated in DESIGN.md §Design Decisions.
DEFAULT_GRID: tuple[int, ...] = (32 << 20, 64 << 20, 128 << 20, 256 << 20)


def _simulate_waste(
    param_sizes: Mapping[ParamId, int],
    exec_order: Sequence[ParamId],
    block_spans: Mapping[BlockId, Sequence[ParamId]],
    S_chunk: int,
) -> int:
    """Sum per-chunk slack (S_chunk - bytes_used) ex-tail; clamp ≥0 for oversize-on-own-chunk."""
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    steps = _build_packing_steps(param_sizes, exec_order, block_spans)
    chunk_bytes = _pack_chunks_with_block_rules(steps, S_chunk)
    if len(chunk_bytes) <= 1:
        return 0
    return sum(max(0, S_chunk - b) for b in chunk_bytes[:-1])


def pick_S_chunk(
    model_state_bytes_per_param: dict[ParamId, int],
    candidates: tuple[int, ...] = DEFAULT_GRID,
    exec_order: Sequence[ParamId] | None = None,
    block_spans: Mapping[BlockId, Sequence[ParamId]] | None = None,
) -> int:
    """Pick S_chunk minimising fragmentation waste; tie-break to smaller S to keep buffer ceiling tight."""
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # Defaults mirror build_layout(empty block_spans, exec_order = insertion order).
    if exec_order is None:
        exec_order = list(model_state_bytes_per_param.keys())
    if block_spans is None:
        block_spans = {}

    # Fail-fast on non-positive candidates instead of silent filtering.
    non_positive = tuple(S for S in candidates if S <= 0)
    if non_positive:
        raise ValueError(
            "candidates must all be positive S_chunk values; got non-positive "
            f"entries {non_positive} in {candidates}"
        )

    # All positive candidates remain legal; oversize is clamped in _simulate_waste.
    best_S = candidates[0]
    best_waste = _simulate_waste(
        model_state_bytes_per_param, exec_order, block_spans, best_S
    )
    for S in candidates[1:]:
        waste = _simulate_waste(model_state_bytes_per_param, exec_order, block_spans, S)
        if waste < best_waste or (waste == best_waste and S < best_S):
            best_S = S
            best_waste = waste

    LOG.debug(
        "pick_S_chunk: selected %d bytes (waste=%d) from grid %s",
        best_S,
        best_waste,
        candidates,
    )
    return best_S


__all__ = ["DEFAULT_GRID", "pick_S_chunk"]
