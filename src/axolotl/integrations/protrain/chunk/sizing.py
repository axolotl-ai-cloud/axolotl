"""S_chunk grid search over the {32, 64, 128, 256} MB grid (Appendix B.1).

We score each candidate by simulating the *exact* packing rules
:func:`build_layout` would apply (block-sealing and block-contiguity from
§3.1.1) and selecting the S_chunk that minimizes the summed
``S_chunk - bytes_used`` across non-tail chunks. Ties are broken toward the
larger candidate so the layout uses fewer, larger chunks. The simulation
matches :func:`build_layout`'s placement exactly: both call
:func:`_pack_chunks_with_block_rules` from ``layout.py`` so the grid search
chooses the same fragmentation-optimal S_chunk that the actual layout will
produce — there is no heuristic skew between simulation and reality.

The signature accepts an ``exec_order`` and ``block_spans`` so simulation can
honor block-grouping. Callers that don't track blocks (legacy / tests) may
omit them and the simulation degrades to plain greedy fit, which is the
same answer ``build_layout`` would produce when ``block_spans`` is empty.
"""

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
    """Return total fragmentation waste under the exact ``build_layout`` rules.

    Drives the same step-based packer :func:`build_layout` uses, then sums the
    per-chunk slack ``S_chunk - bytes_used`` across every chunk except the
    last. The trailing chunk's slack is excluded — it is the natural tail and
    no choice of S_chunk recovers those bytes (a smaller S_chunk would just
    spill into yet another chunk with its own slack).
    """
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    steps = _build_packing_steps(param_sizes, exec_order, block_spans)
    chunk_bytes = _pack_chunks_with_block_rules(steps, S_chunk)
    if len(chunk_bytes) <= 1:
        return 0
    # Exclude the tail chunk from waste accounting — its slack is inherent.
    # Clamp negative values: an oversize tensor placed alone in its own chunk
    # legitimately exceeds S_chunk (build_layout's "place oversize on own
    # chunk" path), so ``S_chunk - bytes_used`` is negative and would
    # erroneously *credit* the layout for the overflow.
    return sum(max(0, S_chunk - b) for b in chunk_bytes[:-1])


def pick_S_chunk(
    model_state_bytes_per_param: dict[ParamId, int],
    candidates: tuple[int, ...] = DEFAULT_GRID,
    exec_order: Sequence[ParamId] | None = None,
    block_spans: Mapping[BlockId, Sequence[ParamId]] | None = None,
) -> int:
    """Pick the ``S_chunk`` from ``candidates`` minimizing fragmentation waste.

    The simulation iterates ``model_state_bytes_per_param`` in dict insertion
    order (Python 3.7+ guarantee), so callers MUST insert params in the
    intended layout/execution order — pass a plain ``dict[ParamId, int]``
    (or a subclass that preserves insertion order). The signature is
    intentionally typed as ``dict`` rather than ``Mapping`` because
    ``Mapping`` does not contract a stable iteration order, and the result
    of this function depends on it.

    When ``exec_order`` and ``block_spans`` are supplied, the simulation
    honors block-sealing/contiguity exactly as :func:`build_layout` does;
    when omitted, the simulation degrades to plain greedy fit (which is what
    ``build_layout`` itself does for an empty ``block_spans``).

    Ties are broken by picking the *larger* candidate — fewer chunks means
    less scheduler overhead and larger individual H2D transfers, both of
    which are strictly preferable at equal waste (App B.1 motivation).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # Default arguments mirror build_layout(empty block_spans, exec_order =
    # dict insertion order). These produce a plain greedy-fit simulation,
    # bit-for-bit equivalent to the previous heuristic when no block info
    # is available.
    if exec_order is None:
        exec_order = list(model_state_bytes_per_param.keys())
    if block_spans is None:
        block_spans = {}

    # Drop non-positive candidates up front: _simulate_waste rejects them with
    # ValueError, and they're never meaningful S_chunk values. Filtering here
    # keeps the baseline-selection invariant ``candidates[0] > 0`` so we never
    # hand a zero/negative size to _simulate_waste below.
    positive = tuple(S for S in candidates if S > 0)
    if not positive:
        raise ValueError(
            f"candidates must contain at least one positive S_chunk; got {candidates}"
        )
    candidates = positive

    # ``build_layout`` supports placing an oversize tensor in its own
    # chunk without splitting (see ``layout.py``: "A single param larger
    # than ``S_chunk`` is placed on its own in a fresh chunk"), and
    # ``_simulate_waste`` already models that path correctly: the
    # ``max(0, S_chunk - bytes_used)`` clamp prevents an oversize chunk
    # from crediting a too-small candidate with negative waste. With the
    # clamp in place, smaller candidates remain legal members of the
    # search and the simulator's tie-break (prefer larger S at equal
    # waste) handles preference cleanly. We therefore keep ALL positive
    # candidates in the search and let ``_simulate_waste`` plus the
    # tie-break (prefer larger S at equal waste) decide — no soft
    # feasibility-filter fallback is needed.

    best_S = candidates[0]
    best_waste = _simulate_waste(
        model_state_bytes_per_param, exec_order, block_spans, best_S
    )
    for S in candidates[1:]:
        waste = _simulate_waste(model_state_bytes_per_param, exec_order, block_spans, S)
        if waste < best_waste or (waste == best_waste and S > best_S):
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
