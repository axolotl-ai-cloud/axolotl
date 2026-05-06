"""S_chunk grid search over the {32, 64, 128, 256} MB grid (Appendix B.1).

We score each candidate by a simple greedy-fit fragmentation heuristic
(summed ``S_chunk - bytes_used`` across non-tail chunks) and pick the
minimizer, breaking ties toward the larger candidate. The input is a
``{ParamId -> bytes}`` map so this runs without a model handle.

Note: ``_simulate_waste`` is a *heuristic* approximation of
``build_layout``'s placement, NOT a full simulation. ``build_layout``
honors block-sealing and block-contiguity rules (Appendix B.1) that the
heuristic ignores, so the chosen ``S_chunk`` may diverge from the
fragmentation-optimal one once the real layout is built. The grid is
small (4 entries), the candidates are within a single order of
magnitude, and the searcher's downstream cost model re-evaluates the
selection — so the heuristic's accuracy is sufficient for grid
selection. Treat ``pick_S_chunk`` as a coarse tie-break, not a
paper-fidelity step.
"""

from __future__ import annotations

from axolotl.integrations.protrain.types import ParamId
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Paper-specified grid; also duplicated in DESIGN.md §Design Decisions.
DEFAULT_GRID: tuple[int, ...] = (32 << 20, 64 << 20, 128 << 20, 256 << 20)


def _simulate_waste(sizes_in_order: list[int], S_chunk: int) -> int:
    """Return total fragmentation waste for a greedy-fit layout.

    Mirrors the non-block-grouped ``build_layout`` inner loop: open a fresh
    chunk once the next param wouldn't fit. The last chunk's trailing slack
    is *not* counted as waste — it's just the natural tail and the caller
    can't recover bytes by picking a different ``S_chunk``. Every earlier
    chunk contributes ``S_chunk - bytes_used``.
    """
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    chunk_bytes: list[int] = [0]
    for sz in sizes_in_order:
        cur = chunk_bytes[-1]
        if cur > 0 and cur + sz > S_chunk:
            chunk_bytes.append(0)
        chunk_bytes[-1] += sz

    if len(chunk_bytes) <= 1:
        return 0
    # Exclude the tail chunk from waste accounting — its slack is inherent.
    return sum(max(0, S_chunk - b) for b in chunk_bytes[:-1])


def pick_S_chunk(
    model_state_bytes_per_param: dict[ParamId, int],
    candidates: tuple[int, ...] = DEFAULT_GRID,
) -> int:
    """Pick the ``S_chunk`` from ``candidates`` minimizing fragmentation waste.

    The simulation iterates ``model_state_bytes_per_param`` in dict insertion
    order (Python 3.7+ guarantee), so callers MUST insert params in the
    intended layout/execution order — pass a plain ``dict[ParamId, int]``
    (or a subclass that preserves insertion order). The signature is
    intentionally typed as ``dict`` rather than ``Mapping`` because
    ``Mapping`` does not contract a stable iteration order, and the result
    of this function depends on it.

    Ties are broken by picking the *larger* candidate — fewer chunks means
    less scheduler overhead and larger individual H2D transfers, both of
    which are strictly preferable at equal waste (App B.1 motivation).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    sizes_in_order = list(model_state_bytes_per_param.values())

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

    # Prefer candidates that can hold the largest single param tensor
    # natively (S_chunk >= max_param_bytes): for such candidates,
    # ``_simulate_waste`` accurately reflects fragmentation. For smaller
    # candidates, any chunk whose sole occupant overflows ``S_chunk``
    # would contribute *zero* waste under the heuristic and could let a
    # too-small candidate win on a tie despite producing more chunks.
    # However, ``build_layout`` *does* support placing an oversize tensor
    # in its own chunk without splitting (see ``layout.py``: "A single
    # param larger than ``S_chunk`` is placed on its own in a fresh
    # chunk"). So if every candidate is smaller than the largest tensor
    # (e.g. an LLM with a >256 MiB embedding under the default 32–256
    # MiB grid), fall back to picking the *largest* candidate rather
    # than raising — that minimizes the number of single-tensor overflow
    # chunks while keeping the layout legal.
    max_param_bytes = max(sizes_in_order, default=0)
    feasible = tuple(S for S in candidates if S >= max_param_bytes)
    if feasible:
        candidates = feasible
    else:
        LOG.debug(
            "pick_S_chunk: no candidate >= max param tensor size (%d B); "
            "falling back to the largest grid entry to minimize the "
            "single-tensor overflow chunk count.",
            max_param_bytes,
        )
        candidates = (max(candidates),)

    best_S = candidates[0]
    best_waste = _simulate_waste(sizes_in_order, best_S)
    for S in candidates[1:]:
        waste = _simulate_waste(sizes_in_order, S)
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
