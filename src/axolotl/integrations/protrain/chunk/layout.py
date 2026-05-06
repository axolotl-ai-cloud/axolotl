"""Param-to-chunk assignment with execution-order intra-chunk reordering.

The ProTrain differentiator vs. Colossal-AI: intra-chunk ordering follows the
first-iteration *execution order*, not initialization order (§3.1.1). Shared
parameters keep their first-occurrence slot, and all parameters of a given
transformer block are forced into the same chunk when they fit — this
minimizes memory accesses when gradient checkpointing forces reverse-order
revisits in backward.

Paper references: §3.1.1, Appendix B.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence, cast

from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    ParamId,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

LOG = get_logger(__name__)

# A placement step describes one atomic packing decision the chunk packer must
# honor. We expose two flavours:
#
# * ``("single", size)`` — place a single non-block param via greedy fit. Seal
#   the current chunk first iff the param wouldn't fit alongside whatever is
#   already there. An oversize param (``size > S_chunk``) gets its own chunk.
# * ``("block", [size_0, size_1, ...])`` — place a block's params contiguously.
#   Seal the current chunk first iff the entire block wouldn't fit alongside
#   whatever is already there. Once the block starts, individual params are
#   placed via the same greedy fit so a block bigger than ``S_chunk`` spans
#   consecutive chunks but no foreign param can interleave.
#
# Sharing this packer between :func:`build_layout` and the S_chunk grid search
# is the paper-fidelity guarantee: the simulation's chunk count matches what
# the layout actually produces. App B.1 says "ProTrain conducts a grid search
# that simulates memory waste across various chunk sizes and selects the one
# that minimizes it" — that simulation must reflect the placement rules.
SingleStep = tuple[str, int]
BlockStep = tuple[str, list[int]]
PackingStep = tuple[str, object]  # union of SingleStep / BlockStep at runtime


def _pack_chunks_with_block_rules(
    steps: Sequence[PackingStep],
    S_chunk: int,
) -> list[int]:
    """Return per-chunk byte usage following the block-aware placement rules.

    Mirrors the inner placement loop of :func:`build_layout` exactly: starts
    with one empty chunk, processes each step in order, and seals the current
    chunk on overflow according to the same rules. The returned list is the
    cumulative bytes assigned to each chunk in order — same length as the
    chunk count :func:`build_layout` would have produced for the same step
    stream, except for the trailing-empty-chunk trim which we replicate at
    the end.
    """
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    chunk_bytes: list[int] = [0]

    def _seal_and_open() -> None:
        chunk_bytes.append(0)

    def _place(size: int) -> None:
        if chunk_bytes[-1] > 0 and chunk_bytes[-1] + size > S_chunk:
            _seal_and_open()
        chunk_bytes[-1] += size

    for kind, payload in steps:
        if kind == "single":
            assert isinstance(payload, int)
            _place(payload)
        elif kind == "block":
            assert isinstance(payload, list)
            block_total = sum(payload)
            if chunk_bytes[-1] > 0 and block_total > S_chunk - chunk_bytes[-1]:
                # Seal-before-block: keep the block chunk-aligned when it
                # won't fit alongside the current contents.
                _seal_and_open()
            for size in payload:
                _place(size)
        else:
            raise ValueError(f"unknown packing step kind: {kind!r}")

    # Drop a trailing empty chunk (matches build_layout's tail-trim).
    while len(chunk_bytes) > 1 and chunk_bytes[-1] == 0:
        chunk_bytes.pop()
    return chunk_bytes


def _build_packing_steps(
    param_sizes: Mapping[ParamId, int],
    exec_order: Sequence[ParamId],
    block_spans: Mapping[BlockId, Sequence[ParamId]],
) -> list[PackingStep]:
    """Translate (exec_order, block_spans) into a packer step stream.

    Replicates :func:`build_layout`'s exec-order walk: shared params kept at
    first occurrence, block params gathered contiguously starting from the
    block's first appearance, then any model params absent from exec_order
    appended at the tail (with leftover-block-grouping the same way).
    """
    placed: set[ParamId] = set()
    pid_to_block: dict[ParamId, BlockId | None] = {}
    pid_owner: dict[ParamId, BlockId] = {}
    for bid, params in block_spans.items():
        for pid in params:
            pid_owner[pid] = bid
    for pid in exec_order:
        pid_to_block[pid] = pid_owner.get(pid)

    steps: list[PackingStep] = []
    n = len(exec_order)
    i = 0
    while i < n:
        pid = exec_order[i]
        if pid in placed:
            i += 1
            continue
        block_id = pid_to_block.get(pid)
        if block_id is None:
            steps.append(("single", param_sizes[pid]))
            placed.add(pid)
            i += 1
            continue

        # Gather every still-unplaced member of this block, exec-order first
        # then any block params not in exec_order at all.
        block_member_set = set(block_spans[block_id])
        pending: list[ParamId] = []
        seen_pending: set[ParamId] = set()
        for j in range(i, n):
            qpid = exec_order[j]
            if (
                qpid in block_member_set
                and qpid not in placed
                and qpid not in seen_pending
            ):
                pending.append(qpid)
                seen_pending.add(qpid)
        for qpid in block_spans[block_id]:
            if qpid not in placed and qpid not in seen_pending:
                pending.append(qpid)
                seen_pending.add(qpid)

        steps.append(("block", [param_sizes[q] for q in pending]))
        placed.update(pending)
        i += 1

    # Tail: model params absent from exec_order. Same block-grouping rule.
    for pid in param_sizes:
        if pid in placed:
            continue
        tail_bid = pid_owner.get(pid)
        if tail_bid is None:
            steps.append(("single", param_sizes[pid]))
            placed.add(pid)
            continue
        pending = [q for q in block_spans[tail_bid] if q not in placed]
        steps.append(("block", [param_sizes[q] for q in pending]))
        placed.update(pending)

    return steps


def _param_bytes(model: "nn.Module") -> dict[ParamId, int]:
    """Return a {ParamId -> byte size} map for every named parameter in ``model``."""
    sizes: dict[ParamId, int] = {}
    for name, param in model.named_parameters():
        # numel * element_size is exact whether on meta, CPU, or CUDA.
        sizes[cast(ParamId, name)] = int(param.numel()) * int(param.element_size())
    return sizes


def _block_of(
    pid: ParamId, block_spans: Mapping[BlockId, Sequence[ParamId]]
) -> BlockId | None:
    """Find the ``BlockId`` owning ``pid``, or ``None`` if the param is unaffiliated.

    Linear scan; block_spans is typically small (N_block on the order of tens
    to low hundreds) and called once per unique param, so O(N_block) is fine.
    """
    for block_id, params in block_spans.items():
        # Membership test on a tuple/list is O(len(params)) but cheaper than
        # eagerly inverting the full mapping when the overwhelming majority
        # of params belong to exactly one block.
        if pid in params:
            return block_id
    return None


def build_layout(
    model: "nn.Module",
    exec_order: list[ParamId],
    S_chunk: int,
    block_spans: Mapping[BlockId, Sequence[ParamId]],
) -> ChunkLayout:
    """Assign params to fixed-size chunks in execution order.

    Algorithm (§3.1.1):

    1. Walk ``exec_order``. Track the current chunk's cumulative byte footprint.
       Skip params already placed (shared params keep the *first* occurrence
       slot — the paper's key eviction-ordering guarantee).
    2. If the next param belongs to a transformer block, try to place *all*
       remaining block params contiguously. If the full block fits in the
       current chunk's remaining budget, place it. Otherwise seal the current
       chunk and start a new one; the block's params become the new chunk's
       prefix. If the block is larger than ``S_chunk`` the block spills across
       consecutive chunks but its params remain contiguous (no non-block param
       may interleave).
    3. Non-block params follow the plain greedy fit rule.

    Returns a populated :class:`ChunkLayout` whose ``chunks`` ordering matches
    the execution order the scheduler will prefetch against.
    """
    if S_chunk <= 0:
        raise ValueError(f"S_chunk must be positive, got {S_chunk}")

    param_sizes = _param_bytes(model)

    # Validate exec_order entries.
    for pid in exec_order:
        if pid not in param_sizes:
            raise KeyError(
                f"exec_order references unknown param {pid!r}; "
                "not present in model.named_parameters()"
            )

    # Validate block_spans entries up front: every ParamId referenced by any
    # block must exist in the model, and no ParamId may belong to two blocks.
    # Without these checks, an unknown ParamId would be silently skipped on
    # the per-iteration ``param_sizes[pid]`` lookup path (or worse, raise
    # deep inside the placement loop with a confusing traceback), and an
    # overlapping ParamId would be silently assigned to the first block by
    # ``_block_of()`` so ``block_to_chunks`` would no longer reflect the
    # caller's spans. Fail fast at the API boundary instead.
    block_referenced: set[ParamId] = set()
    pid_owner: dict[ParamId, BlockId] = {}
    overlaps: dict[ParamId, list[BlockId]] = {}
    for owner_bid, params in block_spans.items():
        seen: set[ParamId] = set()
        for pid in params:
            # Within a single block, a duplicate ``pid`` would slip past
            # the cross-block ``pid_owner`` check (since ``prior == owner_bid``
            # falls through to the else branch) and double-count
            # downstream. Reject duplicates explicitly here.
            if pid in seen:
                raise ValueError(
                    f"block_spans[{owner_bid!r}] lists param {pid!r} more than "
                    "once; each ParamId must appear at most once per block"
                )
            seen.add(pid)
            prior = pid_owner.get(pid)
            if prior is not None and prior != owner_bid:
                bucket = overlaps.setdefault(pid, [prior])
                if owner_bid not in bucket:
                    bucket.append(owner_bid)
            else:
                pid_owner[pid] = owner_bid
            block_referenced.add(pid)
    if overlaps:
        overlap_sorted = sorted(
            f"{pid!r} -> [{', '.join(repr(b) for b in bids)}]"
            for pid, bids in overlaps.items()
        )
        raise ValueError(
            "block_spans contains param(s) assigned to multiple blocks: "
            + "; ".join(overlap_sorted)
        )
    missing_block_pids = block_referenced - param_sizes.keys()
    if missing_block_pids:
        missing_sorted = sorted(repr(p) for p in missing_block_pids)
        raise KeyError(
            f"block_spans references unknown param(s) {', '.join(missing_sorted)}; "
            "not present in model.named_parameters()"
        )

    chunks: list[list[ParamId]] = [[]]
    chunk_bytes: list[int] = [0]
    param_to_chunk: dict[ParamId, ChunkId] = {}
    block_to_chunks: dict[BlockId, list[ChunkId]] = {}

    def _seal_and_open() -> None:
        chunks.append([])
        chunk_bytes.append(0)

    def _place(pid: ParamId, size: int, block_id: BlockId | None) -> None:
        """Append ``pid`` to the current chunk, honoring ``S_chunk`` as a soft cap.

        A single param larger than ``S_chunk`` is placed on its own in a fresh
        chunk (the chunk will overflow the nominal cap but this is the only
        correct thing we can do without tensor splitting, which the M2 scope
        explicitly excludes).
        """
        nonlocal chunks, chunk_bytes
        cur_idx = len(chunks) - 1
        if chunk_bytes[cur_idx] > 0 and chunk_bytes[cur_idx] + size > S_chunk:
            _seal_and_open()
            cur_idx = len(chunks) - 1
        chunks[cur_idx].append(pid)
        chunk_bytes[cur_idx] += size
        cid = cast(ChunkId, cur_idx)
        param_to_chunk[pid] = cid
        if block_id is not None:
            bucket = block_to_chunks.setdefault(block_id, [])
            if not bucket or bucket[-1] != cid:
                bucket.append(cid)

    # Build fast inverse: which block (if any) owns each ParamId.
    # ``pid_owner`` was populated above while walking ``block_spans`` and
    # is the authoritative source — reuse it via ``.get()`` (returns
    # ``None`` for unaffiliated params) rather than rescanning
    # ``block_spans`` once per pid.
    pid_to_block: dict[ParamId, BlockId | None] = {
        pid: pid_owner.get(pid) for pid in exec_order
    }

    # Pre-compute the exec-order sequence of first occurrences of each block's
    # params. We need this to apply the "pack the whole block together" rule:
    # when we hit the first param of a block, we attempt to reserve space for
    # the entire block at once.
    i = 0
    n = len(exec_order)
    while i < n:
        pid = exec_order[i]
        if pid in param_to_chunk:
            # Shared param already placed at its first occurrence; skip.
            i += 1
            continue

        block_id: BlockId | None = pid_to_block.get(pid)
        if block_id is None:
            _place(pid, param_sizes[pid], None)
            i += 1
            continue

        # Gather every param of this block in exec_order starting from i,
        # skipping ones already placed (e.g. a block param shared with an
        # earlier op). We take params belonging to ``block_id`` in the order
        # they appear across the remaining exec_order — this is what "same
        # block grouped, exec-ordered within the block" means in practice.
        block_member_set = set(block_spans[block_id])
        pending: list[ParamId] = []
        seen_in_pending: set[ParamId] = set()
        for j in range(i, n):
            qpid = exec_order[j]
            if (
                qpid in block_member_set
                and qpid not in param_to_chunk
                and qpid not in seen_in_pending
            ):
                pending.append(qpid)
                seen_in_pending.add(qpid)
        # Include any block params that never appear in exec_order at all
        # (e.g. unused params); append at the end so they are still assigned
        # to a chunk and retain block-contiguity.
        for qpid in block_spans[block_id]:
            if qpid not in param_to_chunk and qpid not in seen_in_pending:
                pending.append(qpid)
                seen_in_pending.add(qpid)

        block_total = sum(param_sizes[q] for q in pending)
        cur_idx = len(chunks) - 1
        remaining = S_chunk - chunk_bytes[cur_idx]

        if chunk_bytes[cur_idx] > 0 and block_total > remaining:
            # The full block won't fit next to whatever is already in the
            # current chunk — seal and open a fresh chunk so the block begins
            # chunk-aligned. This is the block-contiguity rule.
            _seal_and_open()

        # Place the block's params contiguously. If ``block_total > S_chunk``
        # the block legitimately spans consecutive chunks; ``_place`` handles
        # the seal-on-overflow transparently, and because we only place block
        # params between here and the loop's next iteration no foreign param
        # can interleave mid-block.
        for qpid in pending:
            _place(qpid, param_sizes[qpid], block_id)

        # Advance ``i`` past this block's occurrences. We still only advance
        # by 1 — other block-mate slots will be skipped via ``param_to_chunk``
        # membership. Advancing by 1 keeps the logic simple and doesn't miss
        # intervening non-block params that appeared in exec_order *between*
        # this block's params (an unusual but legal model).
        i += 1

    # Any params present in the model but absent from exec_order fall through
    # to the end (the profiler may have missed them, or they're unused). They
    # still need a chunk assignment so ``param_to_chunk`` is total. Route them
    # through the same block-aware grouping as the main path: when a leftover
    # param belongs to a block, place every still-unplaced member of that
    # block contiguously (sealing the current chunk first if the whole group
    # won't fit) so ``block_to_chunks`` keeps the same block-contiguity
    # invariant the main loop establishes. True standalone leftovers
    # (``pid_owner.get(pid) is None``) fall back to plain greedy fit.
    for pid, size in param_sizes.items():
        if pid in param_to_chunk:
            continue
        fallback_bid: BlockId | None = pid_owner.get(pid)
        if fallback_bid is None:
            _place(pid, size, None)
            continue

        # Collect every still-unplaced member of this block, preserving the
        # caller's block_spans order so block-internal ordering is stable.
        pending = [
            qpid for qpid in block_spans[fallback_bid] if qpid not in param_to_chunk
        ]
        block_total = sum(param_sizes[qpid] for qpid in pending)
        cur_idx = len(chunks) - 1
        remaining = S_chunk - chunk_bytes[cur_idx]
        if chunk_bytes[cur_idx] > 0 and block_total > remaining:
            # Same seal-before-block rule as the main path: keep the block
            # chunk-aligned when it won't fit alongside the current contents.
            _seal_and_open()
        for qpid in pending:
            _place(qpid, param_sizes[qpid], fallback_bid)

    # Drop a trailing empty chunk that ``_seal_and_open`` may have left open
    # (e.g. the final placement started a fresh chunk for a block but only
    # filled a previous one).
    while len(chunks) > 1 and not chunks[-1]:
        chunks.pop()
        chunk_bytes.pop()

    frozen_chunks: tuple[tuple[ParamId, ...], ...] = tuple(tuple(c) for c in chunks)
    frozen_block_map: dict[BlockId, tuple[ChunkId, ...]] = {
        bid: tuple(cids) for bid, cids in block_to_chunks.items()
    }

    # Compute ``mandatory_persistent``: any chunk containing at least one
    # param NOT referenced by any block in ``block_spans``. Such params
    # live outside ``layout.block_to_chunks`` keys and are therefore
    # never gathered by the block-granularity scheduler — pinning these
    # chunks GPU-resident is the runtime-correctness invariant the paper
    # implicitly assumes (the paper's persistent-set is a prefix
    # ``[0, n_persist)``; the prefix happens to cover the embedding /
    # final-norm chunks in the canonical Llama layout, but our
    # exec-order placement can land them at any chunk index).
    #
    # Note: ``param_sizes`` keys span every named parameter in the model,
    # so the difference ``param_sizes - block_referenced`` enumerates
    # *every* non-block param (including those the profiler never
    # touched).
    nonblock_pids = set(param_sizes.keys()) - block_referenced
    mandatory: set[ChunkId] = set()
    for pid in nonblock_pids:
        cid = param_to_chunk.get(pid)
        if cid is not None:
            mandatory.add(cid)
    frozen_mandatory: frozenset[ChunkId] = frozenset(mandatory)

    LOG.debug(
        "build_layout: N_chunk=%d S_chunk=%d bytes, block_spans=%d "
        "mandatory_persistent=%s",
        len(frozen_chunks),
        S_chunk,
        len(block_spans),
        sorted(frozen_mandatory),
    )

    return ChunkLayout(
        S_chunk=S_chunk,
        N_chunk=len(frozen_chunks),
        chunks=frozen_chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=frozen_block_map,
        mandatory_persistent=frozen_mandatory,
    )


__all__ = ["build_layout"]

# Internal helpers exposed for the S_chunk grid search (sizing.py) so the
# simulation honors the same block-sealing/contiguity rules as build_layout.
# Not part of the public surface.
_pack_chunks_with_block_rules.__module__ = __name__
_build_packing_steps.__module__ = __name__
