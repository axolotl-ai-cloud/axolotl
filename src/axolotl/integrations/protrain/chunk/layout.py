"""Param-to-chunk assignment with execution-order intra-chunk reordering."""

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

# PackingStep = ("single", size) or ("block", [sizes]); shared between build_layout + grid search.
SingleStep = tuple[str, int]
BlockStep = tuple[str, list[int]]
PackingStep = tuple[str, object]  # union of SingleStep / BlockStep at runtime


def _pack_chunks_with_block_rules(
    steps: Sequence[PackingStep],
    S_chunk: int,
) -> list[int]:
    """Return per-chunk byte usage following block-aware placement; mirrors build_layout's loop."""
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
                # Seal-before-block to keep block chunk-aligned.
                _seal_and_open()
            for size in payload:
                _place(size)
        else:
            raise ValueError(f"unknown packing step kind: {kind!r}")

    # Tail-trim trailing empty chunks.
    while len(chunk_bytes) > 1 and chunk_bytes[-1] == 0:
        chunk_bytes.pop()
    return chunk_bytes


def _validate_block_spans(
    block_spans: Mapping[BlockId, Sequence[ParamId]],
    param_sizes: Mapping[ParamId, int],
) -> dict[ParamId, BlockId]:
    """Validate block_spans (per-block uniqueness, no cross-block overlap, existence); return owner map."""
    block_referenced: set[ParamId] = set()
    pid_owner: dict[ParamId, BlockId] = {}
    overlaps: dict[ParamId, list[BlockId]] = {}
    for owner_bid, params in block_spans.items():
        seen: set[ParamId] = set()
        for pid in params:
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
    return pid_owner


def _build_packing_steps(
    param_sizes: Mapping[ParamId, int],
    exec_order: Sequence[ParamId],
    block_spans: Mapping[BlockId, Sequence[ParamId]],
) -> list[PackingStep]:
    """Translate (exec_order, block_spans) into packer step stream; mirrors build_layout's walk."""
    # exec_order validation matches build_layout for a single error contract.
    for pid in exec_order:
        if pid not in param_sizes:
            raise KeyError(
                f"exec_order references unknown param {pid!r}; "
                "not present in model.named_parameters()"
            )

    placed: set[ParamId] = set()
    pid_owner = _validate_block_spans(block_spans, param_sizes)
    pid_to_block: dict[ParamId, BlockId | None] = {
        pid: pid_owner.get(pid) for pid in exec_order
    }

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
    """Return {ParamId -> byte size} for every named parameter; works on meta/CPU/CUDA."""
    sizes: dict[ParamId, int] = {}
    for name, param in model.named_parameters():
        sizes[cast(ParamId, name)] = int(param.numel()) * int(param.element_size())
    return sizes


def _block_of(
    pid: ParamId, block_spans: Mapping[BlockId, Sequence[ParamId]]
) -> BlockId | None:
    """Find BlockId owning pid via linear scan (block_spans is small)."""
    for block_id, params in block_spans.items():
        if pid in params:
            return block_id
    return None


def build_layout(
    model: "nn.Module",
    exec_order: list[ParamId],
    S_chunk: int,
    block_spans: Mapping[BlockId, Sequence[ParamId]],
) -> ChunkLayout:
    """Assign params to chunks in exec order; block params contiguous, shared params at first occurrence."""
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

    pid_owner = _validate_block_spans(block_spans, param_sizes)

    chunks: list[list[ParamId]] = [[]]
    chunk_bytes: list[int] = [0]
    param_to_chunk: dict[ParamId, ChunkId] = {}
    block_to_chunks: dict[BlockId, list[ChunkId]] = {}

    def _seal_and_open() -> None:
        chunks.append([])
        chunk_bytes.append(0)

    def _place(pid: ParamId, size: int, block_id: BlockId | None) -> None:
        """Append pid to current chunk; oversized params get their own chunk."""
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

    # pid_owner already authoritative; .get() returns None for unaffiliated params.
    pid_to_block: dict[ParamId, BlockId | None] = {
        pid: pid_owner.get(pid) for pid in exec_order
    }

    # First-occurrence walk applies "pack whole block together" rule on first hit.
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

        # Gather block params in remaining exec-order, skipping already-placed.
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
        # Append unused block params (absent from exec_order) at the tail.
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
        for qpid in pending:
            _place(qpid, param_sizes[qpid], block_id)

        # Advance i by 1; block-mates skip via param_to_chunk membership.
        i += 1

    # Tail: params absent from exec_order. Same block-aware grouping.
    for pid, size in param_sizes.items():
        if pid in param_to_chunk:
            continue
        fallback_bid: BlockId | None = pid_owner.get(pid)
        if fallback_bid is None:
            _place(pid, size, None)
            continue

        # Preserve block_spans order for block-internal stability.
        pending = [
            qpid for qpid in block_spans[fallback_bid] if qpid not in param_to_chunk
        ]
        block_total = sum(param_sizes[qpid] for qpid in pending)
        cur_idx = len(chunks) - 1
        remaining = S_chunk - chunk_bytes[cur_idx]
        if chunk_bytes[cur_idx] > 0 and block_total > remaining:
            _seal_and_open()
        for qpid in pending:
            _place(qpid, param_sizes[qpid], fallback_bid)

    # Drop trailing empty chunk if _seal_and_open left one open.
    while len(chunks) > 1 and not chunks[-1]:
        chunks.pop()
        chunk_bytes.pop()

    frozen_chunks: tuple[tuple[ParamId, ...], ...] = tuple(tuple(c) for c in chunks)
    frozen_block_map: dict[BlockId, tuple[ChunkId, ...]] = {
        bid: tuple(cids) for bid, cids in block_to_chunks.items()
    }

    # mandatory_persistent = chunks containing non-block params; runtime must pin.
    nonblock_pids = set(param_sizes.keys()) - set(pid_owner.keys())
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

# Internal helpers re-exported for the S_chunk grid search in sizing.py.
_pack_chunks_with_block_rules.__module__ = __name__
_build_packing_steps.__module__ = __name__
