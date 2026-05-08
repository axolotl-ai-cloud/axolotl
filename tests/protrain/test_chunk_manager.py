"""Tests for the ProTrain hierarchical chunk manager (M2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ParamId,
)

if TYPE_CHECKING:
    from axolotl.integrations.protrain.chunk import ChunkManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_gpt2():
    """Return a freshly-initialized 2-block GPT-2 LM (CPU weights).

    Kept small so the tests run in seconds with or without a GPU.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        n_positions=16,
    )
    return GPT2LMHeadModel(cfg)


def _make_block_spans(model) -> dict[BlockId, list[ParamId]]:
    """Extract ``block_id -> [param ids]`` from ``transformer.h.{i}`` submodules."""
    spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        # GPT-2: transformer.h.<i>.<rest>
        try:
            h_idx = parts.index("h")
            block_idx = int(parts[h_idx + 1])
        except (ValueError, IndexError):
            continue
        spans.setdefault(cast(BlockId, block_idx), []).append(cast(ParamId, name))
    return spans


# ---------------------------------------------------------------------------
# layout.py / sizing.py — CPU-only, torch-light tests
# ---------------------------------------------------------------------------


def test_layout_respects_block_grouping():
    """All params of a transformer block land in a single chunk when they fit."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from axolotl.integrations.protrain.chunk.layout import build_layout

    model = _tiny_gpt2()
    block_spans = _make_block_spans(model)
    assert len(block_spans) == 2, "expected n_layer=2"

    # Force a generous S_chunk so the whole model fits in one chunk easily;
    # the block-contiguity rule should still hold trivially. Then also
    # test with a tighter S_chunk sized so each block fits but the full
    # model does not — the stronger assertion.
    all_params = [cast(ParamId, n) for n, _ in model.named_parameters()]
    exec_order = list(all_params)  # pretend exec order = definition order

    # Total model bytes.
    total_bytes = sum(p.numel() * p.element_size() for _, p in model.named_parameters())

    # Pick an S_chunk large enough for each block (and every single param)
    # but smaller than the whole model so we actually get multiple chunks.
    # For the tiny GPT-2 here each block is ~200 KB and total is ~437 KB,
    # so S_chunk just above max(block_bytes) guarantees the block fits in
    # one chunk while forcing at least two chunks overall.
    block_bytes_each = []
    named = dict(model.named_parameters())
    for pids in block_spans.values():
        block_bytes = 0
        for pid in pids:
            param = named[pid]
            block_bytes += param.numel() * param.element_size()
        block_bytes_each.append(block_bytes)
    max_param_bytes = max(p.numel() * p.element_size() for p in named.values())
    # Ensure S_chunk fits the largest single param and any single block, with
    # a modest safety margin, yet is strictly less than ``total_bytes``.
    S_chunk = max(max(block_bytes_each), max_param_bytes) + 1024

    # Safety: S_chunk should be < total so we actually get multiple chunks.
    assert S_chunk < total_bytes, (
        f"test setup: S_chunk={S_chunk} must be < total_bytes={total_bytes} "
        "to exercise multi-chunk layout"
    )

    layout = build_layout(model, exec_order, S_chunk, block_spans)

    # Every block's params must live in exactly one chunk (they fit).
    for block_id, pids in block_spans.items():
        chunk_ids = {layout.param_to_chunk[pid] for pid in pids}
        assert len(chunk_ids) == 1, (
            f"block {block_id} spans chunks {chunk_ids}; "
            f"expected single chunk since block_bytes={block_bytes_each[block_id]} "
            f"fits in S_chunk={S_chunk}"
        )
        assert layout.block_to_chunks[block_id] == tuple(chunk_ids)


def test_layout_preserves_first_occurrence_for_shared_params():
    """A weight referenced twice in exec_order is placed once, at the first slot."""
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.layout import build_layout

    class SharedWeight(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4, bias=False)
            self.b = nn.Linear(4, 4, bias=False)
            # Share: b uses a's weight.
            self.b.weight = self.a.weight
            self.head = nn.Linear(4, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.b(self.a(x)))

    model = SharedWeight()

    # The shared tensor registers under its first dotted path. Collect
    # unique param ids in the canonical named_parameters order.
    param_names = [cast(ParamId, n) for n, _ in model.named_parameters()]
    # Should be: ["a.weight", "head.weight"] — b.weight is a ref to a.weight
    # and named_parameters de-duplicates by identity.
    assert "a.weight" in param_names
    # Construct an exec_order that visits a.weight TWICE (once for self.a,
    # once as b.weight via sharing) to exercise the dedup rule.
    exec_order: list[ParamId] = [
        cast(ParamId, "a.weight"),
        cast(ParamId, "a.weight"),  # shared reference — first-occurrence wins
        cast(ParamId, "head.weight"),
    ]

    S_chunk = 1 << 20  # plenty big
    layout = build_layout(model, exec_order, S_chunk, block_spans={})

    # ``a.weight`` should appear exactly once across all chunks.
    flat = [pid for chunk in layout.chunks for pid in chunk]
    assert flat.count(cast(ParamId, "a.weight")) == 1
    # And it should be in the first chunk (where its first occurrence lives).
    assert cast(ParamId, "a.weight") in layout.chunks[0]


def test_param_exec_order_follows_trace_op_stream_not_declaration_order():
    """Exec order is derived from ``trace.op_order`` (§3.1.1), not param declaration.

    Build a 2-block model that *registers* its blocks in one order
    (``b`` then ``a``) but *executes* them in the opposite order
    (``a`` then ``b``) on the forward pass. The trace-driven helper
    must emit ``a``'s param before ``b``'s, so the gather pattern lines
    up with the actual op stream rather than the storage order.
    """
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _param_exec_order,
    )
    from axolotl.integrations.protrain.types import OpId, OpRecord

    class FlippedOrder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Registration order: b first, then a — opposite to forward order.
            self.b = nn.Linear(4, 4, bias=False)
            self.a = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Execution order: a first, then b.
            return self.b(self.a(x))

    model = FlippedOrder()

    # Sanity: declaration order really is (b, a).
    declared = [n for n, _ in model.named_parameters()]
    assert declared == ["b.weight", "a.weight"], (
        f"test setup invariant broken: declared order is {declared}; "
        "expected ['b.weight', 'a.weight'] so a trace-driven order can "
        "differ from declaration order"
    )

    # Synthesize a minimal trace whose op_order reflects forward order.
    # build_layout doesn't care about non-module-path fields, but we
    # still construct a valid OpRecord for each step.
    def _op(op_id: int, mod_path: str) -> OpRecord:
        return OpRecord(
            op_id=cast(OpId, op_id),
            module_path=mod_path,
            qualified_name="aten::linear",
            shape_signature=((1, 4),),
            block_id=None,
            is_forward=True,
        )

    class FakeTrace:
        op_order = (_op(0, "a"), _op(1, "b"))

    # _param_exec_order ignores block_spans (block grouping happens in
    # build_layout); pass an empty mapping to avoid invoking
    # discover_blocks on this non-transformer toy model.
    exec_order = _param_exec_order(model, {}, FakeTrace())

    assert exec_order == [
        cast(ParamId, "a.weight"),
        cast(ParamId, "b.weight"),
    ], (
        f"trace-driven exec order should be (a, b) — the forward order — "
        f"got {exec_order}"
    )

    # And the layout chunks must reflect the same order.
    from axolotl.integrations.protrain.chunk.layout import build_layout

    layout = build_layout(model, exec_order, S_chunk=1 << 20, block_spans={})
    flat = [pid for chunk in layout.chunks for pid in chunk]
    a_idx = flat.index(cast(ParamId, "a.weight"))
    b_idx = flat.index(cast(ParamId, "b.weight"))
    assert a_idx < b_idx, (
        f"layout still walks declaration order: a@{a_idx} b@{b_idx}; "
        "expected a before b to match forward op stream"
    )


def test_param_exec_order_dedups_weight_tied_params():
    """A tied weight visited twice in the trace keeps only the first slot."""
    pytest.importorskip("torch")

    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _param_exec_order,
    )
    from axolotl.integrations.protrain.types import OpId, OpRecord

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight  # tie

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    model = Tied()

    def _op(op_id: int, mod_path: str) -> OpRecord:
        return OpRecord(
            op_id=cast(OpId, op_id),
            module_path=mod_path,
            qualified_name="aten::linear",
            shape_signature=((1, 4),),
            block_id=None,
            is_forward=True,
        )

    class FakeTrace:
        # second uses the SAME tensor as first; the second op should not
        # introduce a duplicate slot.
        op_order = (_op(0, "first"), _op(1, "second"))

    exec_order = _param_exec_order(model, {}, FakeTrace())

    # named_parameters dedups by tensor identity, exposing the tied
    # weight under its first registered name (``first.weight``).
    assert exec_order.count(cast(ParamId, "first.weight")) == 1
    assert cast(ParamId, "second.weight") not in exec_order


def test_sizing_picks_min_waste():
    """Grid-search chooses the minimum-waste candidate, tie-breaking to the smaller S.

    The algorithm (Appendix B.1) keeps every positive candidate in the
    search — ``_simulate_waste`` already models the "oversize tensor on
    its own chunk" path correctly via the ``max(0, S_chunk - bytes_used)``
    clamp, so smaller candidates that can't hold the largest tensor
    natively remain legal members of the grid. Among the candidates it
    simulates greedy-fit chunking and picks the S_chunk minimizing the
    sum of ``S_chunk - bytes_used`` across every *non-tail* chunk. Ties
    are broken by picking the *smaller* candidate — the slot-pool
    capacity ceiling is ``M_buffer = n_buffer * S_chunk`` (paper Eq. 11),
    so a larger S inflates the resident buffer footprint without
    reducing waste. Picking the smaller S at equal waste keeps the
    buffer ceiling tight while preserving waste minimisation.
    """
    from axolotl.integrations.protrain.chunk.sizing import pick_S_chunk

    MB = 1 << 20

    # Case A — undersized candidate dominates via the oversize-clamp path.
    # 8 × 63 MB params on the {32, 64, 128, 256} MB grid:
    #   S=32 → each 63 MB param spills into its own oversize chunk (8
    #          chunks); every non-tail chunk's ``max(0, 32-63) = 0`` so
    #          waste = 0.
    #   S=64 → each 63 MB param sits alone (8 chunks); 7 non-tail × 1 MB
    #          = 7 MB.
    #   S=128 → pairs fit (4 chunks); 3 non-tail × 2 MB = 6 MB.
    #   S=256 → quads fit (2 chunks); 1 non-tail × 4 MB = 4 MB.
    # S=32 strictly minimizes the simulator's waste metric (the clamp
    # treats every oversize chunk as zero-slack), so it wins outright —
    # no tie-break engages.
    sizes_a: dict[ParamId, int] = {cast(ParamId, f"p{i}"): 63 * MB for i in range(8)}
    picked_a = pick_S_chunk(sizes_a)
    assert picked_a == 32 * MB, (
        f"oversize-clamp scenario: expected S=32 MB (simulator waste=0 via "
        f"the max(0, ...) oversize-chunk clamp, strictly less than the "
        f"4/6/7 MB waste at S=64/128/256); got {picked_a}"
    )

    # Case B — exact-fit regime with an all-tied waste profile. 4 × 64 MB
    # params on the full grid:
    #   S=32 → each 64 MB param spills into its own oversize chunk; clamp
    #          → waste 0.
    #   S=64 → fills each chunk exactly; preceding waste 0.
    #   S=128 → fits pairs exactly; preceding waste 0.
    #   S=256 → fits all four in one tail-only chunk; waste 0.
    # All four candidates tie at 0 waste, so the tie-break rule
    # ("prefer smaller S_chunk" — keeps the n_buffer * S_chunk ceiling
    # tight) selects the smallest grid entry, 32 MB.
    sizes_b: dict[ParamId, int] = {cast(ParamId, f"q{i}"): 64 * MB for i in range(4)}
    picked_b = pick_S_chunk(sizes_b)
    assert picked_b == 32 * MB, (
        f"tie-at-zero-waste scenario: expected S=32 MB via tie-break "
        f"(prefer smaller S to minimise buffer-pool ceiling); got {picked_b}"
    )

    # Case C — undersized candidates win the zero-waste tie. 3 × 100 MB
    # params on the full grid:
    #   S=32 → each 100 MB param overflows; oversize-chunk clamp → waste 0.
    #   S=64 → each 100 MB param overflows; clamp → waste 0.
    #   S=128 → each param sits alone (3 chunks), 2 non-tail × 28 MB =
    #           56 MB.
    #   S=256 → greedy packs [200][100] (2 chunks), 1 non-tail × 56 MB =
    #           56 MB.
    # Minimum waste is 0 at S=32 and S=64; tie-break picks the smaller
    # (S=32) to minimise buffer-pool ceiling.
    sizes_c: dict[ParamId, int] = {cast(ParamId, f"r{i}"): 100 * MB for i in range(3)}
    picked_c = pick_S_chunk(sizes_c)
    assert picked_c == 32 * MB, (
        f"oversize-clamp tie scenario: expected S=32 MB (tie-break among "
        f"the zero-waste oversize candidates {{32, 64}} MB picks the "
        f"smaller); got {picked_c}"
    )

    # Sanity — every pick is drawn from the documented grid.
    for picked in (picked_a, picked_b, picked_c):
        assert picked in (32 * MB, 64 * MB, 128 * MB, 256 * MB)


def test_sizing_simulation_matches_build_layout_with_blocks():
    """``pick_S_chunk``'s simulation honors block-sealing exactly like ``build_layout``.

    Closes the paper-fidelity gap (App B.1): the prior heuristic ignored
    block-contiguity, so for a mixed (block + non-block) model the picked
    S_chunk could diverge from the fragmentation-optimum the *real* layout
    produces. With the shared packer the simulation is bit-for-bit equal.

    Synthetic model:
      * 1 non-block prefix param of 10 MB
      * 2 transformer blocks, each 3 params of 30 MB (block_total = 90 MB)
      * 1 non-block tail param of 5 MB

    For S_chunk = 100 MB:
      * naive greedy fit → [10+30+30+30=100][30+30+30=90][30+30+30+5=95] etc.
      * block-aware fit → seal-before-block at the prefix:
            chunk 0: prefix=10  (sealed before block 0 because 10+90>100)
            chunk 1: block0=90  (90 fits alone, but next block won't fit)
            chunk 2: block1=90 + tail=5
        i.e. 3 chunks instead of 2; non-tail waste = 90 + 10 = 100 MB.
    The legacy heuristic would have under-counted this cost.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.sizing import (
        DEFAULT_GRID,
        _simulate_waste,
        pick_S_chunk,
    )

    MB = 1 << 20

    # Build a tiny synthetic model whose param sizes are easy to reason about.
    # We use float32 (4B/elem) and pick numel so each tensor is exactly the
    # target byte count.
    def _param(num_bytes: int) -> nn.Parameter:
        n_elems = num_bytes // 4
        return nn.Parameter(torch.empty(n_elems, dtype=torch.float32))

    class Synth(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = _param(10 * MB)
            # Two "blocks" of 3 params each, 30 MB apiece (90 MB block total).
            for b in range(2):
                for p in range(3):
                    self.register_parameter(f"b{b}_p{p}", _param(30 * MB))
            self.tail = _param(5 * MB)

    model = Synth()
    param_bytes: dict[ParamId, int] = {
        cast(ParamId, name): int(p.numel()) * int(p.element_size())
        for name, p in model.named_parameters()
    }
    block_spans: dict[BlockId, list[ParamId]] = {
        cast(BlockId, b): [cast(ParamId, f"b{b}_p{p}") for p in range(3)]
        for b in range(2)
    }
    # Use named_parameters() iteration order as the synthetic exec order.
    exec_order = list(param_bytes.keys())

    # 1. For every grid candidate, _simulate_waste's chunk count must match
    #    the number of chunks build_layout actually produces.
    for S_chunk in DEFAULT_GRID:
        if S_chunk < max(param_bytes.values()):
            # Skip candidates the feasibility filter would drop — build_layout
            # accepts them via the oversize-tensor path but the simulation is
            # only meaningful for feasible candidates.
            continue
        layout = build_layout(model, exec_order, S_chunk, block_spans)
        # Drive the simulation directly to compare chunk counts.
        from axolotl.integrations.protrain.chunk.layout import (
            _build_packing_steps,
            _pack_chunks_with_block_rules,
        )

        steps = _build_packing_steps(param_bytes, exec_order, block_spans)
        sim_bytes = _pack_chunks_with_block_rules(steps, S_chunk)
        assert len(sim_bytes) == layout.N_chunk, (
            f"S_chunk={S_chunk}: simulation chunk count {len(sim_bytes)} "
            f"!= build_layout's {layout.N_chunk}"
        )
        # Per-chunk byte counts must also match the actual layout.
        actual_bytes_per_chunk = [
            sum(param_bytes[pid] for pid in chunk) for chunk in layout.chunks
        ]
        assert sim_bytes == actual_bytes_per_chunk, (
            f"S_chunk={S_chunk}: simulation per-chunk bytes {sim_bytes} "
            f"!= actual {actual_bytes_per_chunk}"
        )

    # 2. _simulate_waste with block info must give a different (higher) waste
    #    for S_chunk=64 MB than the naive greedy version: under block-aware
    #    rules, the 10 MB prefix forces a seal-before-block when block 0 (90
    #    MB total) won't fit alongside it, so the prefix's chunk wastes
    #    54 MB. The naive heuristic would have packed prefix+first-block
    #    params together and avoided that seal.
    S = 64 * MB
    waste_with_blocks = _simulate_waste(param_bytes, exec_order, block_spans, S)
    waste_no_blocks = _simulate_waste(param_bytes, exec_order, {}, S)
    assert waste_with_blocks > waste_no_blocks, (
        f"block-aware simulation should report MORE waste than naive greedy "
        f"at S_chunk=64 MB (prefix forces an extra seal); got "
        f"with={waste_with_blocks} vs without={waste_no_blocks}"
    )

    # 3. The grid pick from the block-aware simulation matches the S_chunk
    #    that build_layout actually produces fewest non-tail-waste bytes for.
    picked = pick_S_chunk(param_bytes, exec_order=exec_order, block_spans=block_spans)
    # Compute the ground truth: minimum non-tail waste across feasible S.
    feasible = tuple(S for S in DEFAULT_GRID if S >= max(param_bytes.values()))
    truth_waste: dict[int, int] = {}
    for S in feasible:
        layout = build_layout(model, exec_order, S, block_spans)
        per_chunk = [sum(param_bytes[pid] for pid in chunk) for chunk in layout.chunks]
        truth_waste[S] = (
            sum(max(0, S - b) for b in per_chunk[:-1]) if len(per_chunk) > 1 else 0
        )
    min_waste = min(truth_waste.values())
    valid_picks = {S for S, w in truth_waste.items() if w == min_waste}
    # Tie-break: largest among the minima.
    expected = max(valid_picks)
    assert picked == expected, (
        f"pick_S_chunk picked {picked} but build_layout's actual non-tail "
        f"waste is minimized at {expected} (truth: {truth_waste})"
    )


def test_sizing_oversize_tensor_does_not_credit_undersized_candidate():
    """Oversize-tensor chunks must not give a too-small candidate negative waste.

    When ``S_chunk < max_param_bytes`` and a single-tensor chunk overflows,
    ``_simulate_waste`` would naively report ``S_chunk - bytes < 0`` for that
    chunk. The clamp ``max(0, ...)`` prevents that from crediting the small
    candidate. With all positive candidates kept in the search (no soft
    feasibility-filter fallback), the simulator is the sole arbiter: under
    the clamp every candidate produces an ``[oversize, small-tail]``
    chunking and the small tail is the trailing chunk (excluded from waste
    accounting), so all four candidates tie at zero waste. The tie-break
    rule (prefer the smaller ``S`` at equal waste — keeps the
    ``n_buffer * S_chunk`` capacity ceiling tight) then picks
    ``min(DEFAULT_GRID)``.
    """
    from axolotl.integrations.protrain.chunk.sizing import (
        DEFAULT_GRID,
        _simulate_waste,
        pick_S_chunk,
    )

    MB = 1 << 20
    # One huge param (300 MB > 256 MB max grid) plus several small ones.
    param_bytes: dict[ParamId, int] = {
        cast(ParamId, "huge"): 300 * MB,
        cast(ParamId, "s0"): 10 * MB,
        cast(ParamId, "s1"): 10 * MB,
        cast(ParamId, "s2"): 10 * MB,
    }
    # All candidates produce [300 MB oversize, 30 MB tail]; non-tail waste
    # is clamp(S - 300) = 0 for every S, so the tie-break selects
    # min(DEFAULT_GRID) (= 32 MB).
    picked = pick_S_chunk(param_bytes)
    assert picked == min(DEFAULT_GRID), (
        f"with every candidate tied at zero waste under the oversize clamp, "
        f"the tie-break must pick min(DEFAULT_GRID); got {picked}"
    )

    # Sanity: _simulate_waste reports non-negative waste even when the chunk
    # holding "huge" overflows S_chunk.
    waste = _simulate_waste(param_bytes, list(param_bytes.keys()), {}, 32 * MB)
    assert waste >= 0, (
        f"oversize-tensor chunk must not produce negative waste; got {waste}"
    )


# ---------------------------------------------------------------------------
# pinned_alloc.py — GPU-only
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pinned_alloc_precise_size():
    """cudaHostAlloc path allocates exactly n_buffer * S_chunk bytes."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    n_buffer = 4
    S_chunk = 1 << 20  # 1 MB
    mem = PinnedHostMemory(n_buffer=n_buffer, S_chunk=S_chunk)
    try:
        if not mem.is_precise_size:
            pytest.skip(
                "PinnedHostMemory fell back to torch.empty(pin_memory=True); "
                "precise-size assertion not applicable on this path"
            )
        # Slot 0 and slot (n-1) should both be valid and exactly S_chunk bytes.
        for i in (0, n_buffer - 1):
            t = mem.buffer(i)
            try:
                assert t.numel() == S_chunk
                assert t.dtype == torch.uint8
            finally:
                # Release the borrow so close() doesn't raise the
                # use-after-free guard.
                del t
                mem.release_buffer(i)
        # Total bytes exactly n_buffer * S_chunk (no pow-2 round-up).
        assert mem.total_bytes == n_buffer * S_chunk
        assert mem.total_bytes == 4 << 20  # 4 MB, NOT 8 MB
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# buffer_pool.py — GPU-only
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_buffer_pool_acquire_release():
    """LRU-free semantics: after release, next acquire returns the same physical buffer."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import ChunkId

    n_buffer = 4
    S_chunk = 1 << 20
    host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=S_chunk)
    try:
        pool = BufferPool(
            n_buffer=n_buffer,
            S_chunk=S_chunk,
            pinned_host=host,
            device=torch.device("cuda"),
        )

        # Acquire 3 of 4 — each for a distinct chunk id.
        buf0 = pool.acquire(cast(ChunkId, 0))
        buf1 = pool.acquire(cast(ChunkId, 1))
        buf2 = pool.acquire(cast(ChunkId, 2))
        assert pool.num_in_use == 3
        assert pool.num_free == 1

        # Release one, then acquire for a NEW chunk id (not resident).
        pool.release(cast(ChunkId, 1))
        assert pool.num_free == 2

        # The freshly released buffer's tag is still 1, so lookup_resident works.
        assert pool.lookup_resident(cast(ChunkId, 1)) is buf1

        # Acquire a new chunk id — evicts the LRU free slot. That was slot 3
        # (never-used) first in our FIFO; after releasing chunk 1 its slot
        # went to the tail. So the first free-list pop is slot 3, then slot 1.
        buf3 = pool.acquire(cast(ChunkId, 99))
        # Re-acquire chunk 1 — it's still resident, should return the SAME buffer.
        buf1_again = pool.acquire(cast(ChunkId, 1))
        assert buf1_again.data_ptr() == buf1.data_ptr()
        # And the buffer's physical slot should match.
        assert pool.lookup_resident(cast(ChunkId, 1)) is buf1_again

        # Keep silencing unused-var warnings — verify distinctness.
        assert buf0.data_ptr() != buf2.data_ptr()
        assert buf3.data_ptr() not in {
            buf0.data_ptr(),
            buf1.data_ptr(),
            buf2.data_ptr(),
        }
    finally:
        host.close()


# ---------------------------------------------------------------------------
# Full loss parity — deferred until the scheduler (M4) wires this up
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_loss_parity_n_persist_extremes():
    """Loss values must match between pure-GPU and pure-offload modes.

    End-to-end correctness check that ProTrain's chunk-offload paths do
    not perturb training math. Run 5 steps of a tiny GPT-2 twice with
    identical seeds and batches:

    * Config A: ``n_persist = N_chunk`` (every chunk stays on GPU; no
      offload, no prefetch).
    * Config B: ``n_persist = 0`` (pure offload; every chunk H2D/D2H-
      transits the PCIe bus each iteration).

    The per-step loss trajectories must match to fp16-noise tolerance
    (``|loss_a[i] - loss_b[i]| < 5e-2``) — optimizer math is the same in
    both cases; only the physical residency of params differs.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    device = torch.device("cuda")
    gpt2_cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=16
    )

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(device),
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )

    bs, seq = 1, 8
    # Shared batches — generated once so both configs see the same data.
    torch.manual_seed(123)
    batches = [
        {
            "input_ids": torch.randint(
                0, gpt2_cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
            ),
        }
        for _ in range(5)
    ]
    for b in batches:
        b["labels"] = b["input_ids"].clone()

    def _run_config(n_persist_mode: str) -> list[float]:
        """Run 5 steps and return per-step losses."""
        import gc

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Deterministic init.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)

        if n_persist_mode == "all":
            # force_all_persistent synthesizes n_persist=N_chunk, which is
            # the "pure GPU" config we want here. It also enables CKPT on
            # every block — we don't want that for the math-parity test
            # because CKPT's recompute can swing fp32 activations by a ulp
            # and we need <5e-2 tolerance. Use explicit override instead.
            probe = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                force_all_persistent=True,
            )
            n_chunk = cast("ChunkManager", probe.chunk_manager).layout.N_chunk
            # Tear down and rebuild without CKPT.
            for h in cast("list[Any]", probe._hook_handles):
                try:
                    h.remove()
                except Exception:
                    pass
            del probe
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            model = GPT2LMHeadModel(gpt2_cfg).to(device)
            wrapped = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                n_persist_override=n_chunk,
                n_buffer_override=max(1, n_chunk),
                n_swap_override=0,
                n_checkpoint_override=0,
            )
        elif n_persist_mode == "none":
            # Full offload — need N_chunk. Probe first.
            probe = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                force_all_persistent=True,
            )
            n_chunk = cast("ChunkManager", probe.chunk_manager).layout.N_chunk
            for h in cast("list[Any]", probe._hook_handles):
                try:
                    h.remove()
                except Exception:
                    pass
            del probe
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            model = GPT2LMHeadModel(gpt2_cfg).to(device)
            # n_persist=0, still no CKPT so the math matches A exactly.
            wrapped = protrain_model_wrapper(
                model,
                model_config=gpt2_cfg,
                hardware_profile=hw,
                batch_size=bs,
                seq_len=seq,
                capacity_bytes=2 * (1 << 30),
                n_persist_override=0,
                n_buffer_override=max(2, n_chunk),
                n_swap_override=0,
                n_checkpoint_override=0,
            )
        else:
            raise AssertionError(f"unknown mode {n_persist_mode!r}")

        optim = protrain_optimizer_wrapper(wrapped, lr=1e-4)

        losses: list[float] = []
        for batch in batches:
            out = wrapped.module(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(float(out.loss.detach()))

        # Teardown.
        for h in cast("list[Any]", wrapped._hook_handles):
            try:
                h.remove()
            except Exception:
                pass
        del wrapped, model, optim
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        return losses

    losses_all = _run_config("all")
    losses_none = _run_config("none")

    print(f"\nloss trajectory (n_persist=N_chunk):  {losses_all}")
    print(f"loss trajectory (n_persist=0):        {losses_none}")

    assert len(losses_all) == len(losses_none) == 5
    for i, (a, b) in enumerate(zip(losses_all, losses_none, strict=True)):
        assert abs(a - b) < 5e-2, (
            f"loss divergence at step {i}: n_persist=N_chunk->{a:.6f} "
            f"vs n_persist=0->{b:.6f} (|Δ|={abs(a - b):.6f})"
        )


# ---------------------------------------------------------------------------
# Item 5 follow-up: throughput-fix coverage
#
# These two tests exercise the fast paths added by Fix B and Fix C
# without requiring an actual distributed process group: they call the
# manager's helpers directly with a monkeypatched ``torch.distributed``
# entry point. Distributed-correctness coverage (real 2-rank gloo) lives
# in ``tests/protrain/test_chunk_manager_distributed.py``.
# ---------------------------------------------------------------------------


def _build_one_chunk_persistent_manager_fp32(
    *,
    bias: bool = True,
):
    """Return a single-chunk persistent ChunkManager whose chunk has 2 fp32 params.

    Used by the Fix C unit test. CPU-only, no distributed init.
    Mirrors the helper in :mod:`tests.protrain.test_chunk_manager_distributed`
    but kept local to this test module so the fast suite has zero
    cross-file imports.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    torch.manual_seed(0)
    layer = nn.Linear(4, 4, bias=bias)
    model = nn.Module()
    model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        block_spans.setdefault(cast(BlockId, 0), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 14
    layout = build_layout(model, exec_order, S_chunk, block_spans)
    assert layout.N_chunk == 1, (
        f"setup expects single-chunk layout, got {layout.N_chunk}"
    )

    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=1,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cpu"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=1,  # one persistent chunk == every chunk persistent
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cpu"),
    )
    return model, mgr, host, pool


def test_persistent_grad_reduce_coalesces_same_dtype_grads(monkeypatch):
    """Fix C: persistent-chunk grad reduction issues ONE all_reduce per dtype.

    The legacy implementation looped through every param in the chunk
    and called ``dist.all_reduce(param.grad, op=AVG)`` once per param.
    Fix C replaces that with a coalesced flatten → single all_reduce →
    unflatten (same primitive PyTorch DDP uses). For a chunk holding
    two fp32 params, the coalesced path issues exactly one collective.

    The test monkeypatches ``torch.distributed.all_reduce`` so it
    counts calls without requiring an initialized process group, then
    invokes the manager's coalesce helper directly. This covers the
    no-DDP code path that runs in real 4-GPU Mode-C / Mode-A-no-DDP
    benches.
    """
    pytest.importorskip("torch")
    import torch

    model, mgr, host, _pool = _build_one_chunk_persistent_manager_fp32()

    try:
        # Plant uniform grads on every param. We don't care about the
        # values — the count of dist.all_reduce calls is what's under
        # test. Use distinct values per param so the unflatten step's
        # writeback can be verified end-to-end.
        for i, (_n, p) in enumerate(model.named_parameters()):
            p.grad = torch.full_like(p.data, float(i + 1))

        original_grads = {
            n: p.grad.detach().clone() for n, p in model.named_parameters()
        }

        calls: list[dict] = []

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            calls.append(
                {
                    "numel": int(tensor.numel()),
                    "dtype": tensor.dtype,
                    "op": op,
                }
            )
            # Identity reduction: leave tensor as-is so the post-reduce
            # value matches the input. AVG semantics across world_size=1
            # are identity anyway, so this is faithful.
            return None

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        mgr._coalesced_all_reduce_persistent_grads(cast("ChunkId", 0))

        # Critical assertion: the chunk's two same-dtype grads were
        # coalesced into one collective, not two.
        assert len(calls) == 1, (
            f"expected exactly 1 coalesced all_reduce, got {len(calls)} "
            f"(per-param path resurfaced — Fix C regression)"
        )
        # The coalesced buffer should match the dtype of the param
        # grads and span all of them.
        total_grad_numel = sum(int(p.grad.numel()) for _, p in model.named_parameters())
        # _flatten_dense_tensors may pack with no padding; numel covers
        # every element.
        assert calls[0]["numel"] == total_grad_numel, (
            f"coalesced all_reduce numel ({calls[0]['numel']}) does not "
            f"cover the chunk's grad numel ({total_grad_numel}) — flatten "
            f"missed a tensor"
        )
        assert calls[0]["dtype"] == torch.float32

        # Each param's grad must come back with the original values
        # (identity reduction); confirms the unflatten + copy_back step
        # writes the right slices into the right grads.
        for n, p in model.named_parameters():
            assert torch.equal(p.grad, original_grads[n]), (
                f"unflatten/copy_back perturbed grad for '{n}' under identity reduction"
            )
    finally:
        mgr.uninstall()
        host.close()


def test_persistent_grad_reduce_one_collective_per_dtype_group(monkeypatch):
    """Fix C: mixed-dtype chunks issue ONE all_reduce per dtype group.

    Constructs a 2-param chunk with one fp32 grad and one fp16 grad.
    The coalesce helper groups by dtype and issues one all_reduce per
    group — so we expect exactly 2 collectives (one fp32, one fp16),
    not 2 = one per param coincidentally. The single-grad-per-dtype
    path is also covered: it skips the flatten/unflatten round-trip
    and reduces in-place. Both flavours are routed through the same
    helper; counting is sufficient to lock the structure in.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    torch.manual_seed(0)

    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # fp32 weight — 16 elems
            self.proj = nn.Linear(4, 4, bias=False)
            # fp16 layernorm weight — 4 elems
            self.norm = nn.LayerNorm(4).to(torch.float16)

    layer = _Mixed()
    model = nn.Module()
    model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        block_spans.setdefault(cast(BlockId, 0), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 14
    layout = build_layout(model, exec_order, S_chunk, block_spans)
    assert layout.N_chunk == 1

    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    try:
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=1,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
        )

        try:
            for _n, p in model.named_parameters():
                p.grad = torch.full_like(p.data, 1.0)

            calls: list[torch.dtype] = []

            def fake_all_reduce(tensor, op=None, group=None, async_op=False):
                calls.append(tensor.dtype)
                return None

            monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

            mgr._coalesced_all_reduce_persistent_grads(cast("ChunkId", 0))

            # Two dtype groups → exactly two collectives. Order is
            # dtype-dictionary-iteration order, which Python 3.7+
            # guarantees as insertion order — so fp32 grads (proj.weight)
            # come first, fp16 grads (norm.weight + norm.bias) second.
            dtypes_seen = set(calls)
            assert dtypes_seen == {torch.float32, torch.float16}, (
                f"expected one collective per dtype group "
                f"({{fp32, fp16}}), saw {dtypes_seen}"
            )
            # Per-dtype call count: exactly one per group, regardless of
            # how many params belong to the group.
            from collections import Counter

            per_dtype = Counter(calls)
            assert per_dtype[torch.float32] == 1, (
                f"fp32 group should issue 1 collective, issued "
                f"{per_dtype[torch.float32]}"
            )
            assert per_dtype[torch.float16] == 1, (
                f"fp16 group should issue 1 collective, issued "
                f"{per_dtype[torch.float16]}"
            )
        finally:
            mgr.uninstall()
    finally:
        host.close()


def test_gather_skips_collective_on_pool_resident_hit(monkeypatch):
    """Fix B: gather() short-circuits when ``lookup_resident`` hits.

    The buffer pool's tag survives ``release`` between forward and
    backward, so a chunk that wasn't evicted in the meantime can be
    re-claimed without re-issuing the per-region
    ``all_gather_into_tensor`` collective. This test plants a sharded
    chunk state by hand, simulates the "resident in pool" condition by
    pre-acquiring the buffer with the chunk's id, then calls
    ``gather()`` and asserts ``_gather_sharded`` is NOT invoked.

    No real ``torch.distributed`` group is needed — the cache-hit path
    must short-circuit BEFORE touching any collective.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import (
        ChunkManager,
        _ChunkShardState,
    )
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import ChunkId

    torch.manual_seed(0)
    layer = nn.Linear(4, 4, bias=True)
    model = nn.Module()
    model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        block_spans.setdefault(cast(BlockId, 0), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 14
    layout = build_layout(model, exec_order, S_chunk, block_spans)
    assert layout.N_chunk == 1

    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    try:
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )
        # n_persist=0: the chunk is non-persistent so gather() runs the
        # full path. We do NOT enable zero3_shard at construction
        # (which requires world_size > 1) — instead we will plant a
        # shard state by hand so the sharded fast-path branch is
        # exercised below.
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
        )

        try:
            mgr.materialize_offload()

            # Plant a synthetic shard state so gather() takes the
            # sharded branch when it goes through cache-miss. We never
            # actually exercise the cache-miss path here; the planted
            # state's only role is to demonstrate the fast path bails
            # before touching the sharded collective.
            chunk_id = cast(ChunkId, 0)
            mgr._chunk_shards[chunk_id] = _ChunkShardState(
                regions=[],  # empty regions list — _gather_sharded would
                # iterate it and do nothing; that's fine, the test
                # below sentinels _gather_sharded BEFORE any iteration.
                chunk_bytes=int(layout.S_chunk),
                shard_bytes=int(layout.S_chunk),
            )

            # Pre-acquire the buffer with chunk_id 0 so the pool tags
            # the slot as resident. Then release it so the pool's free
            # list contains it — but the tag survives, exactly as it
            # does at the post_block_forward / pre_block_backward
            # boundary in real training.
            pool.acquire(chunk_id)
            pool.release(chunk_id)
            assert pool.lookup_resident(chunk_id) is not None, (
                "test setup: pool.release dropped the resident tag — "
                "fix B's invariant cannot hold"
            )

            # Sentinel _gather_sharded: if the cache-hit path fires it
            # MUST NOT be called. We replace it with a recorder that
            # raises on invocation so we get a clean traceback if the
            # short-circuit regresses.
            sharded_calls = {"n": 0}
            orig_gather_sharded = mgr._gather_sharded

            def _recording_gather_sharded(*args, **kwargs):
                sharded_calls["n"] += 1
                return orig_gather_sharded(*args, **kwargs)

            monkeypatch.setattr(mgr, "_gather_sharded", _recording_gather_sharded)

            mgr.gather(chunk_id)

            assert sharded_calls["n"] == 0, (
                f"Fix B regression: pool-resident chunk still ran "
                f"_gather_sharded (and therefore all_gather_into_tensor) "
                f"{sharded_calls['n']} time(s) on the cache-hit path"
            )
        finally:
            mgr.uninstall()
    finally:
        host.close()


def test_gather_is_lease_idempotent_within_active_window():
    """Repeated ``gather(cid)`` while the chunk is active must not bump leases.

    The scheduler's ``pre_block_forward`` calls ``gather`` 2-3 times per
    chunk per active window — once via ``ensure_block_resident`` and
    once or twice more via the lookahead-prefetch for the next block
    (which is the same chunk under the block-contiguity rule when two
    adjacent blocks share a chunk). The buffer pool's lease counter
    increments on every ``acquire`` / ``acquire_if_resident`` call, but
    only ONE matching ``offload(cid)`` fires per active window (in the
    last owning block's ``post_block_forward``). If ``gather`` were not
    lease-idempotent, the lease counter would grow without bound and
    the pool would exhaust around block ~6 in a 20-block, n_buffer=2
    OFFLOAD layout — see ``BLOCK_MODE_OFFLOAD_DESIGN.md`` §3.7 / §5.1.

    Invariant under test: after N back-to-back ``gather(cid)`` calls
    followed by ONE ``offload(cid)``, the slot returns to the free
    list (lease drops to zero). Tested directly against the buffer
    pool's ``num_in_use`` counter and the ``_active_chunks`` tracker.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import ChunkId

    torch.manual_seed(0)
    layer = nn.Linear(4, 4, bias=True)
    model = nn.Module()
    model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        block_spans.setdefault(cast(BlockId, 0), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 14
    layout = build_layout(model, exec_order, S_chunk, block_spans)
    assert layout.N_chunk == 1

    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    try:
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )
        # n_persist=0: the chunk is non-persistent so gather() takes the
        # buffer-pool path (the persistent early-return would mask the
        # lease-bookkeeping under test).
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
        )

        try:
            mgr.materialize_offload()
            chunk_id = cast(ChunkId, 0)

            # Five back-to-back gathers — emulates the scheduler's
            # repeated lookahead/ensure-resident calls that exercised
            # the bug in production.
            for _ in range(5):
                mgr.gather(chunk_id)

            # The chunk is "active" in the manager's tracker exactly
            # once, and the pool reports exactly one slot in use.
            assert chunk_id in mgr._active_chunks
            assert pool.num_in_use == 1, (
                f"lease-idempotency regression: 5 gather() calls "
                f"created {pool.num_in_use} buffer leases (expected 1)"
            )

            # One offload should fully release the slot — proving the
            # scheduler's "one offload per active window" contract is
            # honored by the manager.
            mgr.offload(chunk_id)
            assert chunk_id not in mgr._active_chunks
            assert pool.num_in_use == 0, (
                f"single offload() didn't release the slot — "
                f"num_in_use={pool.num_in_use}"
            )

            # And the next gather/offload cycle still works (no stale
            # state from the previous cycle).
            mgr.gather(chunk_id)
            assert pool.num_in_use == 1
            mgr.offload(chunk_id)
            assert pool.num_in_use == 0
        finally:
            mgr.uninstall()
    finally:
        host.close()
