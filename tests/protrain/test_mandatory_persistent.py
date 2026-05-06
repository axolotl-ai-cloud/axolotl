"""Tests for ``ChunkLayout.mandatory_persistent`` (paper line 188 fidelity).

The ProTrain paper allocates persistent chunks sequentially from the
beginning of the model — the persistent set is a prefix ``[0, n_persist)``.
Our integration carries an additional invariant: chunks containing at
least one *non-block* param (``model.norm.weight``, an untied ``lm_head``,
embeddings, etc.) MUST stay GPU-resident regardless of the searcher's
prefix pick, because the block-granularity scheduler cannot gather them
on its own. Earlier versions of the model wrapper conflated these two
sets (``cfg.n_persist`` was collapsed into ``len(_persistent_ids)`` after
the in-place pin); this dual-meaning made the search, telemetry, and
cost model disagree on what ``n_persist`` describes.

These tests pin down the post-fix invariants:

1. ``ChunkLayout.effective_persistent_ids`` returns prefix ∪ mandatory.
2. ``ChunkManager`` honours ``layout.mandatory_persistent`` natively
   (no in-place mutation needed; ``mark_persistent`` returns the
   augmented set for any prefix).
3. ``cfg.n_persist`` is preserved through wrapper construction.
4. ``block_map_runtime_admissible`` admits configs at ``n_persist=0``
   when all non-persistent blocks are mode CKPT/SWAP/OFFLOAD, even
   if mandatory pins exist.
5. ``min_n_buffer_for`` excludes mandatory chunks from buffer slots.
6. ``model_state_present_bytes`` charges for the augmented set.

No torch runtime — pure-data tests against the type / search /
cost-model surface.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.cost.memory import (
    estimate_cpu_footprint,
    model_state_present_bytes,
)
from axolotl.integrations.protrain.search.exhaustive import (
    block_map_runtime_admissible,
    min_n_buffer_for,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerTrace,
)


def _layout(
    *,
    n_chunk: int = 4,
    s_chunk: int = 1024,
    mandatory: frozenset[ChunkId] | None = None,
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]] | None = None,
) -> ChunkLayout:
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    param_to_chunk = {ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)}
    if block_to_chunks is None:
        block_to_chunks = {
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2),),
            # Chunk 3 is non-block — typical "tail" chunk holding
            # ``model.norm.weight`` or an untied ``lm_head``.
        }
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
        mandatory_persistent=mandatory if mandatory is not None else frozenset(),
    )


def _trace(model_state_bytes: int = 0) -> ProfilerTrace:
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes={},
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=1.6e10,
        pcie_d2h_bps=1.6e10,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="mandatory-persistent-test",
        bs=1,
        seq=1,
        sku="cpu",
        world=1,
    )


def _hw(*, gpu_count: int = 1, zero3_shard: bool = False) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=gpu_count,
        pcie_h2d_bps=1.6e10,
        pcie_d2h_bps=1.6e10,
        has_nvlink=False,
        zero3_shard=zero3_shard,
    )


# ---------------------------------------------------------------------------
# 1. ChunkLayout.effective_persistent_ids
# ---------------------------------------------------------------------------


def test_effective_persistent_ids_unions_prefix_and_mandatory() -> None:
    """``layout.effective_persistent_ids(k)`` == ``{0..k-1} ∪ mandatory``."""
    layout = _layout(mandatory=frozenset({ChunkId(3)}))

    # n_persist=0 -> just the mandatory pin
    assert layout.effective_persistent_ids(0) == frozenset({ChunkId(3)})
    # n_persist=2 -> prefix [0,2) plus mandatory
    assert layout.effective_persistent_ids(2) == frozenset(
        {ChunkId(0), ChunkId(1), ChunkId(3)}
    )
    # n_persist == N_chunk -> whole layout
    assert layout.effective_persistent_ids(4) == frozenset(
        {ChunkId(0), ChunkId(1), ChunkId(2), ChunkId(3)}
    )


def test_effective_persistent_ids_default_is_prefix_only() -> None:
    """An empty mandatory set degrades to the paper's prefix-only contract."""
    layout = _layout(mandatory=frozenset())
    assert layout.mandatory_persistent == frozenset()
    assert layout.effective_persistent_ids(2) == frozenset({ChunkId(0), ChunkId(1)})
    # Round-trip ChunkLayout(... ) without ``mandatory_persistent`` keyword.
    bare = ChunkLayout(
        S_chunk=64,
        N_chunk=2,
        chunks=((ParamId("p.0"),), (ParamId("p.1"),)),
        param_to_chunk={ParamId("p.0"): ChunkId(0), ParamId("p.1"): ChunkId(1)},
        block_to_chunks={BlockId(0): (ChunkId(0),)},
    )
    assert bare.mandatory_persistent == frozenset()


# ---------------------------------------------------------------------------
# 2. ChunkManager honours layout.mandatory_persistent natively
# ---------------------------------------------------------------------------


def test_chunk_manager_residency_includes_mandatory_at_n_persist_zero() -> None:
    """ChunkManager built at n_persist=0 still pins the mandatory set.

    This is the load-bearing post-fix invariant: earlier versions
    constructed the manager at the search prefix and then mutated
    ``_persistent_ids`` in-place from the model wrapper. Now the layout
    carries ``mandatory_persistent`` natively and ``mark_persistent``
    augments the prefix on its own.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk import ChunkManager

    # Build a model with 4 params under dotted attribute names matching
    # the layout's ParamIds. ``nn.ParameterDict`` exposes its entries
    # under ``<container>.<key>`` so the keys land as ``params.p_i`` —
    # we adjust the layout to use those keys to keep the test simple
    # without monkey-patching ``named_parameters``.
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.ParameterDict(
                {f"p_{i}": nn.Parameter(torch.zeros(1)) for i in range(4)}
            )

    model = _Tiny()
    chunks = tuple((ParamId(f"params.p_{i}"),) for i in range(4))
    param_to_chunk = {ParamId(f"params.p_{i}"): ChunkId(i) for i in range(4)}
    layout = ChunkLayout(
        S_chunk=1024,
        N_chunk=4,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks={
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2),),
        },
        mandatory_persistent=frozenset({ChunkId(3)}),
    )

    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=0,
        buffer_pool=None,
        device=torch.device("cpu"),
    )

    assert mgr._persistent_ids == {ChunkId(3)}, (
        "ChunkManager built at n_persist=0 must pin the mandatory set; "
        f"got {sorted(mgr._persistent_ids)}"
    )
    assert mgr._non_persistent_ids == {ChunkId(0), ChunkId(1), ChunkId(2)}

    # And ``mark_persistent`` re-applies the mandatory pin on a re-call.
    mgr.mark_persistent(0)
    assert mgr._persistent_ids == {ChunkId(3)}
    mgr.mark_persistent(2)
    # Prefix [0, 2) ∪ {3} == {0, 1, 3}
    assert mgr._persistent_ids == {ChunkId(0), ChunkId(1), ChunkId(3)}
    assert mgr._non_persistent_ids == {ChunkId(2)}


# ---------------------------------------------------------------------------
# 3. cfg.n_persist preserved (no count-collapse)
# ---------------------------------------------------------------------------


def test_cost_config_n_persist_is_prefix_not_count() -> None:
    """``CostConfig.n_persist`` describes the *prefix length*, not a residency
    count. Mandatory pins must NOT inflate ``cfg.n_persist`` post-construction.

    The post-fix invariant: a ``CostConfig`` produced by the search /
    calibration carries the prefix verbatim. Cost model and runtime
    resolve the augmented set from ``layout.effective_persistent_ids``.
    """
    layout = _layout(mandatory=frozenset({ChunkId(3)}))
    cfg = CostConfig(n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=0, n_offload=0)
    # Augmented set is {0, 3}; prefix length is still 1.
    assert cfg.n_persist == 1
    assert layout.effective_persistent_ids(cfg.n_persist) == frozenset(
        {ChunkId(0), ChunkId(3)}
    )
    assert len(layout.effective_persistent_ids(cfg.n_persist)) == 2


# ---------------------------------------------------------------------------
# 4. Search admits n_persist=0 with mandatory pins present
# ---------------------------------------------------------------------------


def test_admissibility_admits_n_persist_zero_with_mandatory_pin() -> None:
    """``block_map_runtime_admissible`` must accept a NONE-mode block whose
    chunks are all mandatory-persistent, even at ``n_persist=0``.

    Without honouring ``mandatory_persistent``, this would have rejected
    the config (chunk 3 looks non-persistent under prefix-only checks).
    """
    layout = _layout(
        mandatory=frozenset({ChunkId(3)}),
        block_to_chunks={
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2),),
            BlockId(2): (ChunkId(3),),  # mandatory-only block
        },
    )
    bm: BlockStrategyMap = {
        BlockId(0): BlockMode.CKPT,  # non-persistent chunks -> CKPT
        BlockId(1): BlockMode.CKPT,
        BlockId(2): BlockMode.NONE,  # all-mandatory -> NONE is admissible
    }
    assert block_map_runtime_admissible(layout, bm, n_persist=0), (
        "NONE on a block whose chunks are all mandatory-persistent must "
        "be admissible regardless of the search prefix."
    )


def test_admissibility_rejects_none_on_truly_nonpersistent_block() -> None:
    """Sanity: NONE on a block with a non-mandatory non-persistent chunk
    is still rejected at ``n_persist=0``.
    """
    layout = _layout(mandatory=frozenset({ChunkId(3)}))
    bm: BlockStrategyMap = {
        BlockId(0): BlockMode.NONE,  # owns chunk 1 (non-mandatory, non-prefix)
        BlockId(1): BlockMode.CKPT,
    }
    assert not block_map_runtime_admissible(layout, bm, n_persist=0)


# ---------------------------------------------------------------------------
# 5. min_n_buffer_for excludes mandatory chunks from buffer slots
# ---------------------------------------------------------------------------


def test_min_n_buffer_for_excludes_mandatory() -> None:
    """``min_n_buffer_for`` must treat mandatory chunks as resident, not
    routed through the buffer pool. Two adjacent blocks A=(0,1), B=(2,3)
    with ``mandatory={3}`` and prefix=0 leaves only chunks {0,1,2}
    needing pool slots — the union for adjacent block pairs is
    ``{0,1} ∪ {2}`` -> 3 slots, NOT 4.
    """
    layout = _layout(
        mandatory=frozenset({ChunkId(3)}),
        block_to_chunks={
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2), ChunkId(3)),
        },
    )
    # Prefix only would compute |{0,1} ∪ {2,3}| = 4. With mandatory={3}
    # excluded, the answer drops to |{0,1} ∪ {2}| = 3.
    assert min_n_buffer_for(layout, n_persist=0) == 3


# ---------------------------------------------------------------------------
# 6. model_state_present_bytes charges for the augmented set
# ---------------------------------------------------------------------------


def test_model_state_present_bytes_charges_augmented_set() -> None:
    """At prefix=0 with one mandatory pin, the resident model-state cost
    is one chunk-worth (not zero).
    """
    layout = _layout(mandatory=frozenset({ChunkId(3)}))
    cfg = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0, n_offload=0)
    trace = _trace(model_state_bytes=layout.N_chunk * layout.S_chunk)
    bytes_present = model_state_present_bytes(cfg, layout, trace)
    # n_persist_eff = len({3}) = 1; persistent_factor = 1.0 (LoRA-shape
    # trace where model_state_bytes == fp16 total).
    assert bytes_present == layout.S_chunk


def test_estimate_cpu_footprint_excludes_mandatory_from_offload() -> None:
    """CPU offload footprint must skip mandatory chunks even at prefix=0.
    Without the fix, all N_chunk chunks would be charged to CPU."""
    layout = _layout(mandatory=frozenset({ChunkId(3)}))
    cfg = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0, n_offload=0)
    hw = _hw()
    footprint = estimate_cpu_footprint(cfg, layout, hw)
    # 3 non-persistent chunks (0,1,2); chunk 3 is mandatory-pinned.
    assert footprint == 3 * layout.S_chunk
