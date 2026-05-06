"""M1 tests for the BlockMode.OFFLOAD rollout (Option B).

Two pure-data tests covering the M1 exit criteria from
``BLOCK_MODE_OFFLOAD_DESIGN.md`` §7:

1. ``test_admissibility_under_offload_rule`` — validates the
   ``block_map_runtime_admissible`` rule (§3.5 + the 2026-05-05 SWAP ×
   non-persistent lift, §6.6): OFFLOAD, CKPT, and SWAP blocks are
   always admissible (each provides its own activation persistence path
   that survives chunk-pool slot reuse). NONE blocks remain admissible
   only when every chunk they own is in the persistent set, because
   NONE installs no hooks and autograd's saved-tensors hold direct
   GPU storage refs that the chunk pool's slot reuse will clobber.

2. ``test_assign_modes_with_offload`` — validates the placement rule
   in the updated ``assign_modes`` (§3.6): OFFLOAD fills the
   "unopt-late" tail before NONE, after SWAP-early and CKPT-interleave.

No torch / nn imports — these are pure-data tests against the type +
search-validator + layout-rule plumbing.
"""

from __future__ import annotations

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.search.exhaustive import (
    block_map_runtime_admissible,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    ParamId,
)

# ---------------------------------------------------------------------------
# Fixtures (small layout: 2 blocks, 4 chunks, n_persist=1)
# ---------------------------------------------------------------------------


def _make_small_layout() -> ChunkLayout:
    """2 blocks, 4 chunks. Block 0 owns chunks {0, 1}; block 1 owns {2, 3}.

    With n_persist=1, only chunk 0 is persistent → block 0 has a
    non-persistent chunk (chunk 1) and block 1 has all-non-persistent
    chunks (2, 3). This exercises both "block has any non-persistent
    chunk" code paths in ``block_map_runtime_admissible``.
    """
    chunks = (
        (ParamId("p.0"),),
        (ParamId("p.1"),),
        (ParamId("p.2"),),
        (ParamId("p.3"),),
    )
    param_to_chunk = {
        ParamId("p.0"): ChunkId(0),
        ParamId("p.1"): ChunkId(1),
        ParamId("p.2"): ChunkId(2),
        ParamId("p.3"): ChunkId(3),
    }
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]] = {
        BlockId(0): (ChunkId(0), ChunkId(1)),
        BlockId(1): (ChunkId(2), ChunkId(3)),
    }
    return ChunkLayout(
        S_chunk=64,
        N_chunk=4,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
    )


def _make_all_persistent_layout() -> ChunkLayout:
    """2 blocks, 4 chunks. Same shape as ``_make_small_layout`` but used
    with ``n_persist=4`` so every chunk is persistent.
    """
    return _make_small_layout()


# ---------------------------------------------------------------------------
# Test 1: admissibility under the new OFFLOAD rule
# ---------------------------------------------------------------------------


def test_admissibility_under_offload_rule() -> None:
    """``block_map_runtime_admissible`` must accept OFFLOAD on
    non-persistent blocks while still rejecting NONE/SWAP on the
    same. CKPT and all-persistent NONE remain admissible (unchanged).
    """
    layout = _make_small_layout()
    n_persist = 1  # only chunk 0 is persistent

    # Sanity: block 1 has only non-persistent chunks (2, 3). Block 0
    # mixes persistent (0) + non-persistent (1) — also "any
    # non-persistent" for the rule.

    # --- Case A: OFFLOAD on the non-persistent block -> admissible.
    bm_offload: BlockStrategyMap = {
        BlockId(0): BlockMode.NONE,  # block 0 has chunk 0 only persistent;
        # but chunk 1 is non-persistent, so NONE here would normally fail.
        # Re-tag block 0 to CKPT so this case isolates block 1's OFFLOAD.
        BlockId(1): BlockMode.OFFLOAD,
    }
    bm_offload[BlockId(0)] = BlockMode.CKPT
    assert block_map_runtime_admissible(layout, bm_offload, n_persist) is True, (
        "OFFLOAD on a non-persistent block must be admissible (Option B §3.5)."
    )

    # --- Case B: NONE on the non-persistent block -> inadmissible (unchanged).
    bm_none: BlockStrategyMap = {
        BlockId(0): BlockMode.CKPT,  # safe regardless
        BlockId(1): BlockMode.NONE,  # block 1's chunks are non-persistent
    }
    assert block_map_runtime_admissible(layout, bm_none, n_persist) is False, (
        "NONE on a block with non-persistent chunks must remain inadmissible."
    )

    # --- Case C: CKPT on the non-persistent block -> admissible (unchanged).
    bm_ckpt: BlockStrategyMap = {
        BlockId(0): BlockMode.CKPT,
        BlockId(1): BlockMode.CKPT,
    }
    assert block_map_runtime_admissible(layout, bm_ckpt, n_persist) is True, (
        "CKPT must remain admissible regardless of chunk persistence."
    )

    # --- Case D: NONE on an all-persistent layout -> admissible (unchanged).
    bm_all_none: BlockStrategyMap = {
        BlockId(0): BlockMode.NONE,
        BlockId(1): BlockMode.NONE,
    }
    n_persist_full = 4  # every chunk persistent
    assert (
        block_map_runtime_admissible(
            _make_all_persistent_layout(), bm_all_none, n_persist_full
        )
        is True
    ), "NONE on a fully-persistent layout must remain admissible."

    # --- Case E: SWAP on a non-persistent block -> ADMISSIBLE (post the
    # 2026-05-05 §6.6 lift). The SwappedBlock pack/unpack pair persists
    # every saved tensor to a pinned-CPU pool slot whose lifetime is
    # independent of param.data and of the chunk buffer's GPU bytes;
    # the scheduler's prefetch_stream.wait_stream(swap_stream) barrier
    # closes the only theoretical D2H/H2D race on the chunk slot bytes.
    # This unlocks the paper's joint optimisation of n_persist /
    # n_swap / n_checkpoint without the previous chunk-residency cross-
    # restriction.
    bm_swap: BlockStrategyMap = {
        BlockId(0): BlockMode.CKPT,
        BlockId(1): BlockMode.SWAP,
    }
    assert block_map_runtime_admissible(layout, bm_swap, n_persist) is True, (
        "SWAP on a block with non-persistent chunks must be admissible "
        "(post the §6.6 SWAP × non-persistent lift; saved tensors are "
        "persisted to a pinned-CPU pool decoupled from param.data)."
    )

    # --- Case F: SWAP on a fully-persistent layout still admissible
    # (sanity check that the lift didn't regress the legacy path).
    bm_swap_persist: BlockStrategyMap = {
        BlockId(0): BlockMode.SWAP,
        BlockId(1): BlockMode.SWAP,
    }
    assert (
        block_map_runtime_admissible(
            _make_all_persistent_layout(), bm_swap_persist, n_persist_full
        )
        is True
    ), "SWAP on a fully-persistent layout must remain admissible."

    # --- Case G: mixed CKPT+SWAP+OFFLOAD on non-persistent blocks all
    # admissible. Demonstrates the post-lift composition: every non-NONE
    # mode can land on a non-persistent block, and the searcher can mix
    # them freely (paper §3.3 jointly-optimised knobs).
    bm_mixed: BlockStrategyMap = {
        BlockId(0): BlockMode.SWAP,  # block 0 has non-persistent chunk 1
        BlockId(1): BlockMode.OFFLOAD,
    }
    assert block_map_runtime_admissible(layout, bm_mixed, n_persist) is True, (
        "Mixed SWAP/OFFLOAD on non-persistent blocks must be admissible — the "
        "lift means the searcher can compose all four modes (NONE for "
        "fully-persistent blocks, plus CKPT/OFFLOAD/SWAP anywhere) without "
        "chunk-residency cross-restrictions."
    )


# ---------------------------------------------------------------------------
# Test 2: assign_modes placement with the new n_offload knob
# ---------------------------------------------------------------------------


def _count_modes(modes: BlockStrategyMap) -> dict[BlockMode, int]:
    counts: dict[BlockMode, int] = {
        BlockMode.NONE: 0,
        BlockMode.CKPT: 0,
        BlockMode.SWAP: 0,
        BlockMode.OFFLOAD: 0,
    }
    for m in modes.values():
        counts[m] = counts[m] + 1
    return counts


def test_assign_modes_with_offload() -> None:
    """``assign_modes`` honours ``n_offload`` under the new placement
    rule from §3.6: SWAP earliest, CKPT interleaved, OFFLOAD in the
    unopt-late tail before NONE, NONE filling the rest.

    Backward-compat: ``n_offload=0`` (default) reproduces the legacy
    SWAP/CKPT/NONE behaviour exactly — verified in the third sub-case.
    """
    # --- Case 1: 2 OFFLOAD + 2 NONE (no SWAP, no CKPT).
    modes_1 = assign_modes(n_swap=0, n_checkpoint=0, N_block=4, n_offload=2)
    counts_1 = _count_modes(modes_1)
    assert counts_1[BlockMode.OFFLOAD] == 2
    assert counts_1[BlockMode.NONE] == 2
    assert counts_1[BlockMode.SWAP] == 0
    assert counts_1[BlockMode.CKPT] == 0
    # OFFLOAD fills earliest free positions (before NONE in the tail).
    assert modes_1[BlockId(0)] is BlockMode.OFFLOAD
    assert modes_1[BlockId(1)] is BlockMode.OFFLOAD
    assert modes_1[BlockId(2)] is BlockMode.NONE
    assert modes_1[BlockId(3)] is BlockMode.NONE

    # --- Case 2: all 4 OFFLOAD (saturating n_offload).
    modes_2 = assign_modes(n_swap=0, n_checkpoint=0, N_block=4, n_offload=4)
    counts_2 = _count_modes(modes_2)
    assert counts_2[BlockMode.OFFLOAD] == 4
    assert counts_2[BlockMode.NONE] == 0
    assert counts_2[BlockMode.SWAP] == 0
    assert counts_2[BlockMode.CKPT] == 0
    for i in range(4):
        assert modes_2[BlockId(i)] is BlockMode.OFFLOAD

    # --- Case 3: n_offload=0 default → all 4 NONE (legacy backward-compat).
    modes_3 = assign_modes(n_swap=0, n_checkpoint=0, N_block=4, n_offload=0)
    counts_3 = _count_modes(modes_3)
    assert counts_3[BlockMode.NONE] == 4
    assert counts_3[BlockMode.OFFLOAD] == 0
    assert counts_3[BlockMode.SWAP] == 0
    assert counts_3[BlockMode.CKPT] == 0
    # And the implicit-default (no n_offload kwarg) must agree.
    modes_3b = assign_modes(n_swap=0, n_checkpoint=0, N_block=4)
    assert modes_3 == modes_3b, (
        "n_offload defaults to 0; omitting it must match the explicit-0 call."
    )

    # --- Case 4: mixed config 1 SWAP, 2 CKPT, 1 OFFLOAD, 0 NONE (N_block=4).
    modes_4 = assign_modes(n_swap=1, n_checkpoint=2, N_block=4, n_offload=1)
    counts_4 = _count_modes(modes_4)
    assert counts_4[BlockMode.SWAP] == 1
    assert counts_4[BlockMode.CKPT] == 2
    assert counts_4[BlockMode.OFFLOAD] == 1
    assert counts_4[BlockMode.NONE] == 0
    # Position checks under the placement rule:
    #   SWAP at block 0 (swap-early)
    #   CKPT at blocks 1, 3 (round-3 R3-I — centered placement: n_swap=1,
    #     remaining=3, n_checkpoint=2 → indices 1+(1*3)//4=1, 1+(3*3)//4=3)
    #   OFFLOAD on the only remaining slot: block 2.
    assert modes_4[BlockId(0)] is BlockMode.SWAP
    assert modes_4[BlockId(1)] is BlockMode.CKPT
    assert modes_4[BlockId(2)] is BlockMode.OFFLOAD
    assert modes_4[BlockId(3)] is BlockMode.CKPT
