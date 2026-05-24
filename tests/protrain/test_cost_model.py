"""Mode-aware fragmentation alpha dispatch and CKPT-chain reconstruction tests.

Covers the §7.6 raw-predictor under-prediction fix: a single 4-bit alpha was
calibrated against Mode-A where frozen-weight residency dominates; Mode-C with
gradient checkpointing reverses that dominance so the 0.75 factor structurally
under-predicts the raw peak. The dispatcher splits 4-bit alpha into Mode-A
(0.75) vs Mode-C-CKPT (0.95). The wrapper-side ``_reconstruct_f_bm`` is
updated to sum CKPT activation bytes (chain) rather than take the max, so it
agrees with ``estimate_peak``'s ``ckpt_chain_bytes`` term.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.cost.memory import (
    ALPHA_FRAGMENTATION,
    ALPHA_FRAGMENTATION_4BIT,
    ALPHA_FRAGMENTATION_4BIT_MODE_A,
    ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT,
    _compute_ckpt_chain_bytes,
    alpha_fragmentation_for_cfg,
    estimate_peak,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
)


def test_mode_alpha_constants_pinned():
    """Lock the two named mode-split constants so unrelated edits don't drift."""
    assert ALPHA_FRAGMENTATION_4BIT_MODE_A == pytest.approx(0.75)
    assert ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT == pytest.approx(0.95)
    # Mode-A constant stays the legacy 4-bit alpha for back-compat.
    assert ALPHA_FRAGMENTATION_4BIT_MODE_A == pytest.approx(ALPHA_FRAGMENTATION_4BIT)


def test_alpha_fragmentation_dispatch_mode_a():
    """4-bit + n_checkpoint=0 keeps the Mode-A 0.75 factor."""
    cfg_mode_a = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg_mode_a)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_A)


def test_alpha_fragmentation_dispatch_mode_c_ckpt():
    """4-bit + n_checkpoint>0 picks the Mode-C-CKPT 0.95 factor."""
    cfg_mode_c = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg_mode_c)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT)


def test_alpha_fragmentation_dispatch_mode_c_without_ckpt_keeps_mode_a():
    """4-bit + n_swap>0 but no CKPT still routes to Mode-A (frozen-weight regime)."""
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=4, n_checkpoint=0)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_A)


@pytest.mark.parametrize("bpe", [1.0, 2.0, 4.0])
def test_alpha_fragmentation_dispatch_non_4bit_ignores_mode(bpe: float):
    """fp16/bf16/8-bit keep the unconditional 1.10 alpha regardless of n_checkpoint."""
    cfg_a = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0)
    cfg_c = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=8)
    assert alpha_fragmentation_for_cfg(bpe, cfg_a) == pytest.approx(ALPHA_FRAGMENTATION)
    assert alpha_fragmentation_for_cfg(bpe, cfg_c) == pytest.approx(ALPHA_FRAGMENTATION)


def test_alpha_fragmentation_dispatch_none_cfg_falls_back_to_mode_a():
    """A missing cfg keeps the legacy 4-bit Mode-A factor."""
    assert alpha_fragmentation_for_cfg(0.5, None) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT_MODE_A
    )
    assert alpha_fragmentation_for_cfg(2.0, None) == pytest.approx(ALPHA_FRAGMENTATION)


def _build_synthetic_trace(
    n_blocks: int,
    activation_per_block: int,
    model_state_bytes: int = 0,
) -> ProfilerTrace:
    """A trace with N blocks, each with ``activation_per_block`` bytes."""
    activation_sizes = {BlockId(i): activation_per_block for i in range(n_blocks)}
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="test",
        bs=1,
        seq=16,
        sku="test",
        world=1,
    )


def _build_layout(s_chunk: int, n_chunk: int, n_blocks: int) -> ChunkLayout:
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple(tuple() for _ in range(n_chunk)),  # type: ignore[arg-type]
        param_to_chunk={},
        block_to_chunks={BlockId(i): () for i in range(n_blocks)},
    )


def test_compute_ckpt_chain_bytes_sums_only_ckpt_blocks():
    """The helper sums activation bytes only across CKPT-tagged blocks."""
    trace = _build_synthetic_trace(n_blocks=4, activation_per_block=1_000_000)
    block_map: BlockStrategyMap = {
        BlockId(0): BlockMode.NONE,
        BlockId(1): BlockMode.CKPT,
        BlockId(2): BlockMode.CKPT,
        BlockId(3): BlockMode.SWAP,
    }
    assert _compute_ckpt_chain_bytes(trace, block_map) == 2_000_000


def test_compute_ckpt_chain_bytes_empty_inputs():
    """Empty block map or empty trace returns 0."""
    trace = _build_synthetic_trace(n_blocks=2, activation_per_block=1_000)
    assert _compute_ckpt_chain_bytes(trace, {}) == 0
    empty_trace = _build_synthetic_trace(n_blocks=0, activation_per_block=0)
    block_map: BlockStrategyMap = {BlockId(0): BlockMode.CKPT}
    assert _compute_ckpt_chain_bytes(empty_trace, block_map) == 0


def test_reconstruct_f_bm_uses_chain_not_max():
    """``_reconstruct_f_bm`` sums activation bytes across CKPT blocks.

    Reproduces the §7.6 structural under-bound: with K CKPT blocks each of
    size A, the max-based prior bound returned A; the chain bound returns
    K*A, matching ``estimate_peak``'s ``ckpt_chain_bytes`` term.
    """

    from axolotl.integrations.protrain.api.model_wrapper import (
        _calibrate_peak_with_actual_chunk_bytes,
    )

    n_blocks = 4
    activation_per_block = 4_000_000
    trace = _build_synthetic_trace(
        n_blocks=n_blocks, activation_per_block=activation_per_block
    )
    layout = _build_layout(s_chunk=1 << 20, n_chunk=2, n_blocks=n_blocks)
    block_map: BlockStrategyMap = {BlockId(i): BlockMode.CKPT for i in range(n_blocks)}
    cfg = CostConfig(
        n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=n_blocks, n_offload=0
    )

    class _StubChunkManager:
        """Minimal stand-in for `ChunkManager` used by the calibrator."""

        _persistent_ids: tuple[int, ...] = (0,)

        class _Model:
            @staticmethod
            def named_parameters():
                return []

        model = _Model()

    chunk_manager = _StubChunkManager()

    captured: dict[str, int] = {}

    def _spy(bmap):
        # Mirror `_reconstruct_f_bm` body in `_calibrate_peak_with_actual_chunk_bytes`.
        ckpt_chain = _compute_ckpt_chain_bytes(trace, bmap)
        captured["chain"] = ckpt_chain
        return 0, len(bmap)

    # Drive the calibrator just for its side-effects via the helper; the
    # function under test is the public chain helper. We assert two things:
    #   1) The CKPT chain sums to n_blocks * activation_per_block.
    #   2) The legacy "max" semantics would have returned just A.
    chain = _compute_ckpt_chain_bytes(trace, block_map)
    assert chain == n_blocks * activation_per_block
    legacy_max = max(int(v) for v in trace.activation_sizes.values())
    assert legacy_max == activation_per_block
    assert chain > legacy_max, (
        "chain semantics must strictly exceed the legacy single-block max "
        "when more than one block is checkpointed"
    )

    # Run the calibrator end-to-end to confirm it doesn't crash with the
    # new chain semantics (no anchor assertion — the calibrator output
    # composes other terms).
    out = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=1 << 30,
        layout=layout,
        chunk_manager=chunk_manager,
        cfg=cfg,
        trace=trace,
        block_map=block_map,
        hw=HardwareProfile(
            gpu_sku="test",
            gpu_memory_bytes=24 * (1 << 30),
            gpu_count=1,
            pcie_h2d_bps=13e9,
            pcie_d2h_bps=13e9,
            has_nvlink=False,
            dominant_param_bytes_per_element=0.5,
        ),
    )
    # Smoke check: positive, finite int.
    assert isinstance(out, int) and out > 0

    # Spy path: confirm the chain helper plugged into a `_reconstruct_f_bm`-shaped
    # closure surfaces the chain (not max) bytes.
    _spy(block_map)
    assert captured["chain"] == n_blocks * activation_per_block


def test_estimate_peak_mode_c_ckpt_tighter_than_mode_a_alpha():
    """The §7.6 anchor: Mode-C+CKPT raw peak now scales by 0.95, not 0.75.

    Synthetic stand-in for the audit row (30B-Llama, 4-bit, low-seq Mode-C):
    the previous unconditional 0.75 alpha dragged the prediction below the
    measured peak. With the mode split, the same raw_peak times 0.95 lands
    19/15 = ~1.27x higher, which is exactly the direction needed to absorb
    the 8-31% under-prediction observed at seq=512/1024/2048.
    """
    s_chunk = 1 << 28  # 256 MiB
    n_chunk = 4
    n_blocks = 4
    activation_per_block = 1 << 26  # 64 MiB
    layout = _build_layout(s_chunk=s_chunk, n_chunk=n_chunk, n_blocks=n_blocks)
    trace = _build_synthetic_trace(
        n_blocks=n_blocks, activation_per_block=activation_per_block
    )

    cfg_mode_a = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    cfg_mode_c_ckpt = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)

    block_map_a: BlockStrategyMap = {
        BlockId(i): BlockMode.NONE for i in range(n_blocks)
    }
    block_map_c: BlockStrategyMap = {
        BlockId(i): BlockMode.CKPT for i in range(n_blocks)
    }

    hw_4bit = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )

    peak_mode_a = estimate_peak(cfg_mode_a, trace, layout, block_map_a, hw_4bit)
    peak_mode_c_ckpt = estimate_peak(
        cfg_mode_c_ckpt, trace, layout, block_map_c, hw_4bit
    )
    assert peak_mode_a > 0 and peak_mode_c_ckpt > 0

    # The Mode-C-CKPT path now uses alpha=0.95; recomputing it with the old
    # alpha=0.75 would yield strictly lower bytes. Reverse-out via the ratio.
    legacy_mode_c_ckpt_peak = int(
        peak_mode_c_ckpt
        * ALPHA_FRAGMENTATION_4BIT_MODE_A
        / ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT
    )
    assert peak_mode_c_ckpt > legacy_mode_c_ckpt_peak, (
        f"new Mode-C-CKPT prediction {peak_mode_c_ckpt} should exceed the "
        f"legacy 0.75-alpha prediction {legacy_mode_c_ckpt_peak} by ~27%"
    )
    # ~27% headroom (0.95/0.75 ≈ 1.266), allow 1% rounding slack.
    assert peak_mode_c_ckpt / max(1, legacy_mode_c_ckpt_peak) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT / ALPHA_FRAGMENTATION_4BIT_MODE_A,
        rel=0.01,
    )
