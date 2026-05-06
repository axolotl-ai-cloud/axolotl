"""Symmetry guards for ``_calibrate_peak_with_actual_chunk_bytes``.

The post-search calibration must reverse out the SAME model-state
charge that :func:`cost.memory.model_state_present_bytes` added (full
optim state under full FT, fp16-only under LoRA-with-frozen-base) and
re-add a calibrated version using actual per-chunk bytes scaled by
the same per-factor breakdown. Pre-fix the calibration only reversed
out / re-added params at 1.0×, leaving the per-chunk full-state
multiplier hiding inside the residual ``f_bm`` and systematically
under-stating peak under full FT.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from axolotl.integrations.protrain.api.model_wrapper import (
    _calibrate_peak_with_actual_chunk_bytes,
)
from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    CostConfig,
    ParamId,
    ProfilerTrace,
)


def _layout(*, n_chunk: int = 4, s_chunk: int = 1024) -> ChunkLayout:
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk={ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)},
        block_to_chunks={
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2),),
            # Chunk 3 left non-block — typical tail.
        },
    )


def _stub_chunk_manager(
    layout: ChunkLayout,
    persistent_ids: set[int],
    chunk_param_bytes: dict[int, int],
) -> SimpleNamespace:
    """Minimal stub matching what ``_calibrate_peak_with_actual_chunk_bytes`` reads.

    ``_chunk_bytes(layout, cm)`` walks ``cm.model.named_parameters()`` and
    sums ``numel * element_size`` per chunk. ParamIds are dotted (e.g.
    ``"p.0"``) and ``nn.Module.register_parameter`` rejects dotted
    names, so we stub ``named_parameters`` directly with the (name,
    Parameter) tuples we need.
    """
    params: list[tuple[str, nn.Parameter]] = []
    for cid, pids in enumerate(layout.chunks):
        target_bytes = chunk_param_bytes.get(cid, 0)
        for pid in pids:
            # fp32 = 4 bytes/element; ceil-div lands at-least the target
            # so the test math stays predictable.
            numel = max(1, (target_bytes + 3) // 4)
            param = nn.Parameter(torch.zeros(numel, dtype=torch.float32))
            params.append((str(pid), param))

    model = SimpleNamespace(named_parameters=lambda: iter(params))
    return SimpleNamespace(
        model=model,
        _persistent_ids={ChunkId(cid) for cid in persistent_ids},
    )


def _trace(model_state_bytes: int) -> ProfilerTrace:
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
        arch_hash="peak-calibration-test",
        bs=1,
        seq=1,
        sku="cpu",
        world=1,
    )


def test_calibrate_peak_scales_persistent_by_full_state_factor() -> None:
    """Calibrated persistent contribution scales with ``persistent_factor``.

    Symmetry with :func:`model_state_present_bytes`: when
    ``model_state_bytes`` (the trace's full-state aggregate) doubles,
    the calibration's persistent term doubles too. Pre-fix the
    calibration always charged 1.0× and the symmetry was broken.
    """
    layout = _layout(n_chunk=4, s_chunk=1024)  # fp16_total = 4096
    persistent_ids = {0, 1}
    chunk_bytes = {0: 800, 1: 700, 2: 500, 3: 600}  # actual_persistent = 1500
    cm = _stub_chunk_manager(layout, persistent_ids, chunk_bytes)
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)

    fp16_total = layout.N_chunk * layout.S_chunk  # 4096
    F_BM = 5000  # synthetic "fragmentation + activations + deltas" residual
    alpha = ALPHA_FRAGMENTATION  # 1.10

    # Buffer-pool footprint = n_buffer * S * buffer_factor (matches
    # what ``BufferPool.__init__`` pre-allocates and what
    # ``model_state_present_bytes`` charges; not clamped to a
    # prefetch window — the pool reserves every slot for the
    # wrapper's lifetime).
    buffer_factor = 2.0  # mirrors model_state_present_bytes
    n_persist_eff = 2  # prefix; layout has no mandatory_persistent here

    # Case A: persistent_factor = 1.0 (LoRA-with-frozen-base).
    trace_lora = _trace(model_state_bytes=fp16_total)  # 4096 → factor=1.0
    cost_state_lora = int(
        n_persist_eff * layout.S_chunk * 1.0
        + cfg.n_buffer * layout.S_chunk * buffer_factor
    )
    original_peak_lora = int(alpha * (cost_state_lora + F_BM))
    calibrated_lora = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak_lora,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace_lora,
    )

    # Case B: persistent_factor = 5.0 (full FT: fp16 + fp32 master + grads + m + v).
    trace_full = _trace(model_state_bytes=5 * fp16_total)  # factor=5.0
    cost_state_full = int(
        n_persist_eff * layout.S_chunk * 5.0
        + cfg.n_buffer * layout.S_chunk * buffer_factor
    )
    original_peak_full = int(alpha * (cost_state_full + F_BM))
    calibrated_full = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak_full,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace_full,
    )

    # Math check (calibration_alpha = min(alpha, 1.05) = 1.05):
    #   LoRA: 1.05 * (1500 * 1.0 + 1 * 1024 * 2.0 + 5000)         = 1.05 * 8548   = 8975
    #   Full: 1.05 * (1500 * 5.0 + 1 * 1024 * 2.0 + 5000)         = 1.05 * 14548  = 15275
    #   Diff: 1.05 * (5 - 1) * actual_persistent = 1.05 * 4 * 1500 = 6300
    actual_persistent = 1500
    expected_diff = int(1.05 * (5.0 - 1.0) * actual_persistent)
    assert abs((calibrated_full - calibrated_lora) - expected_diff) <= 10, (
        f"calibrated_full={calibrated_full}, calibrated_lora={calibrated_lora}, "
        f"diff={calibrated_full - calibrated_lora}, expected_diff~{expected_diff}"
    )

    # Independent absolute check on the LoRA case (factor=1, math is
    # plain arithmetic so we can pin the value tightly):
    expected_lora = int(
        1.05
        * (
            actual_persistent * 1.0
            + cfg.n_buffer * layout.S_chunk * buffer_factor
            + F_BM
        )
    )
    assert abs(calibrated_lora - expected_lora) <= 10, (
        f"calibrated_lora={calibrated_lora}, expected~{expected_lora}"
    )


def test_calibrate_peak_lora_path_unchanged_from_pre_fix() -> None:
    """Under LoRA-with-frozen-base, persistent_factor ≈ 1 so the new
    full-state scaling collapses to the pre-fix arithmetic (modulo the
    buffer-side 2.0× factor that the cost model has always charged).
    Regression guard so we don't accidentally drift the LoRA path.
    """
    layout = _layout(n_chunk=4, s_chunk=1024)
    cm = _stub_chunk_manager(
        layout,
        persistent_ids={0, 1},
        chunk_param_bytes={0: 1024, 1: 1024, 2: 1024, 3: 1024},
    )
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    trace = _trace(model_state_bytes=layout.N_chunk * layout.S_chunk)
    F_BM = 1000
    cost_state = int(
        2 * layout.S_chunk * 1.0  # n_persist_eff * S * persistent_factor
        + cfg.n_buffer * layout.S_chunk * 2.0  # n_buffer * S * buffer_factor
    )
    original_peak = int(ALPHA_FRAGMENTATION * (cost_state + F_BM))
    calibrated = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
    )
    # Fully packed chunks (1024 each) → actual_persistent = 2 * 1024 = 2048,
    # equal to the cost-model upper bound.
    # Buffer-pool footprint = n_buffer * S * buffer_factor
    #   = 1 * 1024 * 2.0 = 2048 bytes (matches what BufferPool actually
    #   reserves; not clamped to a prefetch window).
    # Calibrated = 1.05 * (2048 + 2048 + 1000) = 1.05 * 5096 = 5350.
    expected = int(1.05 * (2048 + 2048 + 1000))
    assert abs(calibrated - expected) <= 5, (
        f"calibrated={calibrated}, expected={expected}"
    )
