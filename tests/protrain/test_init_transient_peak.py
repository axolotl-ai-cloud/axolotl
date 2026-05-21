"""Pin predict_init_transient_peak_bytes: iter-1 alloc spike is ~6.9x the steady predictor and must surface for the feasibility gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.api.model_wrapper import (
    predict_init_transient_peak_bytes,
)
from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    HardwareProfile,
    ParamId,
)

# Empirical iter-1 peak (seq-insensitive) for 30B 4-bit Mode-C: dominated by chunked-pool model load, not activations.
AUDIT_ITER1_PEAK_GIB = 17.20

# Sum_chunk_bytes ground truth derived from param_pool + persistent-share at the 17.20 GiB measured peak.
AUDIT_30B_4BIT_SUM_CHUNK_GIB = 15.27


def _make_layout_with_chunk_bytes(
    *, sum_chunk_bytes: int, n_chunk: int, s_chunk: int
) -> ChunkLayout:
    """ChunkLayout whose chunk-byte sum equals sum_chunk_bytes; the stub controls per-param accounting exactly."""
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk={ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)},
        block_to_chunks={BlockId(0): tuple(ChunkId(i) for i in range(n_chunk))},
    )


def _stub_chunk_manager(layout: ChunkLayout, per_chunk_bytes: int) -> SimpleNamespace:
    """Stub matching _chunk_bytes's chunk_manager.model.named_parameters(); meta-device tensors so 15 GiB worth of chunks costs zero RAM."""
    params: list[tuple[str, nn.Parameter]] = []
    for pids in layout.chunks:
        for pid in pids:
            # fp32 = 4 bytes/element; round up so numel * 4 >= per_chunk_bytes.
            numel = max(1, (per_chunk_bytes + 3) // 4)
            param = nn.Parameter(
                torch.empty(numel, dtype=torch.float32, device="meta"),
                requires_grad=False,
            )
            params.append((str(pid), param))

    model = SimpleNamespace(named_parameters=lambda: iter(params))
    return SimpleNamespace(model=model)


def _hw_profile(*, bpe: float, gpu_memory_gib: int = 24) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=gpu_memory_gib * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=bpe,
    )


def test_audit_30b_4bit_modec_within_10pct_of_measured_peak():
    """Prediction must land within 10% of the measured 17.20 GiB iter-1 peak for 30B 4-bit Mode-C."""
    n_chunk = 302
    s_chunk = 67108864  # 64 MiB — matches ext_30b_safe bootstrap log
    total_target_bytes = int(AUDIT_30B_4BIT_SUM_CHUNK_GIB * (1 << 30))
    per_chunk_bytes = total_target_bytes // n_chunk
    actual_sum_bytes = per_chunk_bytes * n_chunk

    layout = _make_layout_with_chunk_bytes(
        sum_chunk_bytes=actual_sum_bytes, n_chunk=n_chunk, s_chunk=s_chunk
    )
    chunk_manager = _stub_chunk_manager(layout, per_chunk_bytes)
    # bpe=0.5 = bnb-4-bit Params4bit (the audit's actual dtype).
    hw = _hw_profile(bpe=0.5)

    predicted_bytes = predict_init_transient_peak_bytes(layout, hw, chunk_manager)
    predicted_gib = predicted_bytes / (1 << 30)
    measured_gib = AUDIT_ITER1_PEAK_GIB

    residual = abs(predicted_gib - measured_gib) / measured_gib
    assert residual <= 0.10, (
        f"iter-1 transient prediction must land within 10% of the "
        f"audit-measured peak; got prediction={predicted_gib:.2f} GiB, "
        f"measured={measured_gib:.2f} GiB, residual={residual * 100:.1f}%"
    )

    # And on the specific empirical anchor: 15.27 GiB x 1.10 = 16.80 GiB,
    # which should match within tens of MiB (per-chunk byte-rounding +
    # the actual int * float multiply at the prediction site).
    expected_anchor_gib = AUDIT_30B_4BIT_SUM_CHUNK_GIB * ALPHA_FRAGMENTATION
    assert predicted_gib == pytest.approx(expected_anchor_gib, rel=0.005), (
        f"prediction should anchor at sum_chunk_bytes x 1.10 = "
        f"{expected_anchor_gib:.2f} GiB; got {predicted_gib:.2f} GiB"
    )


def test_fp16_30b_dense_predicts_full_residence_at_alpha_1_10():
    """fp16 30B dense layout: iter-1 alpha is dtype-agnostic, bpe=2.0 and bpe=0.5 yield identical predictions."""
    # 60 GiB raw model — Llama-30B at fp16 is ~60 GiB params.
    n_chunk = 240
    s_chunk = 1 << 28  # 256 MiB
    total_target_bytes = 60 * (1 << 30)
    per_chunk_bytes = total_target_bytes // n_chunk
    actual_sum_bytes = per_chunk_bytes * n_chunk

    layout = _make_layout_with_chunk_bytes(
        sum_chunk_bytes=actual_sum_bytes, n_chunk=n_chunk, s_chunk=s_chunk
    )
    cm = _stub_chunk_manager(layout, per_chunk_bytes)

    pred_fp16 = predict_init_transient_peak_bytes(layout, _hw_profile(bpe=2.0), cm)
    pred_4bit = predict_init_transient_peak_bytes(layout, _hw_profile(bpe=0.5), cm)

    # iter-1 alpha is dtype-agnostic; the per-dtype reduction only applies in steady state.
    assert pred_fp16 == pred_4bit, (
        f"iter-1 transient alpha must be dtype-agnostic; fp16 pred "
        f"{pred_fp16} != 4-bit pred {pred_4bit}"
    )

    # 60 GiB x 1.10 = 66 GiB exceeds 24 GiB capacity; surfacing this lets the searcher reject all-persistent layouts.
    expected_gib = 60.0 * ALPHA_FRAGMENTATION
    assert pred_fp16 / (1 << 30) == pytest.approx(expected_gib, rel=0.005)


def test_falls_back_to_layout_upper_bound_without_chunk_manager():
    """No chunk_manager: prediction falls back to N_chunk * S_chunk * alpha, the path used pre-runtime by the feasibility gate."""
    n_chunk = 100
    s_chunk = 1 << 26  # 64 MiB
    layout = _make_layout_with_chunk_bytes(
        sum_chunk_bytes=0,  # unused: no chunk_manager
        n_chunk=n_chunk,
        s_chunk=s_chunk,
    )

    pred = predict_init_transient_peak_bytes(layout, _hw_profile(bpe=0.5))
    expected = int(n_chunk * s_chunk * ALPHA_FRAGMENTATION)
    assert pred == expected, (
        f"fallback path: expected {expected} bytes (N_chunk * S_chunk * alpha), got {pred}"
    )


def test_returns_zero_for_empty_layout():
    """Degenerate N_chunk == 0 collapses to 0, the documented "not computed" sentinel."""
    layout = ChunkLayout(
        S_chunk=0,
        N_chunk=0,
        chunks=(),
        param_to_chunk={},
        block_to_chunks={},
    )
    assert predict_init_transient_peak_bytes(layout, _hw_profile(bpe=0.5)) == 0


def test_search_result_default_sentinel_is_zero():
    """Legacy SearchResult constructions without predicted_init_transient_peak_bytes must default to the 0 sentinel."""
    from axolotl.integrations.protrain.types import (
        BlockMode,
        BlockStrategyMap,
        CostConfig,
        SearchResult,
    )

    block_map: BlockStrategyMap = {BlockId(0): BlockMode.NONE}
    sr = SearchResult(
        cfg=CostConfig(n_persist=0, n_buffer=1, n_swap=0, n_checkpoint=0),
        block_map=block_map,
        predicted_peak_bytes=1 << 30,
        predicted_iter_s=0.5,
    )
    assert sr.predicted_init_transient_peak_bytes == 0


def test_chunk_manager_with_empty_named_parameters_falls_back():
    """Stub chunk_manager with no param overlap must fall back to the N_chunk * S_chunk upper bound, not emit 0."""
    n_chunk = 50
    s_chunk = 1 << 26
    layout = _make_layout_with_chunk_bytes(
        sum_chunk_bytes=0, n_chunk=n_chunk, s_chunk=s_chunk
    )
    # Empty named_parameters() → _chunk_bytes returns all-zero dict.
    cm = SimpleNamespace(
        model=SimpleNamespace(named_parameters=lambda: iter([])),
    )
    pred = predict_init_transient_peak_bytes(layout, _hw_profile(bpe=0.5), cm)
    expected_upper_bound = int(n_chunk * s_chunk * ALPHA_FRAGMENTATION)
    assert pred == expected_upper_bound, (
        f"empty chunk_manager should fall back to upper bound "
        f"{expected_upper_bound}, got {pred}"
    )
