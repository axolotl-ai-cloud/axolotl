"""Pin :func:`predict_init_transient_peak_bytes` against the audit data.

Coverage audit Block G (Phase 2) measured the GPU high-water mark during
the iter-1 init transient — the brief window between HF Trainer's full
GPU model construction and ProTrain's
:meth:`ChunkManager.materialize_offload`. The audit observed:

    +-----------------------------------------+---------+---------+---------+
    | Config                                  | pred GiB| meas it1| meas std|
    +-----------------------------------------+---------+---------+---------+
    | ext_30b_safe seq=512 4-bit Mode-C       |  2.49   |  17.20  |  2.91   |
    | A1 30B seq=1024  4-bit Mode-C           |  2.50   |  17.20  |  3.50   |
    | A2 30B seq=2048  4-bit Mode-C           |  2.54   |  17.20  |  4.68   |
    +-----------------------------------------+---------+---------+---------+

The steady predictor under-calls iter-1 by ~6.9× — surfacing the
transient on :class:`SearchResult` lets downstream consumers (search
feasibility gate, telemetry) catch the OOM at search time rather than
at iter 1.

The bootstrap log for ``ext_30b_safe`` records the chunked-pool size
that produced the 17.20 GiB peak:

    ChunkManager.materialize_offload: offloaded 299 non-persistent chunks
    to pinned CPU memory (param_pool=16.236 GB, grad_pool=0.243 GB;
    precise_size=True/True), freed 16.236 GB on GPU
    ProTrain: materialize_offload freed 15.12 GB (reported),
    alloc 17.20 -> 2.08 GB (torch measured)

That maps to a total ``sum_chunk_bytes`` of roughly
``param_pool + persistent_share ≈ 16.236 GB + (3/302 * 16.236 GB)
≈ 16.40 GB ≈ 15.27 GiB`` (302 chunks total, 3 persistent / 299
non-persistent for this Llama-30B Mode-C layout).

This test reconstructs that chunk-byte footprint via a synthetic
:class:`ChunkLayout` + stub chunk_manager and asserts the prediction
lands within 10% of the measured 17.20 GiB. Pure unit test — no live
model load needed.
"""

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

# Empirical iter-1 peak observed by audit Block G across three 30B
# 4-bit Mode-C configurations (seq ∈ {512, 1024, 2048}). The peak is
# essentially seq-insensitive at this scale because the init transient
# is dominated by the chunked-pool's GPU-resident model load BEFORE
# any forward / activation allocation kicks in.
AUDIT_ITER1_PEAK_GIB = 17.20

# Audit log derivation for ext_30b_safe seq=512 4-bit Mode-C:
#   param_pool=16.236 GB (decimal) → 15.121 GiB
#   grad_pool=0.243 GB (decimal) → 0.226 GiB
#   3 persistent chunks worth ≈ 3/299 × 16.236 GB ≈ 0.163 GB → 0.152 GiB
#   total sum_chunk_bytes ≈ 15.27 GiB
#
# The grad_pool sits in pinned host memory, not GPU, so the strict
# "sum_chunk_bytes" the prediction model consumes is the param-side
# total — but the GPU-resident pre-materialize state also includes a
# small grad-allocator stub, so 15.27 GiB is the most honest single
# number for the empirical sum that produced the 17.20 GiB measured
# peak. The audit's ``alpha_iter1 = 17.20 / 2.49 ≈ 6.9x`` is computed
# against the *steady* prediction; here we compute against the
# *sum_chunk_bytes* ground-truth that the new transient prediction
# anchors against.
AUDIT_30B_4BIT_SUM_CHUNK_GIB = 15.27


def _make_layout_with_chunk_bytes(
    *, sum_chunk_bytes: int, n_chunk: int, s_chunk: int
) -> ChunkLayout:
    """Build a ChunkLayout whose actual chunk-byte sum equals ``sum_chunk_bytes``.

    The layout's chunks each own a single ParamId placeholder; the
    actual per-param byte counts are supplied by ``_stub_chunk_manager``
    so the test controls the ``sum_chunk_bytes`` ground truth exactly.
    """
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk={ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)},
        block_to_chunks={BlockId(0): tuple(ChunkId(i) for i in range(n_chunk))},
    )


def _stub_chunk_manager(layout: ChunkLayout, per_chunk_bytes: int) -> SimpleNamespace:
    """Stub matching :func:`_chunk_bytes`'s ``chunk_manager.model.named_parameters()``.

    Builds one fp32 nn.Parameter per chunk sized so
    ``numel * element_size == per_chunk_bytes``; the helper sums these
    to get the total ``sum_chunk_bytes``.

    CodeRabbit R4-#3 (Major): construct the parameters on the ``meta``
    device so ``numel()`` + ``element_size()`` report the right byte
    accounting without allocating real storage. The audit's
    ``ext_30b_safe`` chunk-byte footprint is ~15 GiB across 302
    64-MiB chunks; allocating that for real on CI would OOM most
    runners. Meta tensors preserve dtype + shape metadata (which is
    all ``_chunk_bytes`` reads) and contribute zero RAM bytes.
    """
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
    """Pin the prediction against the audit's 30B 4-bit Mode-C iter-1 peak.

    Reconstruct the audit's ext_30b_safe chunk-byte footprint
    (15.27 GiB sum_chunk_bytes across 302 chunks at S_chunk=64 MiB) and
    assert the prediction (sum_chunk_bytes × ALPHA_FRAGMENTATION) lands
    within 10% of the measured 17.20 GiB iter-1 peak.

    Expected prediction: 15.27 GiB × 1.10 = 16.80 GiB
    Measured peak:                          17.20 GiB
    Residual: |16.80 - 17.20| / 17.20 ≈ 2.3%  → well inside the 10% bar.
    """
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

    # And on the specific empirical anchor: 15.27 GiB × 1.10 = 16.80 GiB,
    # which should match within tens of MiB (per-chunk byte-rounding +
    # the actual int * float multiply at the prediction site).
    expected_anchor_gib = AUDIT_30B_4BIT_SUM_CHUNK_GIB * ALPHA_FRAGMENTATION
    assert predicted_gib == pytest.approx(expected_anchor_gib, rel=0.005), (
        f"prediction should anchor at sum_chunk_bytes × 1.10 = "
        f"{expected_anchor_gib:.2f} GiB; got {predicted_gib:.2f} GiB"
    )


def test_fp16_30b_dense_predicts_full_residence_at_alpha_1_10():
    """Smoke: a fp16 30B-class dense layout (no offload) anchors against
    the same α=1.10 ceiling. The transient prediction matches the
    steady prediction in Mode-A because there is no separable
    transient window — every chunk stays persistent. The test pins
    the formula's dtype-agnostic behaviour: bpe=2.0 produces the same
    α=1.10 multiplier as bpe=0.5.
    """
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

    # Same α regardless of dtype — the per-dtype reduction does not
    # apply at iter-1 transient time (audit Block G architectural
    # decision; see docstring on ``predict_init_transient_peak_bytes``).
    assert pred_fp16 == pred_4bit, (
        f"iter-1 transient α must be dtype-agnostic; fp16 pred "
        f"{pred_fp16} != 4-bit pred {pred_4bit}"
    )

    # Anchor: 60 GiB × 1.10 = 66 GiB (will not fit on a 3090, which is
    # exactly the signal the searcher's feasibility gate needs to see —
    # surfacing this lets it reject the all-persistent layout and pick
    # an offload-aware Mode-C plan instead).
    expected_gib = 60.0 * ALPHA_FRAGMENTATION
    assert pred_fp16 / (1 << 30) == pytest.approx(expected_gib, rel=0.005)


def test_falls_back_to_layout_upper_bound_without_chunk_manager():
    """When ``chunk_manager`` is None, the prediction falls back to
    ``N_chunk * S_chunk * α`` — the loose upper bound matching the
    layout's soft-cap. This is the path the searcher feasibility gate
    will take before the runtime exists.
    """
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
        f"fallback path: expected {expected} bytes (N_chunk * S_chunk * α), got {pred}"
    )


def test_returns_zero_for_empty_layout():
    """Degenerate ``N_chunk == 0`` collapses to 0 — the SearchResult
    sentinel value, so consumers can keep treating
    ``predicted_init_transient_peak_bytes == 0`` as "not computed".
    """
    layout = ChunkLayout(
        S_chunk=0,
        N_chunk=0,
        chunks=(),
        param_to_chunk={},
        block_to_chunks={},
    )
    assert predict_init_transient_peak_bytes(layout, _hw_profile(bpe=0.5)) == 0


def test_search_result_default_sentinel_is_zero():
    """Backward-compat: every legacy SearchResult construction site
    that doesn't pass ``predicted_init_transient_peak_bytes`` lands at
    0 — the documented "not computed" sentinel.
    """
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
    """Defensive: when a stub chunk_manager has no overlap with the
    layout's param ids (sum collapses to 0), the prediction falls back
    to the ``N_chunk * S_chunk`` upper bound rather than emitting a
    nonsensical 0 — keeps the searcher's feasibility gate honest when
    a test or external caller passes a degenerate stub.
    """
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
