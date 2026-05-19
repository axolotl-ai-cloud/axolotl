# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Parity and overflow-correctness tests for the ``INT64_INDICES``
``tl.constexpr`` knob added to the scattermoe-lora Triton kernels.

The kernel-level fix promotes the per-launch index ranges to int64 only
when the wrapper has detected that ``L_scattered * y_dim`` would
overflow int32. Two properties are tested:

1. **Bitwise parity at small shapes.** When the shape fits in int32,
   ``INT64_INDICES=False`` (the JIT'd int32 variant) and
   ``INT64_INDICES=True`` (the int64 variant) compute the same MMA in
   the same order. Only the index *type* changes, so the outputs must
   be bitwise identical — any deviation indicates the cast leaked into
   the accumulator path.

2. **Overflow correctness at large shapes.** At the previously-failing
   bench config (seq=524288 with 16 shards, L_scattered=262144,
   y_dim=16384 → 2**32 element output), the int64 kernel must populate
   every row of the output and match the chunked workaround within bf16
   tolerance (the chunking workaround changes accumulation order, so
   bit-equality is not expected against it — only against the same-
   layout int32 kernel below the overflow boundary).

The bench-config test is gated by GPU memory; an L_scattered=262144
× 16384 bf16 output is ~8.6 GiB and the up-projection weight is
~64 GiB so we skip when free memory is below the threshold.
"""

from __future__ import annotations

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (
    lora_ops,
    ops as base_ops,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    _scatter2scatter_int32_safe,
    flatten_sort_count,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _requires_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )


pytestmark = _requires_cuda()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _setup(E, K, N, T, top_k, R=16, seed=42):
    """Create synthetic inputs + routing for a (E, K, N, T, k) shape."""
    torch.manual_seed(seed)
    x = torch.randn(T, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE) * 0.02
    lora_A = torch.randn(R * E, K, device=DEVICE, dtype=DTYPE) * 0.01
    lora_B = torch.randn(N, R * E, device=DEVICE, dtype=DTYPE) * 0.01
    logits = torch.randn(T, E, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, E)
    return x, W, lora_A, lora_B, sei, ssi, eo


# ─── Parity tests at non-overflow shapes (bitwise identity) ──────────────────


def test_dense_scatter2scatter_int64_parity_small():
    """Dense scatter2scatter: INT64_INDICES=True == INT64_INDICES=False at small shape."""
    x, W, *_, sei, ssi, _ = _setup(E=8, K=512, N=1024, T=256, top_k=4)
    k = 4
    out_i32 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        x_grouped=False,
        y_grouped=True,
        int64_indices=False,
    )
    out_i64 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_i32, out_i64), (
        "INT64_INDICES must not change accumulation order at non-overflow shapes"
    )


def test_dense_scatter2scatter_int64_parity_ungrouped_out():
    """Same parity but y_grouped=False (uses M_idx scatter lookup, not M_block)."""
    x, W, *_, sei, ssi, _ = _setup(E=8, K=512, N=1024, T=256, top_k=4)
    k = 4
    out_i32 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        x_grouped=False,
        y_grouped=False,
        int64_indices=False,
    )
    out_i64 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        x_grouped=False,
        y_grouped=False,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_i32, out_i64)


def test_scatter2scatter_lora_int64_parity_small():
    """scatter2scatter_lora: int32 vs int64 must agree bitwise."""
    # Pick a shape that lands on the fused path (not split): few-large-experts
    # split threshold is E<=32 with K*N >= 20M, so use a small K*N to stay
    # on the fused kernel.
    x, W, lA, lB, sei, ssi, _ = _setup(E=64, K=256, N=512, T=128, top_k=4)
    k = 4
    scaling = 0.5
    out_i32 = lora_ops.scatter2scatter_lora(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        lora_A=lA,
        lora_B=lB,
        scaling=scaling,
        x_grouped=False,
        y_grouped=True,
        int64_indices=False,
    )
    out_i64 = lora_ops.scatter2scatter_lora(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        lora_A=lA,
        lora_B=lB,
        scaling=scaling,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_i32, out_i64)


def test_scatter2scatter_lora_dX_int64_parity_small():
    """scatter2scatter_lora_dX: int32 vs int64 must agree bitwise."""
    _, W, lA, lB, sei, ssi, _ = _setup(E=64, K=256, N=512, T=128, top_k=4)
    k = 4
    scaling = 0.5
    M_grouped = sei.size(0)  # ungrouped k=1 dy_grouped=True
    dy = torch.randn(M_grouped, W.size(2), device=DEVICE, dtype=DTYPE) * 0.01
    dX_i32 = lora_ops.scatter2scatter_lora_dX(
        DY=dy,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        lora_A=lA,
        lora_B=lB,
        scaling=scaling,
        dy_grouped=True,
        dx_grouped=False,
        int64_indices=False,
    )
    dX_i64 = lora_ops.scatter2scatter_lora_dX(
        DY=dy,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=k,
        lora_A=lA,
        lora_B=lB,
        scaling=scaling,
        dy_grouped=True,
        dx_grouped=False,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(dX_i32, dX_i64)


def test_group_bwd_lora_int64_parity_small():
    """group_bwd_lora (split kernel): int32 vs int64 must agree bitwise."""
    x, W, lA, lB, sei, ssi, eo = _setup(E=16, K=256, N=512, T=128, top_k=2)
    grouped_x = base_ops.group(x, ssi, fan_out=2)
    M = grouped_x.size(0)
    dy = torch.randn(M, W.size(2), device=DEVICE, dtype=DTYPE) * 0.01
    scaling = 0.5
    dA_i32, dB_i32 = lora_ops.group_bwd_lora(
        DY=dy,
        X=grouped_x,
        lora_A=lA,
        lora_B=lB,
        expert_offsets=eo,
        E=16,
        scaling=scaling,
        int64_indices=False,
    )
    dA_i64, dB_i64 = lora_ops.group_bwd_lora(
        DY=dy,
        X=grouped_x,
        lora_A=lA,
        lora_B=lB,
        expert_offsets=eo,
        E=16,
        scaling=scaling,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(dA_i32, dA_i64)
    assert torch.equal(dB_i32, dB_i64)


def test_group_bwd_lora_fused_int64_parity_small():
    """group_bwd_lora_fused: int32 vs int64 must agree within bf16 tolerance.

    Unlike the split kernel, the fused kernel writes dA/dB via
    ``tl.atomic_add``; the order in which (E, K-tile, N-tile) thread blocks
    land their atomics is non-deterministic, so bit-equality is not
    achievable even between two launches of the *same* kernel variant.
    The INT64 cast only changes index *types*, not the MMA path or the
    atomic reduction, so the two variants must still match within
    ``torch.allclose`` bf16 tolerance.
    """
    x, _W, lA, lB, _sei, ssi, eo = _setup(E=16, K=256, N=512, T=128, top_k=2)
    k = 2
    M_total = ssi.size(0)
    N = lB.size(0)
    dy = torch.randn(M_total, N, device=DEVICE, dtype=DTYPE) * 0.01
    scaling = 0.5
    dA_i32, dB_i32 = lora_ops.group_bwd_lora_fused(
        DY=dy,
        X=x,
        lora_A=lA,
        lora_B=lB,
        expert_offsets=eo,
        sorted_scattered_idxs=ssi,
        E=16,
        k=k,
        scaling=scaling,
        dy_grouped=False,
        int64_indices=False,
    )
    dA_i64, dB_i64 = lora_ops.group_bwd_lora_fused(
        DY=dy,
        X=x,
        lora_A=lA,
        lora_B=lB,
        expert_offsets=eo,
        sorted_scattered_idxs=ssi,
        E=16,
        k=k,
        scaling=scaling,
        dy_grouped=False,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    # Tolerance: a few bf16 ULPs is expected from atomic-add ordering nondet.
    assert torch.allclose(dA_i32, dA_i64, rtol=1e-2, atol=5e-4), (
        f"max_abs_diff dA: {(dA_i32.float() - dA_i64.float()).abs().max()}"
    )
    assert torch.allclose(dB_i32, dB_i64, rtol=1e-2, atol=5e-4), (
        f"max_abs_diff dB: {(dB_i32.float() - dB_i64.float()).abs().max()}"
    )


# ─── Overflow correctness at the bench shape ─────────────────────────────────


# Bench-shape constants (mirror ``test_parallel_experts_large_batch_repro.py``).
_T = 32768
_TOP_K = 8
_NUM_EXPERTS = 128
_HIDDEN = 2048
_INTERMEDIATE = 8192


def _has_free_gpu_mem(min_gb: float) -> bool:
    if not torch.cuda.is_available():
        return False
    free, _total = torch.cuda.mem_get_info()
    return free / (1024**3) >= min_gb


@pytest.mark.skipif(
    not _has_free_gpu_mem(80.0),
    reason="bench shape needs ~80 GiB free GPU memory",
)
def test_dense_scatter2scatter_int64_at_overflow_shape():
    """Direct int64 kernel call at the bench shape produces no zero rows
    and matches the chunked workaround within bf16 tolerance."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        _SCATTER2SCATTER_INT32_LIMIT,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    x = torch.randn(_T, _HIDDEN, device=device, dtype=DTYPE)
    W = (
        torch.randn(
            _NUM_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device, dtype=DTYPE
        )
        * 0.01
    )

    logits = torch.randn(_T, _NUM_EXPERTS, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _TOP_K, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, _NUM_EXPERTS)

    L_scattered = sei.size(0)
    y_dim = W.size(-1)
    assert L_scattered * y_dim >= _SCATTER2SCATTER_INT32_LIMIT, (
        "precondition: shape must straddle the int32 overflow boundary"
    )

    out_i64 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()

    # Sample rows on both sides of the int32 overflow boundary.
    overflow_threshold_row = _SCATTER2SCATTER_INT32_LIMIT // y_dim
    sample_rows = [
        0,
        overflow_threshold_row - 1,
        overflow_threshold_row,
        overflow_threshold_row + 1,
        L_scattered - 1,
    ]
    for row in sample_rows:
        assert (out_i64[row] != 0).any().item(), (
            f"int64 kernel left row {row} all-zero (overflow boundary "
            f"= {overflow_threshold_row}, L_scattered = {L_scattered})"
        )

    # Compare to the chunked workaround within bf16 tolerance. Chunking
    # changes accumulation order so we cannot expect bit-equality, but the
    # outputs should be numerically close.
    out_chunked = _scatter2scatter_int32_safe(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=False,
    )
    torch.cuda.synchronize()
    diff = (out_i64.float() - out_chunked.float()).abs().max().item()
    ref = out_chunked.float().abs().max().item()
    # bf16 has ~3 decimal digits, so a few-percent max-abs-diff against the
    # chunked path is expected at this scale; assert a generous relative bound.
    assert diff <= 5e-2 * (ref + 1.0), (
        f"int64 kernel diverges from chunked path: max_abs_diff={diff:g} "
        f"vs ref_max={ref:g}"
    )


@pytest.mark.skipif(
    not _has_free_gpu_mem(80.0),
    reason="bench shape needs ~80 GiB free GPU memory",
)
def test_parallel_linear_overflow_takes_direct_int64_path(monkeypatch):
    """``ParallelLinear.forward`` at the bench shape must route through
    the direct int64 kernel call, *not* the chunking workaround.

    The auto-dispatch should set ``needs_int64=True`` and the
    ``_scatter2scatter_int32_safe`` wrapper's new ``int64_indices=True``
    branch should then bypass the chunking loop entirely.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora import parallel_experts
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        parallel_linear,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    x = torch.randn(_T, _HIDDEN, device=device, dtype=DTYPE)
    W = (
        torch.randn(
            _NUM_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device, dtype=DTYPE
        )
        * 0.01
    )

    logits = torch.randn(_T, _NUM_EXPERTS, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _TOP_K, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, _NUM_EXPERTS)

    # Spy on the chunking path: count how many times the per-chunk
    # ``scatter2scatter_compileable`` is invoked. If the int64 fast
    # path is taken, this stays at zero.
    chunk_calls = {"count": 0}
    real_compileable = (
        parallel_experts.kernels.ops.scatter2scatter_compileable
    )

    def _spy_compileable(*args, **kwargs):
        chunk_calls["count"] += 1
        return real_compileable(*args, **kwargs)

    monkeypatch.setattr(
        parallel_experts.kernels.ops,
        "scatter2scatter_compileable",
        _spy_compileable,
    )

    with torch.no_grad():
        out = parallel_linear(
            x,
            W,
            _TOP_K,
            sei,
            ssi,
            eo,
            grouped_in=False,
            grouped_out=True,
        )
        torch.cuda.synchronize()

    # The direct int64 kernel path doesn't call scatter2scatter_compileable
    # — it goes through ``base_ops.scatter2scatter`` which calls
    # scatter2scatter_compileable exactly once for the whole launch. The
    # chunking path would invoke it once per chunk (>=2 at the bench shape).
    assert chunk_calls["count"] == 1, (
        f"expected exactly one kernel launch (direct int64 path), "
        f"got {chunk_calls['count']} (chunking workaround was taken)"
    )
    assert out.shape == (_T * _TOP_K, 2 * _INTERMEDIATE)
    assert torch.isfinite(out).all().item()
