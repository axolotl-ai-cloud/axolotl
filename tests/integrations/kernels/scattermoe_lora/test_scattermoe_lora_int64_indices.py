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
    flatten_sort_count,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Sufficient condition for int32 pointer arithmetic to overflow in the
# Triton kernel: any indexed buffer has >= 2**31 elements.
_INT32_LIMIT = 2**31


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
    """Direct int64 kernel call at the bench shape produces no zero rows.

    With ``INT64_INDICES=True`` the kernel's pointer arithmetic stays in
    int64 across the full output row range, so rows past the would-be
    int32 overflow boundary (``M_block >= 2**31 / y_dim``) are populated
    rather than silently dropped.
    """
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
    assert L_scattered * y_dim >= _INT32_LIMIT, (
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
    overflow_threshold_row = _INT32_LIMIT // y_dim
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
    assert torch.isfinite(out_i64).all().item(), (
        "int64 kernel produced non-finite values at overflow shape"
    )


@pytest.mark.skipif(
    not _has_free_gpu_mem(80.0),
    reason="bench shape needs ~80 GiB free GPU memory",
)
def test_parallel_linear_overflow_takes_int64_kernel_path(monkeypatch):
    """``ParallelLinear.forward`` at the bench shape must route through
    the int64 kernel path (single launch, ``int64_indices=True``).

    The auto-dispatch should set ``needs_int64=True`` and dispatch a
    single ``scatter2scatter`` launch with that flag. A regressed path
    that called the kernel multiple times (e.g. a chunking workaround)
    would invoke ``scatter2scatter_compileable`` more than once and
    fail this assertion.
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

    # Spy on the kernel launches. The kernel-level int64 fix dispatches
    # exactly one ``scatter2scatter_compileable`` call with
    # ``int64_indices=True``. A re-introduced chunking workaround would
    # invoke it once per chunk (>=2 at this shape).
    launches = []
    real_compileable = parallel_experts.kernels.ops.scatter2scatter_compileable

    def _spy_compileable(*args, **kwargs):
        # int64_indices is positional arg 9 (after b, x_grouped, y_grouped).
        launches.append(
            {
                "args_len": len(args),
                "int64": args[9]
                if len(args) > 9
                else kwargs.get("int64_indices", False),
            }
        )
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

    assert len(launches) == 1, (
        f"expected exactly one kernel launch (direct int64 path), got {len(launches)}"
    )
    assert launches[0]["int64"] is True, (
        "auto-dispatch should have set int64_indices=True at the overflow shape"
    )
    assert out.shape == (_T * _TOP_K, 2 * _INTERMEDIATE)
    assert torch.isfinite(out).all().item()


# ─── Smaller-shape overflow (runs on L40S / 24 GiB GPUs) ─────────────────────
# L_scattered * y_dim = 2**32 (2× past 2**31); peak VRAM ≈ 8 GiB.
_SMALL_T = 131072
_SMALL_TOP_K = 8
_SMALL_E = 8
_SMALL_K = 256
_SMALL_INTERMEDIATE = 2048
_SMALL_MIN_FREE_GIB = 12.0


@pytest.mark.skipif(
    not _has_free_gpu_mem(_SMALL_MIN_FREE_GIB),
    reason=f"small overflow shape needs ~{_SMALL_MIN_FREE_GIB:.0f} GiB free GPU memory",
)
def test_dense_scatter2scatter_int64_at_overflow_shape_small():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    y_dim = 2 * _SMALL_INTERMEDIATE
    x = torch.randn(_SMALL_T, _SMALL_K, device=device, dtype=DTYPE)
    W = torch.randn(_SMALL_E, _SMALL_K, y_dim, device=device, dtype=DTYPE) * 0.01

    logits = torch.randn(_SMALL_T, _SMALL_E, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _SMALL_TOP_K, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, _SMALL_E)

    L_scattered = sei.size(0)
    assert L_scattered * y_dim >= _INT32_LIMIT, (
        f"precondition: L_scattered * y_dim ({L_scattered * y_dim}) must "
        f"straddle the int32 overflow boundary ({_INT32_LIMIT})"
    )

    out_i64 = base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_SMALL_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()

    overflow_threshold_row = _INT32_LIMIT // y_dim
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
    assert torch.isfinite(out_i64).all().item()


@pytest.mark.skipif(
    not _has_free_gpu_mem(_SMALL_MIN_FREE_GIB),
    reason=f"small overflow shape needs ~{_SMALL_MIN_FREE_GIB:.0f} GiB free GPU memory",
)
def test_parallel_linear_overflow_takes_int64_kernel_path_small(monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora import parallel_experts
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        parallel_linear,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    y_dim = 2 * _SMALL_INTERMEDIATE
    x = torch.randn(_SMALL_T, _SMALL_K, device=device, dtype=DTYPE)
    W = torch.randn(_SMALL_E, _SMALL_K, y_dim, device=device, dtype=DTYPE) * 0.01

    logits = torch.randn(_SMALL_T, _SMALL_E, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _SMALL_TOP_K, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, _SMALL_E)

    launches = []
    real_compileable = parallel_experts.kernels.ops.scatter2scatter_compileable

    def _spy_compileable(*args, **kwargs):
        launches.append(
            {
                "int64": args[9]
                if len(args) > 9
                else kwargs.get("int64_indices", False),
            }
        )
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
            _SMALL_TOP_K,
            sei,
            ssi,
            eo,
            grouped_in=False,
            grouped_out=True,
        )
        torch.cuda.synchronize()

    assert len(launches) == 1, (
        f"expected exactly one kernel launch (direct int64 path), got {len(launches)}"
    )
    assert launches[0]["int64"] is True, (
        "auto-dispatch should have set int64_indices=True at the overflow shape"
    )
    assert out.shape == (_SMALL_T * _SMALL_TOP_K, y_dim)
    assert torch.isfinite(out).all().item()
