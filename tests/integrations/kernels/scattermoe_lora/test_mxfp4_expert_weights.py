# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Correctness tests for MXFP4 expert weight support in ScatterMoE LoRA.

Validates both strategies against a bf16-dequantized reference:

  Strategy A — selective dequant:
    The kernel runs on the dequantized [num_active, K, N] bf16 buffer,
    so outputs must be bitwise identical to the baseline that supplies
    the same bf16 weights directly.

  Strategy B — fused Triton (when enabled):
    The kernel unpacks MXFP4 + applies E8M0 scales in its K-loop. Output
    differs from the bf16 reference by MX-rounding-tolerance only.

Shapes covered:
  - small:           [E=8, K=128, N=256], M=16, top_k=2, rank=8
  - representative:  [E=32, K=2048, N=1024], M=64, top_k=4, rank=16
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (
    selective_mx_weights_fwd,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    parallel_linear_lora,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
    get_active_experts,
    is_mxfp4_param,
    remap_expert_indices,
    selective_expert_weights,
    selective_lora_weights,
)

torchao = pytest.importorskip("torchao")
from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MX kernels"
)


SHAPES = [
    # (E, K, N, M, top_k, R, seed)
    pytest.param(8, 128, 256, 16, 2, 8, 0, id="small"),
    pytest.param(32, 2048, 1024, 64, 4, 16, 1, id="representative"),
]

# Per-shape Strategy-B tolerances. Forward outputs accumulate K dot-products
# in fp32 then cast to bf16, so they stay within a few ULPs of the bf16
# baseline. The dX path reduces over N (which is typically larger than K and
# uses a different MMA tile layout than the bf16 reference), so we apply a
# looser ULP-aware tolerance there. These are still tight compared to
# torchao's own bf16 vs fp32 GEMM noise.
_STRATEGY_B_FWD_TOL = {
    "small": dict(atol=2e-3, rtol=2e-3),
    "representative": dict(atol=1e-2, rtol=5e-3),
}
# dX tolerance: ~1 bf16 ULP at the typical output magnitude (rtol dominates;
# atol caps near-zero entries where MMA-reordering manifests as full ULP).
_STRATEGY_B_DX_TOL = {
    "small": dict(atol=0.5, rtol=2e-2),
    "representative": dict(atol=2.0, rtol=3e-2),
}
# dA / dB tolerance: the fused dA/dB kernel accumulates via atomic_add from
# multiple N-block programs per expert, and the number of in-flight programs
# differs between the full-E baseline and the compact-active MX path —
# atomic ordering then introduces bf16 ULP-scale noise. Looser than the
# forward bound because the gradients integrate over both M and N.
_STRATEGY_B_LORA_GRAD_TOL = {
    "small": dict(atol=2e-2, rtol=2e-2),
    "representative": dict(atol=2e-1, rtol=3e-2),
}


def _tol_for_shape(K, *, dx: bool = False, lora_grad: bool = False):
    if lora_grad:
        table = _STRATEGY_B_LORA_GRAD_TOL
    elif dx:
        table = _STRATEGY_B_DX_TOL
    else:
        table = _STRATEGY_B_FWD_TOL
    return table["small"] if K <= 128 else table["representative"]


def _make_mxfp4_weights(E, K, N, seed):
    """Build a `[E, N, K]` MXFP4 ``MXTensor`` (block axis = K, the contraction
    axis). Returns (mx, W_ref_bf16) — the bf16 reference is the dequantization
    of the full MX tensor so Strategy A can hit bitwise equality."""
    torch.manual_seed(seed)
    # Natural axolotl storage is [E, N, K] where K is the contraction axis;
    # `experts.gate_up_proj.transpose(2, 1)` then yields [E, K, N] for the kernel.
    W_dense = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE)
    mx = MXTensor.to_mx(W_dense, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
    W_ref = mx.dequantize(DTYPE).contiguous()
    return mx, W_ref


def _setup_routing_and_lora(E, K, N, M, top_k, R, seed):
    torch.manual_seed(seed + 100)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    lora_A = torch.randn(R * E, K, device=DEVICE, dtype=DTYPE) * 0.01
    lora_B = torch.randn(N, R * E, device=DEVICE, dtype=DTYPE) * 0.01
    logits = torch.randn(M, E, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, E)
    return x, lora_A, lora_B, sei, ssi, eo


class _MockExperts:
    """Bare object exposing ``gate_up_proj`` so `selective_expert_weights`
    can branch on it."""

    def __init__(self, mx_param, num_experts):
        self.gate_up_proj = mx_param
        self.num_experts = num_experts


def _run_baseline(
    W_ref,
    x,
    lora_A,
    lora_B,
    scaling,
    sei,
    ssi,
    eo,
    top_k,
    *,
    use_fused_dX: bool = False,
    use_fused_gather: bool = False,
):
    """Full-E bf16 baseline: dense weights, full LoRA, full expert indices."""
    W_kernel = W_ref.transpose(2, 1).contiguous()  # [E, K, N]
    return parallel_linear_lora(
        x,
        W_kernel,
        top_k,
        sei,
        ssi,
        eo,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        use_fused_dX=use_fused_dX,
        use_fused_gather=use_fused_gather,
    )


def _run_strategy_a(mx, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k, E):
    """Strategy A: selective dequant via MXTensor branch in
    `selective_expert_weights`. Compact weights + remapped indices."""
    experts = _MockExperts(mx, E)
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)
    W_compact = (
        selective_expert_weights(experts, "gate_up_proj", active)
        .transpose(2, 1)
        .contiguous()
    )  # [num_active, K, N]
    A_compact, B_compact = selective_lora_weights(lora_A, lora_B, active, E)
    return parallel_linear_lora(
        x,
        W_compact,
        top_k,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_compact,
        lora_B=B_compact,
        scaling=scaling,
    ), active


# ─── Strategy A — bitwise identity vs bf16 baseline ───────────────────────────


@pytest.mark.parametrize("E,K,N,M,top_k,R,seed", SHAPES)
def test_strategy_a_forward_matches_bf16(E, K, N, M, top_k, R, seed):
    mx, W_ref = _make_mxfp4_weights(E, K, N, seed)
    assert is_mxfp4_param(mx)
    x, lora_A, lora_B, sei, ssi, eo = _setup_routing_and_lora(
        E, K, N, M, top_k, R, seed
    )
    scaling = 0.5

    out_baseline = _run_baseline(W_ref, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k)
    out_a, _ = _run_strategy_a(mx, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k, E)

    assert out_baseline.shape == out_a.shape
    assert torch.equal(out_baseline, out_a), (
        f"Strategy A forward must match bf16 baseline bitwise. "
        f"max abs diff = {(out_baseline - out_a).abs().max().item()}"
    )


@pytest.mark.parametrize("E,K,N,M,top_k,R,seed", SHAPES)
def test_strategy_a_backward_matches_bf16(E, K, N, M, top_k, R, seed):
    """Forward + backward parity. dX must be bitwise identical; the LoRA
    grads dA/dB are compared on the active expert slices only (the full
    LoRA tensors differ in shape between baseline and compact paths)."""
    mx, W_ref = _make_mxfp4_weights(E, K, N, seed)
    x_base, lora_A_base, lora_B_base, sei, ssi, eo = _setup_routing_and_lora(
        E, K, N, M, top_k, R, seed
    )
    scaling = 0.5

    # Baseline backward
    x_b = x_base.detach().clone().requires_grad_(True)
    A_b = lora_A_base.detach().clone().requires_grad_(True)
    B_b = lora_B_base.detach().clone().requires_grad_(True)
    out_b = _run_baseline(W_ref, x_b, A_b, B_b, scaling, sei, ssi, eo, top_k)
    grad_out = torch.randn_like(out_b)
    out_b.backward(grad_out)

    # Strategy A backward
    x_a = x_base.detach().clone().requires_grad_(True)
    experts = _MockExperts(mx, E)
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)
    W_compact = (
        selective_expert_weights(experts, "gate_up_proj", active)
        .transpose(2, 1)
        .contiguous()
    )
    A_full = lora_A_base.detach().clone().requires_grad_(True)
    B_full = lora_B_base.detach().clone().requires_grad_(True)
    A_compact, B_compact = selective_lora_weights(A_full, B_full, active, E)
    out_a = parallel_linear_lora(
        x_a,
        W_compact,
        top_k,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_compact,
        lora_B=B_compact,
        scaling=scaling,
    )
    out_a.backward(grad_out)

    # Forward parity (Strategy A contract: bitwise identical to bf16 baseline).
    # Asserted here so a forward bug that produces a constant offset (and
    # therefore zero gradient delta) doesn't slip past the bwd-only checks.
    assert torch.equal(out_b, out_a), (
        f"forward mismatch (Strategy A): max abs diff = "
        f"{(out_b - out_a).abs().max().item()}"
    )

    # dX: bitwise identical
    assert torch.equal(x_b.grad, x_a.grad), (
        f"dX mismatch (Strategy A): max abs diff = "
        f"{(x_b.grad - x_a.grad).abs().max().item()}"
    )

    # dA / dB — gather active slices from the baseline full grads and compare
    row_idx = (
        active.long()[:, None] * R + torch.arange(R, device=DEVICE)[None, :]
    ).reshape(-1)
    dA_b_active = A_b.grad[row_idx]
    dB_b_active = B_b.grad[:, row_idx]

    # A_compact is a view (advanced indexing produces a copy, so the grad lands
    # on the full lora_A via the slice). torch.autograd flows back through
    # selective_lora_weights, so A_full.grad has gradient on rows for active
    # experts only.
    dA_a_active = A_full.grad[row_idx]
    dB_a_active = B_full.grad[:, row_idx]

    assert torch.equal(dA_b_active, dA_a_active), (
        f"dA active slice mismatch: max diff = "
        f"{(dA_b_active - dA_a_active).abs().max().item()}"
    )
    assert torch.equal(dB_b_active, dB_a_active), (
        f"dB active slice mismatch: max diff = "
        f"{(dB_b_active - dB_a_active).abs().max().item()}"
    )


# ─── Strategy A — backward through fused dX/gather paths ──────────────────────


@pytest.mark.parametrize("use_fused_dX", [False, True])
@pytest.mark.parametrize("use_fused_gather", [False, True])
def test_strategy_a_backward_fused_variants(use_fused_dX, use_fused_gather):
    """Strategy A must match baseline across all four fused-bwd flag
    combinations exercised in production by ``HFScatterMoEGatedMLP``.
    Asserts parity on dX, dA, and dB (active expert slice for dA/dB)."""
    E, K, N, M, top_k, R = 8, 128, 256, 16, 2, 8
    mx, W_ref = _make_mxfp4_weights(E, K, N, seed=7)
    x_base, lora_A_base, lora_B_base, sei, ssi, eo = _setup_routing_and_lora(
        E, K, N, M, top_k, R, seed=7
    )
    scaling = 0.25

    def run(W_kernel, sei_, eo_, k_, lora_A_in, lora_B_in, grad_out):
        x_g = x_base.detach().clone().requires_grad_(True)
        A_g = lora_A_in.detach().clone().requires_grad_(True)
        B_g = lora_B_in.detach().clone().requires_grad_(True)
        out = parallel_linear_lora(
            x_g,
            W_kernel,
            k_,
            sei_,
            ssi,
            eo_,
            lora_A=A_g,
            lora_B=B_g,
            scaling=scaling,
            use_fused_dX=use_fused_dX,
            use_fused_gather=use_fused_gather,
        )
        out.backward(grad_out)
        return out, x_g.grad, A_g.grad, B_g.grad

    # Non-trivial grad (a constant grad zeros out cross-token differences in
    # the fused-gather accumulation, which can mask reordering bugs).
    torch.manual_seed(7)
    W_baseline = W_ref.transpose(2, 1).contiguous()
    # Forward once on baseline shape to size grad_out.
    out_shape_probe = parallel_linear_lora(
        x_base,
        W_baseline,
        top_k,
        sei,
        ssi,
        eo,
        lora_A=lora_A_base,
        lora_B=lora_B_base,
        scaling=scaling,
    )
    grad_out = torch.randn_like(out_shape_probe) * 0.1

    out_b, dx_b, dA_b, dB_b = run(
        W_baseline,
        sei,
        eo,
        top_k,
        lora_A_base,
        lora_B_base,
        grad_out,
    )

    experts = _MockExperts(mx, E)
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)
    W_compact = (
        selective_expert_weights(experts, "gate_up_proj", active)
        .transpose(2, 1)
        .contiguous()
    )
    A_compact, B_compact = selective_lora_weights(lora_A_base, lora_B_base, active, E)
    out_a, dx_a, dA_a, dB_a = run(
        W_compact,
        remapped,
        compact_offsets,
        top_k,
        A_compact,
        B_compact,
        grad_out,
    )

    # Strategy A is bitwise on forward / dX (same bf16 weights, same kernel).
    assert torch.equal(out_b, out_a), (
        f"forward mismatch: max diff = {(out_b - out_a).abs().max().item()}"
    )
    assert torch.equal(dx_b, dx_a), (
        f"dX mismatch: max diff = {(dx_b - dx_a).abs().max().item()}"
    )

    # dA / dB — compare active-expert slice of the dense baseline grads
    # against the compact-path grads. Same row_idx pattern as
    # test_strategy_a_backward_matches_bf16. Bitwise (``torch.equal``) holds
    # for forward and dX, but the fused dA/dB kernel uses ``atomic_add``
    # across N-block programs and the in-flight program count differs
    # between the full-E baseline and the compact-active path; combined
    # with FMA reordering, this introduces 1–2 bf16 ULPs (~2e-4 at the
    # values seen here) on the ``use_fused_dX=True`` configs. Use a
    # tolerance an order of magnitude below that — tight enough to catch
    # any real bug but tolerant of the unavoidable atomic-order noise.
    lora_grad_tol = dict(atol=1e-3, rtol=1e-3)
    row_idx = (
        active.long()[:, None] * R + torch.arange(R, device=DEVICE)[None, :]
    ).reshape(-1)
    dA_b_active = dA_b[row_idx]
    dB_b_active = dB_b[:, row_idx]

    assert torch.allclose(dA_b_active, dA_a, **lora_grad_tol), (
        f"dA active slice mismatch: max diff = "
        f"{(dA_b_active - dA_a).abs().max().item()}"
    )
    assert torch.allclose(dB_b_active, dB_a, **lora_grad_tol), (
        f"dB active slice mismatch: max diff = "
        f"{(dB_b_active - dB_a).abs().max().item()}"
    )


# ─── Strategy B — fused MXFP4 Triton kernel ──────────────────────────────────


# MX rounding tolerance — the Triton path can reorder FMAs vs the torchao
# dequant + bf16 matmul reference, and the dequant arithmetic is
# fp32-codebook * fp32-scale -> bf16. See ``_STRATEGY_B_TOL`` above.


def _run_strategy_b(mx, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k, E):
    """Strategy B: pass MXWeights container directly to parallel_linear_lora;
    the fused MX kernel does dequant inside the K-loop."""
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)
    mx_active = selective_mx_weights_fwd(mx, active)
    A_compact, B_compact = selective_lora_weights(lora_A, lora_B, active, E)
    return parallel_linear_lora(
        x,
        mx_active,
        top_k,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_compact,
        lora_B=B_compact,
        scaling=scaling,
    ), active


@pytest.mark.parametrize("E,K,N,M,top_k,R,seed", SHAPES)
def test_strategy_b_forward_matches_bf16(E, K, N, M, top_k, R, seed):
    """Strategy B forward must match bf16 baseline within MX rounding tol."""
    mx, W_ref = _make_mxfp4_weights(E, K, N, seed)
    x, lora_A, lora_B, sei, ssi, eo = _setup_routing_and_lora(
        E, K, N, M, top_k, R, seed
    )
    scaling = 0.5
    tol = _tol_for_shape(K)

    out_baseline = _run_baseline(W_ref, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k)
    out_b, _ = _run_strategy_b(mx, x, lora_A, lora_B, scaling, sei, ssi, eo, top_k, E)

    assert out_baseline.shape == out_b.shape
    diff = (out_baseline.float() - out_b.float()).abs()
    rel = diff / (out_baseline.float().abs() + 1e-6)
    assert torch.allclose(out_baseline, out_b, **tol), (
        f"Strategy B forward exceeds MX tolerance: max abs={diff.max().item():.4e}, "
        f"max rel={rel.max().item():.4e}"
    )


@pytest.mark.parametrize("E,K,N,M,top_k,R,seed", SHAPES)
def test_strategy_b_backward_matches_bf16(E, K, N, M, top_k, R, seed):
    """Strategy B forward+backward; dX, dA, dB compared to bf16 baseline on
    the active expert slice within MX rounding tol."""
    mx, W_ref = _make_mxfp4_weights(E, K, N, seed)
    x_base, lora_A_base, lora_B_base, sei, ssi, eo = _setup_routing_and_lora(
        E, K, N, M, top_k, R, seed
    )
    scaling = 0.5
    fwd_tol = _tol_for_shape(K)
    dx_tol = _tol_for_shape(K, dx=True)
    lg_tol = _tol_for_shape(K, lora_grad=True)

    # Baseline — match the MX path's fused-bwd kernel selection so dA/dB MMA
    # accumulation order is the same and bf16 noise stays at single ULPs.
    x_b = x_base.detach().clone().requires_grad_(True)
    A_b = lora_A_base.detach().clone().requires_grad_(True)
    B_b = lora_B_base.detach().clone().requires_grad_(True)
    out_b = _run_baseline(
        W_ref,
        x_b,
        A_b,
        B_b,
        scaling,
        sei,
        ssi,
        eo,
        top_k,
        use_fused_dX=True,
        use_fused_gather=True,
    )
    grad_out = torch.randn_like(out_b)
    out_b.backward(grad_out)

    # Strategy B
    x_s = x_base.detach().clone().requires_grad_(True)
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)
    mx_active = selective_mx_weights_fwd(mx, active)
    A_full = lora_A_base.detach().clone().requires_grad_(True)
    B_full = lora_B_base.detach().clone().requires_grad_(True)
    A_compact, B_compact = selective_lora_weights(A_full, B_full, active, E)
    out_s = parallel_linear_lora(
        x_s,
        mx_active,
        top_k,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_compact,
        lora_B=B_compact,
        scaling=scaling,
    )
    out_s.backward(grad_out)

    # Forward parity within MX rounding tol — asserted here so a forward bug
    # that produces a constant offset (and therefore zero gradient delta)
    # doesn't slip past the bwd-only checks.
    assert torch.allclose(out_b, out_s, **fwd_tol), (
        f"Strategy B forward mismatch: max abs diff = "
        f"{(out_b - out_s).abs().max().item():.4e}"
    )

    # dX tolerance (looser; see _STRATEGY_B_DX_TOL comment)
    assert torch.allclose(x_b.grad, x_s.grad, **dx_tol), (
        f"Strategy B dX mismatch: max diff = "
        f"{(x_b.grad - x_s.grad).abs().max().item():.4e}"
    )

    # Uniform-scaling drift guard: the allclose bound above is generous to
    # accommodate accumulated bf16 MMA noise over N. A bug that scales every
    # dX element by a constant factor (e.g. an off-by-one on the E8M0
    # exponent shifting the whole tile by 2x) would still pass that bound.
    # Catch it by requiring the per-element ratio std stays small after
    # masking out near-zero baseline elements (where the ratio is dominated
    # by quantization noise rather than uniform drift).
    bf16_dX = x_b.grad.float()
    mx_dX = x_s.grad.float()
    eps = 1e-3 * bf16_dX.abs().max().clamp(min=1e-6)
    mask = bf16_dX.abs() > eps
    if mask.any():
        ratio = mx_dX[mask] / bf16_dX[mask]
        ratio_std = ratio.std().item()
        assert ratio_std < 0.5, (
            f"Strategy B dX uniform-scaling drift: std(mx/bf16) = "
            f"{ratio_std:.4f} (mean = {ratio.mean().item():.4f}); "
            f"a uniform multiplicative bug would slip past the allclose "
            f"bound but is caught here."
        )

    # dA / dB — compare active expert slices (use forward tolerance — these
    # come from the LoRA-only grad path which doesn't touch the W matmul)
    row_idx = (
        active.long()[:, None] * R + torch.arange(R, device=DEVICE)[None, :]
    ).reshape(-1)
    dA_b_active = A_b.grad[row_idx]
    dA_s_active = A_full.grad[row_idx]
    dB_b_active = B_b.grad[:, row_idx]
    dB_s_active = B_full.grad[:, row_idx]

    assert torch.allclose(dA_b_active, dA_s_active, **lg_tol), (
        f"Strategy B dA active slice mismatch: max diff = "
        f"{(dA_b_active - dA_s_active).abs().max().item():.4e}"
    )
    assert torch.allclose(dB_b_active, dB_s_active, **lg_tol), (
        f"Strategy B dB active slice mismatch: max diff = "
        f"{(dB_b_active - dB_s_active).abs().max().item():.4e}"
    )
