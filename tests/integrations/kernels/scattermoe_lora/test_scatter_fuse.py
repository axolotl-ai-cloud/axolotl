"""Fused padded gather/scatter vs the torch index op chain (fwd + grads)."""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.scatter_fuse import (
    padded_gather,
    weighted_scatter,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


TOP_K = 4


def _layout(N=97, top_k=TOP_K, H=384, tile=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    E = 8
    idx = torch.randint(0, E, (N, top_k), device="cuda", generator=g)
    flat = idx.reshape(-1)
    rep = torch.arange(N, device="cuda").repeat_interleave(top_k)
    order = flat.argsort()
    rep = rep[order]
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    ptiles = (counts + tile - 1) // tile
    roff = torch.cat([ptiles.new_zeros(1), ptiles.cumsum(0)]) * tile
    coff = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    padded_row = roff[exp_sorted] + (
        torch.arange(exp_sorted.numel(), device="cuda") - coff[exp_sorted]
    )
    Mt = int(ptiles.sum()) * tile
    return rep, padded_row, Mt, N, H


def _assert_bf16_sum_close(got, ref, abs_sum, terms):
    """``got`` sums up to ``terms`` bf16 roundings in arbitrary order (atomics); each is within
    2**-8 of the running magnitude, which never exceeds ``abs_sum`` (the fp32 sum of |addends|),
    so compare against the fp32 ``ref`` per element on that scale."""
    err = (got.float() - ref).abs()
    tol = terms * 2**-8 * abs_sum + 1e-6
    assert (err <= tol).all(), f"{int((err > tol).sum())} elements past the bf16 bound"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padded_gather_matches_index_copy(dtype):
    rep, padded_row, Mt, N, H = _layout()
    hidden = torch.randn(N, H, device="cuda", dtype=dtype, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)

    A = padded_gather(hidden, rep, padded_row, Mt)
    A_ref = hidden_ref.new_zeros(Mt, H).index_copy(0, padded_row, hidden_ref[rep])
    torch.testing.assert_close(A, A_ref)

    gA = torch.randn_like(A)
    A.backward(gA)
    if dtype == torch.float32:
        A_ref.backward(gA)
        torch.testing.assert_close(hidden.grad, hidden_ref.grad)
        return
    # each token's grad sums its TOP_K gathered rows with bf16 atomics
    rows = gA.float()[padded_row]
    ref = torch.zeros(N, H, device="cuda").index_add_(0, rep, rows)
    abs_sum = torch.zeros(N, H, device="cuda").index_add_(0, rep, rows.abs())
    _assert_bf16_sum_close(hidden.grad, ref, abs_sum, terms=TOP_K)


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("router_grad", [False, True])
def test_weighted_scatter_matches_index_add(w_dtype, router_grad):
    rep, padded_row, Mt, N, H = _layout(seed=1)
    dn = torch.randn(Mt, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(rep.numel(), device="cuda", dtype=w_dtype)
    w.requires_grad_(router_grad)

    out = weighted_scatter(dn, w, rep, padded_row, N)
    # fp32 product rounded to bf16, then TOP_K bf16 atomic adds per token
    rows = dn.detach().float()[padded_row] * w.detach().float()[:, None]
    ref = torch.zeros(N, H, device="cuda").index_add_(0, rep, rows)
    abs_sum = torch.zeros(N, H, device="cuda").index_add_(0, rep, rows.abs())
    _assert_bf16_sum_close(out, ref, abs_sum, terms=TOP_K + 1)

    go = torch.randn_like(out)
    out.backward(go)
    # dn's grad is one fp32 product per padded row, rounded once to bf16
    gdn = go.float()[rep] * w.detach().float()[:, None]
    ref_gdn = torch.zeros(Mt, H, device="cuda").index_copy_(0, padded_row, gdn)
    _assert_bf16_sum_close(dn.grad, ref_gdn, ref_gdn.abs(), terms=1)
    if router_grad:
        ref_gw = (dn.detach()[padded_row].float() * go[rep].float()).sum(-1)
        # same fp32 row sums; the bf16 router weight rounds once
        rtol = 1e-5 if w_dtype == torch.float32 else 2**-7
        torch.testing.assert_close(w.grad.float(), ref_gw, rtol=rtol, atol=1e-6)
    else:
        assert w.grad is None


def test_weighted_scatter_router_grad_under_checkpoint():
    from torch.utils.checkpoint import checkpoint

    rep, padded_row, Mt, N, H = _layout(seed=2)
    dn = torch.randn(Mt, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(rep.numel(), device="cuda", requires_grad=True)
    go = torch.randn(N, H, device="cuda", dtype=torch.bfloat16)

    def fwd(dn, w):
        return weighted_scatter(dn * 1, w * 1, rep, padded_row, N)

    fwd(dn, w).backward(go)
    ref = (dn.grad.clone(), w.grad.clone())
    dn.grad = w.grad = None
    checkpoint(fwd, dn, w, use_reentrant=False).backward(go)
    torch.testing.assert_close(dn.grad, ref[0])
    torch.testing.assert_close(w.grad, ref[1])


def test_empty_routing():
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    hidden = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    A = padded_gather(hidden, empty, empty, 64)
    assert A.shape == (64, 32) and A.abs().sum() == 0
    dn = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(0, device="cuda")
    out = weighted_scatter(dn, w, empty, empty, 5)
    assert out.shape == (5, 32) and out.abs().sum() == 0
