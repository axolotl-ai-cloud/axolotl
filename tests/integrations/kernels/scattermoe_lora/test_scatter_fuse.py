"""Fused padded gather/scatter vs the torch index op chain (fwd + grads)."""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.scatter_fuse import (
    padded_gather,
    weighted_scatter,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _layout(N=97, top_k=4, H=384, tile=64, seed=0):
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
    A_ref.backward(gA)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("router_grad", [False, True])
def test_weighted_scatter_matches_index_add(w_dtype, router_grad):
    rep, padded_row, Mt, N, H = _layout(seed=1)
    dn = torch.randn(Mt, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(rep.numel(), device="cuda", dtype=w_dtype)
    w.requires_grad_(router_grad)
    dn_ref = dn.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(router_grad)

    out = weighted_scatter(dn, w, rep, padded_row, N)
    out_ref = dn_ref.new_zeros(N, H).index_add(
        0, rep, dn_ref[padded_row] * w_ref[:, None].to(dn_ref.dtype)
    )
    # fused multiplies in fp32 before the bf16 round; ref rounds w to bf16 first
    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)

    go = torch.randn_like(out)
    out.backward(go)
    out_ref.backward(go)
    torch.testing.assert_close(dn.grad, dn_ref.grad, rtol=2e-2, atol=2e-2)
    if router_grad:
        torch.testing.assert_close(w.grad, w_ref.grad, rtol=2e-2, atol=2e-1)
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
