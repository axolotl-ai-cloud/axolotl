"""Correctness for the chunked bf16 lm_head + cross-entropy.

The chunked path must produce a loss and dL/dhidden that match the same-weight
materialized ``F.cross_entropy`` reference (bit-close, not approximate), and both
must be FINITE — it is the convergence-safe alternative to the fused CCE/Liger
paths that collapsed under NVFP4 stochastic-rounding grads.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    pytest.skip("CUDA + bf16 required for bf16 fused CE", allow_module_level=True)

from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import (  # noqa: E402
    bf16_lm_head_cross_entropy,
)


@pytest.mark.parametrize("num_items", [None, 137.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bf16_fused_ce_matches_materialized(num_items, dtype):
    torch.manual_seed(0)
    M, H, V = 192, 256, 4096 + 512  # crosses a vocab-tile boundary
    lm_head = nn.Linear(H, V, bias=False).cuda().to(dtype)
    lm_head.weight.requires_grad_(False)

    hidden = torch.randn(M, H, device="cuda", dtype=dtype)
    labels = torch.randint(0, V, (M,), device="cuda")
    labels[::7] = -100  # mask some tokens

    # reference: full bf16 logits (upcast to fp32), standard CE
    h_ref = hidden.clone().requires_grad_(True)
    logits = (h_ref @ lm_head.weight.t()).float()
    reduction = "sum" if num_items is not None else "mean"
    ref = F.cross_entropy(logits, labels, ignore_index=-100, reduction=reduction)
    if num_items is not None:
        ref = ref / num_items
    ref.backward()

    # fused (shift=False to align with the un-shifted reference)
    h_fused = hidden.clone().requires_grad_(True)
    fused = bf16_lm_head_cross_entropy(
        h_fused, lm_head, labels, num_items_in_batch=num_items, shift=False
    )
    fused.backward()

    assert torch.isfinite(fused).all()
    assert torch.isfinite(h_fused.grad).all()

    loss_rel = (fused - ref).abs() / (ref.abs() + 1e-9)
    grad_rel = (h_fused.grad - h_ref.grad).float().norm() / (
        h_ref.grad.float().norm() + 1e-9
    )
    # fp32 is bit-tight; bf16 carries the GEMM's intrinsic rounding noise.
    loss_tol = 1e-6 if dtype == torch.float32 else 1e-4
    grad_tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert loss_rel < loss_tol, (dtype, num_items, loss_rel.item())
    assert grad_rel < grad_tol, (dtype, num_items, grad_rel.item())


def test_bf16_fused_ce_all_masked_is_finite():
    """A fully-masked microbatch must give a finite zero loss / zero grad."""
    torch.manual_seed(0)
    M, H, V = 64, 128, 8192
    lm_head = nn.Linear(H, V, bias=False).cuda().bfloat16()
    lm_head.weight.requires_grad_(False)

    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.full((M,), -100, device="cuda")
    loss = bf16_lm_head_cross_entropy(hidden, lm_head, labels, shift=False)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert float(loss) == 0.0
    assert torch.isfinite(hidden.grad).all()
    assert float(hidden.grad.abs().max()) == 0.0


def test_bf16_fused_ce_rejects_non_plain_head():
    """Trainable / biased lm_head -> None (caller falls back to materialized)."""
    H, V = 128, 4096
    labels = torch.randint(0, V, (32,), device="cuda")
    hidden = torch.randn(32, H, device="cuda", dtype=torch.bfloat16)

    trainable = nn.Linear(H, V, bias=False).cuda().bfloat16()  # requires_grad default
    assert bf16_lm_head_cross_entropy(hidden, trainable, labels) is None

    biased = nn.Linear(H, V, bias=True).cuda().bfloat16()
    biased.weight.requires_grad_(False)
    assert bf16_lm_head_cross_entropy(hidden, biased, labels) is None
