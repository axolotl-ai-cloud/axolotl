# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""ScatterMoE multi-adapter LoRA: forward + backward vs a dense reference.

Co-trains several LoRA adapters (tenants) over one frozen expert stack. The base
GEMM keys on expert; the LoRA keys on the combined (expert, tenant) group.
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.multi_lora import (
    build_multilora_routing,
    scatter2scatter_multilora,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(x, w, lora_a, lora_b, expert, tenant, num_tenants, rank, scaling):
    rows = []
    for n in range(x.size(0)):
        e, t = expert[n].item(), tenant[n].item()
        g = e * num_tenants + t
        a_g = lora_a[g * rank : (g + 1) * rank]  # [R, K]
        b_g = lora_b[:, g * rank : (g + 1) * rank]  # [N, R]
        rows.append(x[n] @ w[e] + scaling * ((x[n] @ a_g.t()) @ b_g.t()))
    return torch.stack(rows)


@pytest.mark.parametrize(
    "num_experts,num_tenants,rank",
    [
        (3, 2, 4),
        (4, 3, 8),
        (2, 1, 4),  # T=1 reduces to single-adapter
        (6, 4, 8),  # 24 (expert,tenant) groups vs 48 tokens -> some empty groups
        (4, 4, 64),  # rank 64 exercises BLOCK_R > 16
    ],
)
def test_multilora_forward_backward_matches_reference(num_experts, num_tenants, rank):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.float32
    e, t, r = num_experts, num_tenants, rank
    K, N, M = 16, 12, 48
    scaling = 0.5

    w = torch.randn(e, K, N, device=dev, dtype=dt)
    a = (torch.randn(e * t * r, K, device=dev, dtype=dt) * 0.1).requires_grad_(True)
    b = (torch.randn(N, e * t * r, device=dev, dtype=dt) * 0.1).requires_grad_(True)
    x = torch.randn(M, K, device=dev, dtype=dt, requires_grad=True)
    expert = torch.randint(0, e, (M,), device=dev)
    tenant = torch.randint(0, t, (M,), device=dev)

    se, sc, ss, eo, co = build_multilora_routing(expert, tenant, e, t)
    y = scatter2scatter_multilora(x, w, 1, se, sc, ss, eo, co, a, b, scaling)

    xr = x.detach().clone().requires_grad_(True)
    ar = a.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True)
    yr = _reference(xr, w, ar, br, expert, tenant, t, r, scaling)

    # tf32 matmul tolerance; far tighter than any misroute would produce
    scale = yr.abs().mean().item()
    assert (y - yr).abs().max().item() < 0.05 * max(1.0, scale)

    grad = torch.randn_like(y)
    y.backward(grad)
    yr.backward(grad)
    assert torch.allclose(x.grad, xr.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(a.grad, ar.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(b.grad, br.grad, atol=5e-2, rtol=5e-2)


def test_multilora_true_fp32_when_tf32_disabled():
    """With TF32 off, the fused multi-adapter dA/dB must be true-fp32 (~1e-7), not
    TF32 (~1e-3). Guards the ALLOW_TF32 live-binding fix: an import-time value copy
    left the grouped-Gram dA/dB kernel on TF32 while the forward honored the flag."""
    import axolotl.integrations.kernels.libs.scattermoe_lora.kernels.lora_ops as lo
    import axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops as base_ops

    torch.manual_seed(0)
    dev, dt = "cuda", torch.float32
    e, t, r = 4, 3, 8
    K, N, M, scaling = 16, 12, 48, 0.5

    saved = (torch.backends.cuda.matmul.allow_tf32, lo.ALLOW_TF32, base_ops.ALLOW_TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    lo.ALLOW_TF32 = base_ops.ALLOW_TF32 = False
    try:
        w = torch.randn(e, K, N, device=dev, dtype=dt)
        a = (torch.randn(e * t * r, K, device=dev, dtype=dt) * 0.1).requires_grad_(True)
        b = (torch.randn(N, e * t * r, device=dev, dtype=dt) * 0.1).requires_grad_(True)
        x = torch.randn(M, K, device=dev, dtype=dt, requires_grad=True)
        expert = torch.randint(0, e, (M,), device=dev)
        tenant = torch.randint(0, t, (M,), device=dev)

        se, sc, ss, eo, co = build_multilora_routing(expert, tenant, e, t)
        y = scatter2scatter_multilora(x, w, 1, se, sc, ss, eo, co, a, b, scaling)

        xr = x.detach().clone().requires_grad_(True)
        ar = a.detach().clone().requires_grad_(True)
        br = b.detach().clone().requires_grad_(True)
        yr = _reference(xr, w, ar, br, expert, tenant, t, r, scaling)

        grad = torch.randn_like(y)
        y.backward(grad)
        yr.backward(grad)
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            lo.ALLOW_TF32,
            base_ops.ALLOW_TF32,
        ) = saved

    def _rel(p, q):
        return (p - q).abs().max().item() / max(q.abs().max().item(), 1e-12)

    assert _rel(y, yr) < 1e-4
    assert _rel(x.grad, xr.grad) < 1e-4
    assert _rel(a.grad, ar.grad) < 1e-4  # dA: TF32 leak would be ~1e-3
    assert _rel(b.grad, br.grad) < 1e-4  # dB: TF32 leak would be ~1e-3


def test_misrouting_would_fail():
    """Sanity: a wrong tenant assignment produces O(1) error, confirming the
    tolerance above actually checks routing."""
    torch.manual_seed(1)
    dev, dt = "cuda", torch.float32
    e, t, r, K, N, M = 3, 2, 4, 16, 12, 48
    w = torch.randn(e, K, N, device=dev, dtype=dt)
    a = torch.randn(e * t * r, K, device=dev, dtype=dt)
    b = torch.randn(N, e * t * r, device=dev, dtype=dt)
    x = torch.randn(M, K, device=dev, dtype=dt)
    expert = torch.randint(0, e, (M,), device=dev)
    tenant = torch.randint(0, t, (M,), device=dev)
    wrong_tenant = 1 - tenant  # flip

    se, sc, ss, eo, co = build_multilora_routing(expert, tenant, e, t)
    y = scatter2scatter_multilora(x, w, 1, se, sc, ss, eo, co, a, b, 1.0)
    y_wrong = _reference(x, w, a, b, expert, wrong_tenant, t, r, 1.0)
    assert (y - y_wrong).abs().max().item() > 1.0
