# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SonicMoE multi-adapter LoRA: validate the E*T materialization + id remap.

The opaque CUTLASS grouped GEMM is unchanged and isn't exercised here; an exact
PyTorch grouped GEMM stands in for it so the materialize + combined-routing logic
can be checked end-to-end (forward + backward) against a per-tenant reference.
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize
from axolotl.integrations.kernels.libs.sonicmoe.multi_lora import (
    combined_expert_ids,
    materialize_multi_lora_experts,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _grouped_gemm(hidden, w_eff, group_ids, token_idx, out_dim):
    """Exact stand-in for the CUTLASS grouped GEMM: out[token_idx[i]] += h @ W_eff[g_i]."""
    out = hidden.new_zeros(hidden.shape[0], out_dim)
    for i in range(group_ids.shape[0]):
        n = token_idx[i].item()
        out[n] = out[n] + hidden[n] @ w_eff[group_ids[i].item()].t()
    return out


@pytest.mark.parametrize("E,T,r", [(4, 3, 8), (3, 1, 4), (5, 4, 16)])
def test_multilora_materialize_matches_per_tenant(E, T, r):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.float32
    out_dim, in_dim, M = 12, 16, 32
    base = torch.randn(E, out_dim, in_dim, device=dev, dtype=dt)
    A = (torch.randn(T, E, r, in_dim, device=dev, dtype=dt) * 0.1).requires_grad_(True)
    B = (torch.randn(T, E, out_dim, r, device=dev, dtype=dt) * 0.1).requires_grad_(True)
    scaling = torch.rand(T, device=dev, dtype=dt) + 0.5

    hidden = torch.randn(M, in_dim, device=dev, dtype=dt)
    expert = torch.randint(0, E, (M,), device=dev, dtype=torch.int32)
    tenant = torch.randint(0, T, (M,), device=dev, dtype=torch.int32)
    token_idx = torch.arange(M, device=dev, dtype=torch.int32)

    # combined path
    w_eff = materialize_multi_lora_experts(base, A, B, scaling)  # [E*T, out, in]
    g_ids = combined_expert_ids(expert, tenant, token_idx, T)
    out = _grouped_gemm(hidden, w_eff, g_ids, token_idx, out_dim)

    # reference: each token uses its tenant's expert-e effective weight, directly
    Ar = A.detach().clone().requires_grad_(True)
    Br = B.detach().clone().requires_grad_(True)
    out_ref = hidden.new_zeros(M, out_dim)
    for n in range(M):
        e, t = expert[n].item(), tenant[n].item()
        w = base[e] + scaling[t] * (Br[t, e] @ Ar[t, e])
        out_ref[n] = hidden[n] @ w.t()

    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    assert torch.allclose(A.grad, Ar.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(B.grad, Br.grad, atol=1e-4, rtol=1e-4)


def test_t1_reduces_to_single_adapter():
    """T==1 must produce exactly the single-adapter MoELoRAMaterialize result."""
    torch.manual_seed(1)
    dev, dt = "cuda", torch.float32
    E, r, out_dim, in_dim = 4, 8, 12, 16
    base = torch.randn(E, out_dim, in_dim, device=dev, dtype=dt)
    # single-adapter PEFT layout: A [r*E, in], B [out, r*E]
    A_flat = torch.randn(r * E, in_dim, device=dev, dtype=dt) * 0.1
    B_flat = torch.randn(out_dim, r * E, device=dev, dtype=dt) * 0.1
    scaling = 0.5
    w_single = MoELoRAMaterialize.apply(base, A_flat, B_flat, scaling)

    # same weights reshaped into the [T=1, E, ...] stacked layout. Single-adapter
    # PEFT layout is asymmetric: A rows are E-outer/r-inner, B cols r-outer/E-inner.
    A_stk = A_flat.reshape(E, r, in_dim).unsqueeze(0).contiguous()  # [1,E,r,in]
    B_stk = (
        B_flat.reshape(out_dim, r, E).permute(2, 0, 1).unsqueeze(0).contiguous()
    )  # [1,E,out,r]
    w_multi = materialize_multi_lora_experts(
        base, A_stk, B_stk, torch.tensor([scaling], device=dev, dtype=dt)
    )
    assert torch.allclose(w_single, w_multi, atol=1e-5, rtol=1e-5)


def test_combined_ids_routing():
    dev = "cuda"
    T = 3
    expert = torch.tensor([0, 2, 1, 0], device=dev, dtype=torch.int32)
    tenant = torch.tensor([1, 0, 2, 1], device=dev, dtype=torch.int32)  # per token
    token_idx = torch.tensor([0, 1, 2, 3], device=dev, dtype=torch.int32)
    g = combined_expert_ids(expert, tenant, token_idx, T)
    # e*T + t
    assert g.tolist() == [0 * 3 + 1, 2 * 3 + 0, 1 * 3 + 2, 0 * 3 + 1]
    assert g.dtype == torch.int32
