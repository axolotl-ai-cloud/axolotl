# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Parity tests for the GLM-DSA sparse top-k attention kernel and its registered custom ops.

``sparse_attn`` gathers each query's selected keys and runs flash online-softmax over just those;
the reference gathers the same keys in fp32 torch and takes a masked softmax. Forward outputs and
input gradients (dq/dk/dv) must agree. Also asserts the kernels are dispatcher-visible as
``torch.ops.axolotl.*`` (torch.compile opacity + selective-checkpointing matching)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import axolotl.integrations.kernels.libs.glm_dsa.attention_mla_absorb  # noqa: F401,E402
from axolotl.integrations.kernels.libs.glm_dsa.attention_topk import (  # noqa: E402
    sparse_attn,
)

DEV = "cuda"


def test_glm_dsa_ops_registered():
    for name in (
        "glm_dsa_sparse_topk_attn_fwd",
        "glm_dsa_sparse_topk_attn_bwd",
        "glm_dsa_mla_absorb_attn_fwd",
        "glm_dsa_mla_absorb_attn_bwd",
    ):
        assert hasattr(torch.ops.axolotl, name), f"torch.ops.axolotl.{name} missing"


def _causal_window_topk(B, S, T, device):
    """Per query ``s``: the last-T causal positions, padded with DISTINCT future (masked) ones so
    slot-summed gather == de-duplicated dense softmax."""
    idx = torch.empty(B, S, T, dtype=torch.int32, device=device)
    for s in range(S):
        causal = torch.arange(max(0, s - T + 1), s + 1, device=device)
        pad = torch.arange(s + 1, s + 1 + T - causal.numel(), device=device)
        idx[:, s, :] = torch.cat([causal, pad]).to(torch.int32)
    return idx


def _ref_sparse_attn(q, k, v, idx, scale):
    B, H, S, D = q.shape
    DV = v.shape[-1]
    T = idx.shape[-1]
    i = idx.long()[:, None].expand(B, H, S, T)
    kg = torch.gather(k, 2, i.reshape(B, H, S * T, 1).expand(-1, -1, -1, D)).view(
        B, H, S, T, D
    )
    vg = torch.gather(v, 2, i.reshape(B, H, S * T, 1).expand(-1, -1, -1, DV)).view(
        B, H, S, T, DV
    )
    scores = (q.unsqueeze(-2) * kg).sum(-1) * scale
    valid = idx.long()[:, None] <= torch.arange(S, device=q.device)[None, None, :, None]
    p = scores.masked_fill(~valid, float("-inf")).softmax(dim=-1)
    return (p.unsqueeze(-1) * vg).sum(-2)


def test_sparse_topk_matches_reference_fwd_and_grads():
    torch.manual_seed(0)
    B, H, S, D, DV, T = 1, 2, 32, 64, 64, 8
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, S, DV, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    idx = _causal_window_topk(B, S, T, DEV)
    scale = D**-0.5
    # bf16-exact cotangent so both paths backprop the identical dout (unit scale -> the grad
    # tolerance is meaningful, unlike a mean-loss whose grads are ~1e-4)
    g = torch.randn(B, H, S, DV, device=DEV, dtype=torch.bfloat16).float()

    out = sparse_attn(q, k, v, idx, scale)
    (out.float() * g).sum().backward()

    qr = q.detach().float().requires_grad_(True)
    kr = k.detach().float().requires_grad_(True)
    vr = v.detach().float().requires_grad_(True)
    oref = _ref_sparse_attn(qr, kr, vr, idx, scale)
    (oref * g).sum().backward()

    assert torch.isfinite(out).all()
    assert (out.float() - oref).abs().max().item() < 3e-2
    for got, ref in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        assert torch.isfinite(got).all()
        assert (got.float() - ref).abs().max().item() < 3e-2
