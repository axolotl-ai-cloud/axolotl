# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""scattermoe Triton path for gpt_oss-style experts (transposed [E,H,2I] weights,
interleaved gate/up, per-expert bias, clamped sigmoid-GLU).

Validated against an eager reference: output and every gradient must match. The LoRA
fusion is the same ``scatter2scatter_lora`` the standard path uses, so the eager
reference folds the kernel's delta ``ΔW_e = scaling * A_e^T @ W_B[e]`` into W_eff.
TF32 is disabled so the fp32 comparison is tight (the kernel GEMMs otherwise use TF32).
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as ex  # noqa: E402
import axolotl.integrations.kernels.libs.scattermoe_lora.kernels.lora_ops as _lo  # noqa: E402
import axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops as _ops  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    scattermoe_experts_forward,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (  # noqa: E402
    peft_lora_B_to_scattermoe,
    peft_lora_to_scattermoe,
)

DEV = "cuda"
ALPHA, LIMIT = 1.702, 7.0
E, H, IM, N, K, R, SC = 8, 256, 128, 64, 4, 16, 0.5


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = _lo.ALLOW_TF32, _ops.ALLOW_TF32
    _lo.ALLOW_TF32 = _ops.ALLOW_TF32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old
    _lo.ALLOW_TF32, _ops.ALLOW_TF32 = a, b


def _glu(gate_up):
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    return (up + 1) * (gate * torch.sigmoid(gate * ALPHA))


def _module(dt):
    def mk(*s):
        return (torch.randn(*s, device=DEV, dtype=dt) * 0.02).requires_grad_(True)

    return SimpleNamespace(
        num_experts=E,
        gate_up_proj=mk(E, H, 2 * IM),
        gate_up_proj_bias=mk(E, 2 * IM),
        down_proj=mk(E, IM, H),
        down_proj_bias=mk(E, H),
        alpha=ALPHA,
        limit=LIMIT,
        is_transposed=True,
        is_concatenated=False,
        has_bias=True,
        has_gate=True,
    )


def _eager(x, m, idx, w, weff=None):
    gup, dp = weff if weff else (m.gate_up_proj, m.down_proj)
    fe = idx.reshape(-1)
    xg = x.repeat_interleave(K, dim=0)
    gate_up = torch.bmm(xg.unsqueeze(1), gup[fe]).squeeze(1) + m.gate_up_proj_bias[fe]
    h = _glu(gate_up)
    o = torch.bmm(h.unsqueeze(1), dp[fe]).squeeze(1) + m.down_proj_bias[fe]
    o = o * w.reshape(-1, 1)
    tok = torch.arange(x.size(0), device=DEV).repeat_interleave(K)
    return torch.zeros(x.size(0), H, device=DEV, dtype=x.dtype).index_add_(0, tok, o)


def _rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_gptoss_base_matches_eager(dt):
    m = _module(dt)
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    grad = torch.randn(N, H, device=DEV, dtype=dt)
    ps = [m.gate_up_proj, m.gate_up_proj_bias, m.down_proj, m.down_proj_bias]

    def run(eager):
        for p in ps:
            p.grad = None
        x = torch.randn(
            N, H, device=DEV, dtype=dt, generator=torch.Generator(DEV).manual_seed(3)
        ).requires_grad_(True)
        out = (
            _eager(x, m, idx, w) if eager else scattermoe_experts_forward(m, x, idx, w)
        )
        out.backward(grad)
        return out.detach(), x.grad, [p.grad.clone() for p in ps]

    ok, dxk, gk = run(False)
    oe, dxe, ge = run(True)
    tol = 2e-4 if dt == torch.float32 else 5e-2
    assert _rel(ok, oe) < tol
    assert _rel(dxk, dxe) < tol
    for a, b in zip(gk, ge, strict=True):
        assert _rel(a, b) < tol


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_gptoss_lora_matches_eager(dt, monkeypatch):
    g = torch.Generator(device=DEV).manual_seed(0)
    base = SimpleNamespace(
        num_experts=E,
        gate_up_proj=torch.randn(E, H, 2 * IM, device=DEV, dtype=dt, generator=g)
        * 0.02,
        gate_up_proj_bias=torch.randn(E, 2 * IM, device=DEV, dtype=dt, generator=g)
        * 0.02,
        down_proj=torch.randn(E, IM, H, device=DEV, dtype=dt, generator=g) * 0.02,
        down_proj_bias=torch.randn(E, H, device=DEV, dtype=dt, generator=g) * 0.02,
        alpha=ALPHA,
        limit=LIMIT,
        is_transposed=True,
        is_concatenated=False,
        has_bias=True,
        has_gate=True,
    )

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=dt, generator=g) * 0.05
        ).requires_grad_(True)

    pA1, pB1 = mk(R * E, H), mk(2 * IM, R * E)
    pA2, pB2 = mk(R * E, IM), mk(H, R * E)
    lora = [pA1, pB1, pA2, pB2]
    gup_l = (*peft_lora_to_scattermoe(pA1, pB1, E, R), SC)
    dwn_l = (*peft_lora_to_scattermoe(pA2, pB2, E, R), SC)
    monkeypatch.setattr(ex, "_has_peft_wrapper", lambda s: True)
    monkeypatch.setattr(ex, "_unwrap_experts_lora", lambda s: (s, gup_l, dwn_l))

    # eager W_eff: ΔW_e = scaling * A_e^T @ W_B[e], W_B = lora_B.T.reshape(E,R,out)
    A1 = pA1.reshape(E, R, H)
    WB1 = peft_lora_B_to_scattermoe(pB1, E, R).t().reshape(E, R, 2 * IM)
    A2 = pA2.reshape(E, R, IM)
    WB2 = peft_lora_B_to_scattermoe(pB2, E, R).t().reshape(E, R, H)
    gup_eff = base.gate_up_proj + SC * torch.bmm(A1.transpose(1, 2), WB1)
    dp_eff = base.down_proj + SC * torch.bmm(A2.transpose(1, 2), WB2)

    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    grad = torch.randn(N, H, device=DEV, dtype=dt)

    def run(eager):
        for t in lora:
            t.grad = None
        x = torch.randn(
            N, H, device=DEV, dtype=dt, generator=torch.Generator(DEV).manual_seed(3)
        ).requires_grad_(True)
        out = (
            _eager(x, base, idx, w, weff=(gup_eff, dp_eff))
            if eager
            else scattermoe_experts_forward(base, x, idx, w)
        )
        out.backward(grad)
        return out.detach(), x.grad, [t.grad.clone() for t in lora]

    ok, dxk, gk = run(False)
    oe, dxe, ge = run(True)
    tol = 3e-4 if dt == torch.float32 else 5e-2
    assert _rel(ok, oe) < tol
    assert _rel(dxk, dxe) < tol
    for a, b in zip(gk, ge, strict=True):
        assert _rel(a, b) < tol
        assert a.abs().sum() > 0  # LoRA grads actually populated
