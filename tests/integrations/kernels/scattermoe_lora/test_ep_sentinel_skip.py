# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""scattermoe DeepEP local path skips ``-1`` sentinels instead of compute-and-mask.

Oracle: on the SAME experts module, ``scattermoe_experts_forward_ep`` (raw -1-tagged
routing) must match ``scattermoe_experts_forward`` with sentinels mapped to expert 0 /
weight 0 -- output, dX, and LoRA dA/dB. Sentinel slots carry weight 0 so both compute
the same math; the skip path just omits the wasted rows (and the load-imbalanced
expert-0 bucket the masked path creates).
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as ex  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    scattermoe_experts_forward,
    scattermoe_experts_forward_ep,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (  # noqa: E402
    peft_lora_to_scattermoe,
)

DEV = "cuda"


def _module(E, H, IM, dt, rank, monkeypatch):
    self = SimpleNamespace(
        num_experts=E,
        gate_up_proj=torch.randn(E, 2 * IM, H, device=DEV, dtype=dt) * 0.02,
        down_proj=torch.randn(E, H, IM, device=DEV, dtype=dt) * 0.02,
        act_fn=torch.nn.functional.silu,
        is_transposed=False,
        is_concatenated=True,
        has_bias=False,
        has_gate=True,
    )
    lora = None
    if rank:
        A1 = (torch.randn(rank * E, H, device=DEV, dtype=dt) * 0.02).requires_grad_(
            True
        )
        B1 = (
            torch.randn(2 * IM, rank * E, device=DEV, dtype=dt) * 0.02
        ).requires_grad_(True)
        A2 = (torch.randn(rank * E, IM, device=DEV, dtype=dt) * 0.02).requires_grad_(
            True
        )
        B2 = (torch.randn(H, rank * E, device=DEV, dtype=dt) * 0.02).requires_grad_(
            True
        )
        gup = (*peft_lora_to_scattermoe(A1, B1, E, rank), 0.5)
        dwn = (*peft_lora_to_scattermoe(A2, B2, E, rank), 0.5)
        monkeypatch.setattr(ex, "_has_peft_wrapper", lambda m: True)
        monkeypatch.setattr(ex, "_unwrap_experts_lora", lambda m: (m, gup, dwn))
        lora = (A1, B1, A2, B2)
    return self, lora


def _routing(N, K, E, ep, dt):
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    remote = torch.rand(N, K, device=DEV, generator=g) < (1 - 1.0 / ep)
    remote[:, 0] = False  # >=1 valid slot per received token (DeepEP guarantee)
    idx = torch.where(remote, torch.full_like(idx, -1), idx)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    return idx, w


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 16])
@pytest.mark.parametrize("ep", [2, 4])
def test_ep_skip_matches_masked(dt, rank, ep, monkeypatch):
    E, H, IM, N, K = 32, 512, 256, 512, 8
    self, lora = _module(E, H, IM, dt, rank, monkeypatch)
    idx, w = _routing(N, K, E, ep, dt)
    safe_idx = torch.where(idx >= 0, idx, torch.zeros_like(idx))
    safe_w = w * (idx >= 0).to(dt)
    grad = torch.randn(N, H, device=DEV, dtype=dt)

    def run(fn, ii, ww):
        x = torch.randn(
            N,
            H,
            device=DEV,
            dtype=dt,
            generator=torch.Generator(DEV).manual_seed(1),
        ).requires_grad_(True)
        if lora:
            for t in lora:
                t.grad = None
        fn(self, x, ii, ww).backward(grad)
        return x.grad.clone(), [t.grad.clone() for t in lora] if lora else []

    # forward equality
    with torch.no_grad():
        x = torch.randn(N, H, device=DEV, dtype=dt)
        o_m = scattermoe_experts_forward(self, x, safe_idx, safe_w)
        o_s = scattermoe_experts_forward_ep(self, x, idx, w)
    tol = 1e-4 if dt == torch.float32 else 4e-2
    assert torch.allclose(o_s, o_m, rtol=tol, atol=tol), (o_s - o_m).abs().max().item()

    dx_m, gl_m = run(scattermoe_experts_forward, safe_idx, safe_w)
    dx_s, gl_s = run(scattermoe_experts_forward_ep, idx, w)
    assert torch.allclose(dx_s, dx_m, rtol=tol, atol=tol), (
        (dx_s - dx_m).abs().max().item()
    )
    for a, b in zip(gl_s, gl_m, strict=True):
        assert torch.allclose(a, b, rtol=tol, atol=tol), (a - b).abs().max().item()


def test_ep_skip_handles_all_sentinel_token():
    """A row with every slot remote (-1) contributes nothing and must not crash."""
    E, H, IM, N, K = 8, 256, 128, 16, 4
    self, _ = _module(E, H, IM, torch.float32, 0, None)
    idx = torch.full((N, K), -1, device=DEV, dtype=torch.long)
    idx[1:, 0] = 0  # token 0 fully remote; others have one valid slot
    w = torch.rand(N, K, device=DEV)
    out = scattermoe_experts_forward_ep(self, torch.randn(N, H, device=DEV), idx, w)
    assert out.shape == (N, H)
    assert torch.equal(out[0], torch.zeros(H, device=DEV))
