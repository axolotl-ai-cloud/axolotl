# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Regression tests: the gates-branch backward must run in one dtype under bf16 autocast.

Under bf16 autocast the gating bmm is autocast-eligible, so backward gets a bf16
grad_out while the saved output_expanded stays fp32. Pre-fix the d_gates matmul
raises "expected scalar type Float but found BFloat16"; post-fix it returns
finite fp32 grads.
"""

from __future__ import annotations

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora import (
    parallel_linear,
    parallel_linear_lora,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel launch"
)


_T = 64
_E = 8
_K = 2
_H = 32
_O = 48


def _routing(logits: torch.Tensor):
    """Softmax->topk->renorm routing, mirroring layers.py; returns an fp32
    leaf ``gates`` (requires_grad) plus the sorted routing ids."""
    routing_weights = torch.softmax(logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, _K, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    gates = routing_weights.detach().clone().requires_grad_(True)
    sei, ssi, eo = flatten_sort_count(selected_experts, _E)
    return gates, selected_experts, sei, ssi, eo


def test_parallel_linear_gates_backward_under_bf16_autocast():
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    x = torch.randn(_T, _H, device=device, dtype=torch.float32, requires_grad=True)
    # ParallelExperts stores weight as [E, O, H] and passes .permute(0, 2, 1).
    W = (
        torch.randn(_E, _O, _H, device=device, dtype=torch.float32, requires_grad=True)
        * 0.02
    )
    logits = torch.randn(_T, _E, device=device)
    gates, selected_experts, sei, ssi, eo = _routing(logits)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = parallel_linear(
            x,
            W.permute(0, 2, 1),
            _K,
            sei,
            ssi,
            eo,
            gates=gates,
        )
    # repro precondition: autocast makes the output bf16, so backward gets a bf16 grad_out
    assert out.dtype == torch.bfloat16

    grad = torch.randn_like(out)
    # pre-fix: raises at the d_gates matmul; post-fix: fp32 grads
    gx, gW, gg = torch.autograd.grad(out, [x, W, gates], grad_outputs=grad)

    for g in (gx, gW, gg):
        assert g.dtype == torch.float32
        assert torch.isfinite(g).all()

    # check d_gates vs an fp32 reference: pins that grad_out is upcast, not output_expanded downcast
    grad_fp32 = grad.float()
    all_out = torch.einsum("th,eoh->teo", x.detach(), W.detach())
    picked = all_out[torch.arange(_T, device=device).unsqueeze(1), selected_experts]
    ref = (gates.unsqueeze(1) @ picked).squeeze(1)
    (gg_ref,) = torch.autograd.grad(ref, gates, grad_outputs=grad_fp32)
    assert torch.allclose(gg, gg_ref, atol=1e-2)


def test_scattermoe_lora_gates_backward_under_bf16_autocast():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    rank = 4

    x = torch.randn(_T, _H, device=device, dtype=torch.float32, requires_grad=True)
    # Frozen base weights [E, H, O]; only the LoRA adapters get gradients.
    W = torch.randn(_E, _H, _O, device=device, dtype=torch.float32) * 0.02
    lora_A = (
        torch.randn(
            _E * rank, _H, device=device, dtype=torch.float32, requires_grad=True
        )
        * 0.01
    )
    lora_B = (
        torch.randn(
            _O, _E * rank, device=device, dtype=torch.float32, requires_grad=True
        )
        * 0.01
    )
    logits = torch.randn(_T, _E, device=device)
    gates, _, sei, ssi, eo = _routing(logits)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = parallel_linear_lora(
            x,
            W,
            _K,
            sei,
            ssi,
            eo,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            gates=gates,
        )
    assert out.dtype == torch.bfloat16

    grad = torch.randn_like(out)
    # same crash site in the LoRA path; pre-fix raises, post-fix runs in fp32
    gx, gA, gB, gg = torch.autograd.grad(
        out, [x, lora_A, lora_B, gates], grad_outputs=grad
    )

    for g in (gx, gA, gB, gg):
        assert g.dtype == torch.float32
        assert torch.isfinite(g).all()
