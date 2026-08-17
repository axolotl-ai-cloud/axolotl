# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""MXFP4 base + LoRA at large workload hits the non-fused grouped-Gram dA/dB path.

Guards the edge case: the LoRA weight gradients (dA/dB) must be base-agnostic --
identical for an MXFP4 base and a bf16 dense base that is its dequantization --
and dX must still route through the MX kernel (the non-fused dX path is never
taken for MX). The workload is sized above the fuse-gather threshold so dA/dB go
through the grouped-Gram path rather than the fused-gather kernel.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MXTensor = pytest.importorskip(
    "torchao.prototype.mx_formats.mx_tensor", reason="torchao required for MXFP4"
).MXTensor

from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (  # noqa: E402
    selective_mx_weights_fwd,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (  # noqa: E402
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (  # noqa: E402
    ScatterMoELoRA,
)

_FUSE_GATHER_THRESHOLD = 2**24


def _run(W, x, A, B, sei, ssi, eo, topk, scaling, grad):
    xg = x.detach().requires_grad_(True)
    Ag = A.detach().requires_grad_(True)
    Bg = B.detach().requires_grad_(True)
    out = ScatterMoELoRA.apply(
        xg, W, topk, sei, ssi, eo, Ag, Bg, scaling, None, None, False, False, True, True
    )
    out.backward(grad)
    return out, xg.grad, Ag.grad, Bg.grad


def test_mxfp4_lora_grads_base_agnostic_large_workload():
    dev, dt = "cuda", torch.bfloat16
    E, K, N, M, topk, R = 8, 2048, 2048, 8192, 2, 16  # M_total*max(K,N) > 2**24
    scaling = 0.5
    torch.manual_seed(0)

    W_dense = torch.randn(E, N, K, device=dev, dtype=dt)
    mx = MXTensor.to_mx(W_dense, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
    W_ref = mx.dequantize(dt).contiguous()  # exactly what the MX kernel dequantizes to
    mx_w = selective_mx_weights_fwd(mx, torch.arange(E, device=dev))
    W_kernel = W_ref.transpose(2, 1).contiguous()  # [E, K, N] for the dense kernel

    x = torch.randn(M, K, device=dev, dtype=dt)
    A = torch.randn(R * E, K, device=dev, dtype=dt) * 0.02
    B = torch.randn(N, R * E, device=dev, dtype=dt) * 0.02
    _, top = torch.topk(torch.softmax(torch.randn(M, E, device=dev), -1), topk, dim=-1)
    sei, ssi, eo = flatten_sort_count(top, E)

    assert (
        sei.size(0) * max(K, N) > _FUSE_GATHER_THRESHOLD
    )  # exercises grouped-Gram dA/dB

    grad = torch.randn(sei.size(0), N, device=dev, dtype=dt)
    _, dx_mx, da_mx, db_mx = _run(mx_w, x, A, B, sei, ssi, eo, topk, scaling, grad)
    _, _, da_bf, db_bf = _run(W_kernel, x, A, B, sei, ssi, eo, topk, scaling, grad)

    # LoRA grads never touch the (frozen) base -> identical for MX vs dense base
    assert torch.equal(da_mx, da_bf), (da_mx - da_bf).abs().max().item()
    assert torch.equal(db_mx, db_bf), (db_mx - db_bf).abs().max().item()
    # and everything is finite + actually trained
    for g in (dx_mx, da_mx, db_mx):
        assert torch.isfinite(g).all() and g.abs().sum() > 0
