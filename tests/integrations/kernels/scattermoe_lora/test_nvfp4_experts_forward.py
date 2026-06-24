# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""scattermoe routes base NVFP4 + LoRA through the fused kernel (no bf16 dequant).

NVFP4 shares the FP4 E2M1 packing with MXFP4 but scales per 16-element block with an
E4M3 (fp8) value (times an optional per-tensor scale). It reuses the MXWeights container
with ``scale_is_linear=True`` (the E4M3 block scale pre-folded with the per-tensor scale)
and the same fused Triton kernel with a linear-scale branch + block-16. Validated vs a
reference = the same forward on the dequantized weights, within NV rounding tolerance,
through both the standard and EP (sentinel-skip) forwards.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as ex  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    scattermoe_experts_forward,
    scattermoe_experts_forward_ep,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (  # noqa: E402
    is_mxfp4_param,
    is_nvfp4_param,
)

DEV = "cuda"
E, H, IM, N, K, R, SC = 8, 1024, 512, 512, 4, 16, 0.5


def _module(gu, dn):
    return SimpleNamespace(
        num_experts=E,
        gate_up_proj=gu,
        down_proj=dn,
        act_fn=torch.nn.functional.silu,
        is_transposed=False,
        is_concatenated=True,
        has_bias=False,
        has_gate=True,
    )


def _rel(a, b):
    return (a - b).float().abs().max().item() / max(b.float().abs().max().item(), 1e-6)


def test_is_nvfp4_param_disjoint_from_mxfp4():
    nv = NVFP4Tensor.to_nvfp4(
        torch.randn(E, IM, H, device=DEV, dtype=torch.bfloat16), block_size=16
    )
    assert is_nvfp4_param(nv) and not is_mxfp4_param(nv)
    assert not is_nvfp4_param(torch.randn(2, 2, device=DEV))  # plain tensor


def _routing_std(g):
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    return idx, torch.rand(N, K, device=DEV, generator=g).to(torch.bfloat16)


def _routing_ep(g):
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    remote = torch.rand(N, K, device=DEV, generator=g) < 0.5
    remote[:, 0] = False  # >=1 valid slot per token (DeepEP guarantee)
    idx = torch.where(remote, torch.full_like(idx, -1), idx)
    return idx, torch.rand(N, K, device=DEV, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize(
    "fwd,routing",
    [
        (scattermoe_experts_forward, _routing_std),
        (scattermoe_experts_forward_ep, _routing_ep),
    ],
)
def test_nvfp4_fused_matches_dequant(fwd, routing, monkeypatch):
    dt = torch.bfloat16
    # On Blackwell (sm100/sm120) NVFP4+LoRA must run the grouped fp4 path; the legacy fused-MX kernel
    # SIGSEGVs there and is now guarded off. Select the supported grouped backend for the test.
    monkeypatch.setattr(ex.RUNTIME, "fp4_grouped_mode", "nvfp4")
    g = torch.Generator(device=DEV).manual_seed(0)
    gu_nv = NVFP4Tensor.to_nvfp4(
        torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1,
        block_size=16,
    )
    dn_nv = NVFP4Tensor.to_nvfp4(
        torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1, block_size=16
    )
    gu_deq, dn_deq = (
        gu_nv.dequantize(dt).contiguous(),
        dn_nv.dequantize(dt).contiguous(),
    )

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=dt, generator=g) * 0.05
        ).requires_grad_(True)

    A1, B1, A2, B2 = mk(R * E, H), mk(2 * IM, R * E), mk(R * E, IM), mk(H, R * E)
    lora = [A1, B1, A2, B2]
    monkeypatch.setattr(ex, "_has_peft_wrapper", lambda s: True)
    monkeypatch.setattr(
        ex, "_unwrap_experts_lora", lambda s: (s, (A1, B1, SC), (A2, B2, SC))
    )

    idx, w = routing(g)
    valid = idx.reshape(-1) >= 0
    active = torch.unique(torch.sort(idx.reshape(-1)[valid]).values)
    row = (active.long()[:, None] * R + torch.arange(R, device=DEV)[None, :]).reshape(
        -1
    )
    grad = torch.randn(N, H, device=DEV, dtype=dt, generator=g)

    def run(m):
        for t in lora:
            t.grad = None
        x = torch.randn(
            N, H, device=DEV, dtype=dt, generator=torch.Generator(DEV).manual_seed(7)
        ).requires_grad_(True)
        fwd(m, x, idx, w).backward(grad)
        return x.grad.detach(), [t.grad.clone() for t in lora]

    dx_nv, gl_nv = run(_module(gu_nv, dn_nv))  # fused NVFP4 (no dequant)
    dx_bf, gl_bf = run(_module(gu_deq, dn_deq))  # bf16 dequant reference
    assert _rel(dx_nv, dx_bf) < 6e-2
    for i, slc in (
        (0, row),
        (1, (slice(None), row)),
        (2, row),
        (3, (slice(None), row)),
    ):
        assert _rel(gl_nv[i][slc], gl_bf[i][slc]) < 6e-2, f"lora grad {i}"
        assert torch.isfinite(gl_nv[i]).all()
