# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""scattermoe_experts_forward routes base MXFP4 + LoRA through the fused MX kernel
(no bf16 dequant of the frozen base) and is bit-for-bit unaffected for non-MXFP4 models.

The fused path keeps the packed 4-bit weights and dequantizes inside the kernel K-loop.
Validated vs a reference = the same forward on the dequantized weights, within MX
rounding tolerance, on output / dX / the active-expert slice of LoRA dA, dB.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MXTensor = pytest.importorskip(
    "torchao.prototype.mx_formats.mx_tensor", reason="torchao required"
).MXTensor

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as ex  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    scattermoe_experts_forward,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (  # noqa: E402
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (  # noqa: E402
    get_active_experts,
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


def test_mxfp4_forward_uses_fused_kernel_matches_dequant(monkeypatch):
    dt = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(0)

    def mx(W):
        return MXTensor.to_mx(W, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)

    gu_mx = mx(torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1)
    dn_mx = mx(torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1)
    gu_deq, dn_deq = (
        gu_mx.dequantize(dt).contiguous(),
        dn_mx.dequantize(dt).contiguous(),
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

    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    sei, _, _ = flatten_sort_count(idx, E)
    active = get_active_experts(sei, E)
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
        scattermoe_experts_forward(m, x, idx, w).backward(grad)
        return x.grad.detach(), [t.grad.clone() for t in lora]

    dx_mx, gl_mx = run(_module(gu_mx, dn_mx))  # fused MX path
    dx_bf, gl_bf = run(_module(gu_deq, dn_deq))  # bf16 dequant reference

    def rel(a, b):
        return (a - b).float().abs().max().item() / max(
            b.float().abs().max().item(), 1e-6
        )

    assert rel(dx_mx, dx_bf) < 6e-2
    for i, slc in (
        (0, row),
        (1, (slice(None), row)),
        (2, row),
        (3, (slice(None), row)),
    ):
        assert rel(gl_mx[i][slc], gl_bf[i][slc]) < 6e-2, f"lora grad {i}"
        assert torch.isfinite(gl_mx[i]).all()


def test_mxfp4_ep_matches_dequant(monkeypatch):
    """Fused MXFP4 through the EP (sentinel-skip) forward matches the bf16-dequant ref,
    and _sonicmoe_local routes there on a device the sonic-moe kernel can't run."""
    from axolotl.integrations.expert_parallel.experts_fn import _sonicmoe_local
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        scattermoe_experts_forward_ep,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        _sonicmoe_kernel_supported,
    )

    dt = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(0)

    def mx(W):
        return MXTensor.to_mx(W, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)

    gu_mx = mx(torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1)
    dn_mx = mx(torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1)
    gu_deq, dn_deq = (
        gu_mx.dequantize(dt).contiguous(),
        dn_mx.dequantize(dt).contiguous(),
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

    # routing with -1 remote sentinels (post-DeepEP-dispatch shape)
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    remote = torch.rand(N, K, device=DEV, generator=g) < 0.5
    remote[:, 0] = False
    idx = torch.where(remote, torch.full_like(idx, -1), idx)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    active = torch.unique(torch.sort(idx.reshape(-1)[idx.reshape(-1) >= 0]).values)
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
        scattermoe_experts_forward_ep(m, x, idx, w).backward(grad)
        return x.grad.detach(), [t.grad.clone() for t in lora]

    dx_mx, gl_mx = run(_module(gu_mx, dn_mx))
    dx_bf, gl_bf = run(_module(gu_deq, dn_deq))

    def rel(a, b):
        return (a - b).float().abs().max().item() / max(
            b.float().abs().max().item(), 1e-6
        )

    assert rel(dx_mx, dx_bf) < 6e-2
    for i, slc in (
        (0, row),
        (1, (slice(None), row)),
        (2, row),
        (3, (slice(None), row)),
    ):
        assert rel(gl_mx[i][slc], gl_bf[i][slc]) < 6e-2, f"lora grad {i}"

    if (
        not _sonicmoe_kernel_supported()
    ):  # e.g. sm_120: sonicmoe+EP falls back to scattermoe
        out = _sonicmoe_local(
            _module(gu_mx, dn_mx), torch.randn(N, H, device=DEV, dtype=dt), idx, w
        )
        assert out.shape == (N, H) and torch.isfinite(out).all()


def test_mxfp4_without_lora_falls_back_to_dequant(monkeypatch):
    """MXFP4 base with no LoRA must not hit the fused (LoRA-only) MX kernel."""
    dt = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(0)

    def mx(W):
        return MXTensor.to_mx(W, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)

    m = _module(
        mx(torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1),
        mx(torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1),
    )
    monkeypatch.setattr(ex, "_has_peft_wrapper", lambda s: False)  # no LoRA
    x = torch.randn(N, H, device=DEV, dtype=dt)
    idx = torch.randint(0, E, (N, K), device=DEV)
    w = torch.rand(N, K, device=DEV).to(dt)
    out = scattermoe_experts_forward(
        m, x, idx, w
    )  # dequant-on-cast path, must not raise
    assert out.shape == (N, H) and torch.isfinite(out).all()
