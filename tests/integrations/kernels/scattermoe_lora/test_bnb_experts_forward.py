# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""bnb-4bit MoE expert LoRA: both the default 1-launch (``moe_bnb_fast``) path and the
chunked-dequant fallback must match a full-dequant reference in forward AND gradients.

The experts are stored as bnb 4-bit (the same parametrization ``quantize_moe_experts``
installs at load via ``replace_parameter_4bit``). The reference dequantizes those exact
4-bit weights to bf16 and runs the standard scattermoe path, so the only differences are
GEMM ordering / bf16 rounding between the two bnb paths and the reference.

Also covers the divisibility guard in ``_selective_dequant_bnb4`` (F3) and the
non-persistent Marlin workspace buffer (F4). All tests are CUDA + bitsandbytes gated.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
bnb_parametrize = pytest.importorskip(
    "bitsandbytes.nn.parametrize", reason="bitsandbytes required"
)

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as ex  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora.chunked_bnb import (  # noqa: E402
    set_bnb_fast,
    set_chunk_size_override,
    set_layer_gc_active,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    scattermoe_experts_forward,
)

DEV = "cuda"
E, H, IM, N, K, R, SC = 8, 256, 128, 32, 2, 8, 0.5


def _bf16_module(gu, dn):
    """Plain bf16 experts module (routes through the standard dequant path)."""
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


class _BnbExperts(torch.nn.Module):
    """Real nn.Module so torch parametrize (bnb 4-bit) can attach to the params."""

    def __init__(self, gu, dn):
        super().__init__()
        self.num_experts = E
        self.act_fn = torch.nn.functional.silu
        self.is_transposed = False
        self.is_concatenated = True
        self.has_bias = False
        self.has_gate = True
        self.gate_up_proj = torch.nn.Parameter(gu, requires_grad=False)
        self.down_proj = torch.nn.Parameter(dn, requires_grad=False)
        bnb_parametrize.replace_parameter_4bit(
            self, "gate_up_proj", compress_statistics=True, quant_type="nf4"
        )
        bnb_parametrize.replace_parameter_4bit(
            self, "down_proj", compress_statistics=True, quant_type="nf4"
        )


def _rel(a, b):
    return (a - b).float().abs().max().item() / max(b.float().abs().max().item(), 1e-6)


def _run(module, lora, idx, w, grad, monkeypatch):
    monkeypatch.setattr(ex, "_has_peft_wrapper", lambda s: True)
    monkeypatch.setattr(
        ex,
        "_unwrap_experts_lora",
        lambda s: (s, (lora[0], lora[1], SC), (lora[2], lora[3], SC)),
    )
    for t in lora:
        t.grad = None
    x = torch.randn(
        N,
        H,
        device=DEV,
        dtype=torch.bfloat16,
        generator=torch.Generator(DEV).manual_seed(7),
    ).requires_grad_(True)
    out = scattermoe_experts_forward(module, x, idx, w)
    out.backward(grad)
    return out.detach(), x.grad.detach(), [t.grad.clone() for t in lora]


@pytest.mark.parametrize("fast", [True, False])
def test_bnb_moe_lora_matches_dequant(fast, monkeypatch):
    """moe_bnb_fast={True,False} both match the full-dequant bf16 reference (fwd + grads)."""
    pytest.importorskip("triton")
    dt = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(0)
    gu = torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1
    dn = torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1

    bnb_mod = _BnbExperts(gu, dn)
    # Reference uses the EXACT dequantized 4-bit weights so only kernel numerics differ.
    gu_deq = bnb_mod.gate_up_proj.detach().contiguous()
    dn_deq = bnb_mod.down_proj.detach().contiguous()

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=dt, generator=g) * 0.05
        ).requires_grad_(True)

    lora = [mk(R * E, H), mk(2 * IM, R * E), mk(R * E, IM), mk(H, R * E)]
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    grad = torch.randn(N, H, device=DEV, dtype=dt, generator=g)

    active = torch.unique(idx.reshape(-1))
    row = (active.long()[:, None] * R + torch.arange(R, device=DEV)[None, :]).reshape(
        -1
    )

    set_layer_gc_active(False)
    set_chunk_size_override(None)
    try:
        set_bnb_fast(fast)
        out_bnb, dx_bnb, gl_bnb = _run(bnb_mod, lora, idx, w, grad, monkeypatch)
    finally:
        set_bnb_fast(None)

    out_ref, dx_ref, gl_ref = _run(
        _bf16_module(gu_deq, dn_deq), lora, idx, w, grad, monkeypatch
    )

    # forward output parity vs the full-dequant reference
    assert torch.isfinite(out_bnb).all()
    assert _rel(out_bnb, out_ref) < 6e-2, "forward output"
    # input grad
    assert torch.isfinite(dx_bnb).all()
    assert _rel(dx_bnb, dx_ref) < 6e-2, "input grad"
    for i, slc in (
        (0, row),
        (1, (slice(None), row)),
        (2, row),
        (3, (slice(None), row)),
    ):
        assert torch.isfinite(gl_bnb[i]).all()
        assert _rel(gl_bnb[i][slc], gl_ref[i][slc]) < 6e-2, f"lora grad {i}"


def test_bnb_fast_and_chunked_agree(monkeypatch):
    """The two bnb paths should agree with each other (same math, different launch shape)."""
    pytest.importorskip("triton")
    dt = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(3)
    gu = torch.randn(E, 2 * IM, H, device=DEV, dtype=dt, generator=g) * 0.1
    dn = torch.randn(E, H, IM, device=DEV, dtype=dt, generator=g) * 0.1
    bnb_mod = _BnbExperts(gu, dn)

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=dt, generator=g) * 0.05
        ).requires_grad_(True)

    lora = [mk(R * E, H), mk(2 * IM, R * E), mk(R * E, IM), mk(H, R * E)]
    idx = torch.randint(0, E, (N, K), device=DEV, generator=g)
    w = torch.rand(N, K, device=DEV, generator=g).to(dt)
    grad = torch.randn(N, H, device=DEV, dtype=dt, generator=g)

    set_layer_gc_active(False)
    set_chunk_size_override(None)
    try:
        set_bnb_fast(True)
        out_fast, dx_fast, gl_fast = _run(bnb_mod, lora, idx, w, grad, monkeypatch)
        set_bnb_fast(False)
        out_chunk, dx_chunk, gl_chunk = _run(bnb_mod, lora, idx, w, grad, monkeypatch)
    finally:
        set_bnb_fast(None)

    assert _rel(out_fast, out_chunk) < 5e-2, "forward output"
    assert _rel(dx_fast, dx_chunk) < 5e-2, "input grad"
    for i in range(4):
        assert _rel(gl_fast[i], gl_chunk[i]) < 5e-2, f"lora grad {i}"


# --- F3: divisibility guard in _selective_dequant_bnb4 -------------------------------------


@pytest.mark.parametrize(
    "expert_shape",
    [
        (8, 8),  # numel 64: divisible by blocksize(64) and 2 -> selective slice path
        (4, 4),  # numel 16: not divisible by blocksize(64) -> full-dequant fallback
    ],
)
def test_selective_dequant_bnb4_matches_full(expert_shape):
    """Selective dequant equals full-dequant+index in BOTH the divisible and fallback cases."""
    pytest.importorskip("triton")
    import bitsandbytes.functional as bnb_f

    from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
        _selective_dequant_bnb4,
    )

    blocksize = 64
    n_exp = 4
    expert_numel = expert_shape[0] * expert_shape[1]
    flat = torch.randn(n_exp * expert_numel, device=DEV, dtype=torch.bfloat16) * 0.1
    packed, qs = bnb_f.quantize_4bit(
        flat, blocksize=blocksize, quant_type="nf4", compress_statistics=False
    )

    full = bnb_f.dequantize_4bit(packed, qs).reshape(n_exp, *expert_shape)
    active = torch.tensor([0, 2], device=DEV, dtype=torch.long)

    out = _selective_dequant_bnb4(packed, qs, active, expert_shape)
    torch.testing.assert_close(out, full[active], atol=2e-2, rtol=2e-2)


# --- F4: Marlin non-expert workspace buffer must not enter state_dict ----------------------


def test_marlin_nonexpert_workspace_not_persistent():
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
        marlin_w4a16_available,
    )

    if not marlin_w4a16_available():
        pytest.skip("Marlin W4A16 kernel not available on this device")

    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.nonexpert_linear import (
        MarlinW4A16Linear,
    )

    w = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16) * 0.1
    mod = MarlinW4A16Linear(w, bias=None)
    sd = mod.state_dict()
    assert "_workspace" not in sd, "scratch workspace leaked into state_dict"
    assert "_workspace" in mod._non_persistent_buffers_set
    # persistent buffers still present
    assert "qweight" in sd and "scales" in sd
