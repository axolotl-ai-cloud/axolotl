# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""``experts_implementation: scattermoe`` on transformers' ``FP8Experts``, no expert LoRA:
block-FP8 experts run the stock FP8 forward and NVFP4Tensor experts run scattermoe, both
matching a forward on the dequantized weights."""

import pytest
import torch
from torch import nn

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="CUDA with FP8 support (sm89+) required",
    ),
]

from transformers import PretrainedConfig  # noqa: E402
from transformers.integrations import finegrained_fp8  # noqa: E402
from transformers.integrations.moe import use_experts_implementation  # noqa: E402

from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    register_scattermoe_experts,
)
from axolotl.kernels.quantize import dequantize_fp8  # noqa: E402

DEV = "cuda"
E, H, I, T, K = 4, 256, 256, 96, 2  # noqa: E741
BLOCK = (128, 128)

_Experts = use_experts_implementation(
    type("_FP8Experts", (finegrained_fp8.FP8Experts,), {}),
    experts_interface=finegrained_fp8.ALL_FP8_EXPERTS_FUNCTIONS,
)


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


def _quantize(w):
    rows, cols = w.shape[-2] // BLOCK[0], w.shape[-1] // BLOCK[1]
    blocks = w.reshape(E, rows, BLOCK[0], cols, BLOCK[1])
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    qdata = (blocks / scale[:, :, None, :, None]).reshape(w.shape)
    return qdata.to(torch.float8_e4m3fn), scale.float()


def _experts(fmt):
    register_scattermoe_experts()
    cfg = PretrainedConfig(
        hidden_size=H, moe_intermediate_size=I, num_local_experts=E, hidden_act="silu"
    )
    experts = _Experts(cfg, block_size=BLOCK).to(DEV)
    dense = {}
    for name, shape in (("gate_up_proj", (E, 2 * I, H)), ("down_proj", (E, H, I))):
        w = torch.randn(shape, device=DEV) * 0.05
        if fmt == "block_fp8":
            qdata, scale_inv = _quantize(w)
            setattr(experts, name, nn.Parameter(qdata, requires_grad=False))
            setattr(
                experts,
                f"{name}_scale_inv",
                nn.Parameter(scale_inv, requires_grad=False),
            )
            dense[name] = torch.stack(
                [
                    dequantize_fp8(qdata[e], scale_inv[e], torch.bfloat16)
                    for e in range(E)
                ]
            )
        else:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            nv = NVFP4Tensor.to_nvfp4(w.to(torch.bfloat16), block_size=16)
            setattr(experts, name, nn.Parameter(nv, requires_grad=False))
            # the NVFP4 quantizer drops the block-scale placeholders
            experts._parameters.pop(f"{name}_scale_inv", None)
            dense[name] = nv.dequantize(torch.bfloat16)
    return experts, dense


def _reference(experts, dense, hidden, top_k_index, top_k_weights):
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for t in range(hidden.shape[0]):
        for k in range(K):
            e = top_k_index[t, k]
            gate_up = hidden[t] @ dense["gate_up_proj"][e].t()
            h = experts._apply_gate(gate_up) @ dense["down_proj"][e].t()
            out[t] += top_k_weights[t, k].float() * h.float()
    return out.to(hidden.dtype)


def _routing():
    g = torch.Generator(device=DEV).manual_seed(0)
    hidden = torch.randn(T, H, device=DEV, dtype=torch.bfloat16, generator=g)
    top_k_index = torch.randint(0, E, (T, K), device=DEV, generator=g)
    top_k_weights = torch.rand(T, K, device=DEV, generator=g).to(torch.bfloat16)
    return hidden, top_k_index, top_k_weights


@pytest.mark.parametrize("fmt", ["block_fp8", "nvfp4"])
@torch.no_grad()
def test_scattermoe_fp8_experts_forward_matches_dequant_reference(fmt):
    torch.manual_seed(0)
    experts, dense = _experts(fmt)
    experts.config._experts_implementation = "scattermoe"
    hidden, top_k_index, top_k_weights = _routing()

    out = experts(hidden, top_k_index, top_k_weights)
    ref = _reference(experts, dense, hidden, top_k_index, top_k_weights)

    assert out.shape == hidden.shape and out.dtype == hidden.dtype
    # block-FP8: the stock kernel quantizes the activation to fp8 (1x128)
    assert _rel(out, ref) < (8e-2 if fmt == "block_fp8" else 2e-2)
    if fmt == "block_fp8":
        experts.config._experts_implementation = "grouped_mm"
        assert torch.equal(out, experts(hidden, top_k_index, top_k_weights))


def test_scattermoe_block_fp8_experts_backward_matches_stock():
    torch.manual_seed(0)
    experts, _ = _experts("block_fp8")
    hidden, top_k_index, top_k_weights = _routing()
    grads = []
    for impl in ("scattermoe", "grouped_mm"):
        experts.config._experts_implementation = impl
        x = hidden.clone().requires_grad_()
        experts(x, top_k_index, top_k_weights).float().square().sum().backward()
        grads.append(x.grad)
    # the gather backward accumulates with atomics
    assert _rel(grads[0], grads[1]) < 1e-3
