# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The DeepSeek-V4 native-fp8 attention LoRA calls transformers' ``fp8_linear`` with its
current signature and matches a dequantized reference forward and backward.

``fp8_linear`` needs the CUDA finegrained-fp8 kernel, so it is replaced by an autospec of the
real function: a kwarg the installed transformers no longer accepts raises ``TypeError`` here
just as it would on GPU."""

from unittest import mock

import pytest
import torch
from torch import nn

pytest.importorskip("triton")  # libs/dsv4/__init__ imports the fused Triton kernels

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers.integrations import finegrained_fp8  # noqa: E402

from axolotl.integrations.kernels.libs.dsv4.lora_fp8 import (  # noqa: E402
    _NativeFp8Lora,
    patch_dsv4_attn_fp8_lora,
)
from axolotl.kernels.quantize import dequantize_fp8  # noqa: E402

BLOCK = [128, 128]


def _quantize(out_features, in_features):
    w = torch.randn(out_features, in_features) * 0.05
    rows, cols = out_features // BLOCK[0], in_features // BLOCK[1]
    blocks = w.reshape(rows, BLOCK[0], cols, BLOCK[1])
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8) / 448.0
    qdata = (blocks / scale[:, None, :, None]).reshape(w.shape)
    return qdata.to(torch.float8_e4m3fn), scale.float()


def _dequant_linear(
    input,
    weight,
    weight_scale_inv,
    block_size=None,
    bias=None,
    activation_scale=None,
    allow_deepgemm=True,
):
    out = input @ dequantize_fp8(weight, weight_scale_inv, input.dtype).t()
    return out if bias is None else out + bias


@pytest.fixture
def fp8_linear(monkeypatch):
    fake = mock.create_autospec(finegrained_fp8.fp8_linear, side_effect=_dequant_linear)
    monkeypatch.setattr(finegrained_fp8, "fp8_linear", fake)
    return fake


def test_native_fp8_lora_matches_dequant_reference(fp8_linear):
    torch.manual_seed(0)
    qdata, scale_inv = _quantize(384, 256)
    x = torch.randn(2, 3, 256, requires_grad=True)
    A = (torch.randn(4, 256) * 0.1).requires_grad_()
    B = (torch.randn(384, 4) * 0.1).requires_grad_()
    bias = torch.randn(384)

    out = _NativeFp8Lora.apply(x, qdata, scale_inv, BLOCK, A, B, 2.0, bias)
    out.square().sum().backward()
    assert fp8_linear.call_count == 1
    assert out.dtype == x.dtype

    x_ref, A_ref, B_ref = (t.detach().clone().requires_grad_() for t in (x, A, B))
    w = dequantize_fp8(qdata, scale_inv, torch.float32)
    ref = x_ref @ w.t() + bias + 2.0 * (x_ref @ A_ref.t()) @ B_ref.t()
    ref.square().sum().backward()

    torch.testing.assert_close(out, ref)
    for got, want in ((x, x_ref), (A, A_ref), (B, B_ref)):
        torch.testing.assert_close(got.grad, want.grad)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_b_proj = finegrained_fp8.FP8Linear(256, 384, block_size=tuple(BLOCK))
        qdata, scale_inv = _quantize(384, 256)
        self.q_b_proj.weight = nn.Parameter(qdata, requires_grad=False)
        self.q_b_proj.weight_scale_inv = nn.Parameter(scale_inv, requires_grad=False)

    def forward(self, x):
        return self.q_b_proj(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, x):
        return self.self_attn(x)


def test_patched_fp8_linear_lora_matches_peft(fp8_linear):
    torch.manual_seed(0)
    model = get_peft_model(
        _Model(),
        LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_b_proj"], init_lora_weights=False
        ),
    )
    assert patch_dsv4_attn_fp8_lora(model, enabled=True) == 1
    layer = model.base_model.model.self_attn.q_b_proj
    lora_params = [layer.lora_A["default"].weight, layer.lora_B["default"].weight]
    x = torch.randn(2, 3, 256)

    def run(forward):
        for p in lora_params:
            p.grad = None
        out = forward(x)
        out.square().sum().backward()
        return out.detach(), [p.grad.clone() for p in lora_params]

    out, grads = run(model)
    ref, ref_grads = run(layer._dsv4_orig_forward)

    torch.testing.assert_close(out, ref)
    for got, want in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(got, want)
