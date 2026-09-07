# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU tests for the Phase-1 NVFP4 base-weight helpers of the sonicmoe backend."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import (
    dequantize_expert_weight,
    gated_activation,
    grouped_down_gemm,
    grouped_up_gemm,
    is_nvfp4_param,
    resolve_gated_activation,
)


def _random_offsets(E: int, counts: list[int]) -> torch.Tensor:
    assert len(counts) == E
    return torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int64
    )


def test_is_nvfp4_param_false_for_plain_tensor():
    assert is_nvfp4_param(torch.randn(4, 8)) is False
    assert is_nvfp4_param(torch.randn(2, 6, 3)) is False


def test_dequantize_identity_for_dense_tensor():
    w = torch.randn(3, 8, 5)
    out = dequantize_expert_weight(w)
    assert out is w


def test_grouped_up_gemm_matches_manual_loop():
    torch.manual_seed(0)
    E, H, I = 4, 6, 5
    counts = [3, 0, 2, 4]
    T = sum(counts)
    offsets = _random_offsets(E, counts)

    x = torch.randn(T, H, dtype=torch.float32)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float32)

    out = grouped_up_gemm(x, w1, offsets, backend="torch", concat=True)

    ref_rows = []
    start = 0
    for e in range(E):
        end = int(offsets[e + 1])
        ref_rows.append(F.linear(x[start:end], w1[e]))
        start = end
    ref = torch.cat(ref_rows, dim=0)

    assert out.shape == (T, 2 * I)
    torch.testing.assert_close(out, ref)


def test_grouped_down_gemm_matches_manual_loop():
    torch.manual_seed(1)
    E, H, I = 3, 7, 4
    counts = [2, 5, 1]
    T = sum(counts)
    offsets = _random_offsets(E, counts)

    a = torch.randn(T, I, dtype=torch.float32)
    w2 = torch.randn(E, H, I, dtype=torch.float32)

    out = grouped_down_gemm(a, w2, offsets, backend="torch")

    ref_rows = []
    start = 0
    for e in range(E):
        end = int(offsets[e + 1])
        ref_rows.append(F.linear(a[start:end], w2[e]))
        start = end
    ref = torch.cat(ref_rows, dim=0)

    assert out.shape == (T, H)
    torch.testing.assert_close(out, ref)


def test_gated_activation_swiglu_concat():
    torch.manual_seed(2)
    I = 5
    h = torch.randn(7, 2 * I, dtype=torch.float32)
    out = gated_activation(h, "swiglu", concat=True)
    gate, up = h[..., :I], h[..., I:]
    ref = up * F.silu(gate.to(torch.float32)).to(gate.dtype)
    assert out.shape == (7, I)
    torch.testing.assert_close(out, ref)
    # silu is an alias for swiglu.
    torch.testing.assert_close(gated_activation(h, "silu", concat=True), ref)


def test_gated_activation_swiglu_interleaved():
    torch.manual_seed(3)
    I = 4
    h = torch.randn(6, 2 * I, dtype=torch.float32)
    out = gated_activation(h, "swiglu", concat=False)
    gate, up = h[..., 0::2], h[..., 1::2]
    ref = up * F.silu(gate.to(torch.float32)).to(gate.dtype)
    assert out.shape == (6, I)
    torch.testing.assert_close(out, ref)


def test_gated_activation_geglu_concat():
    torch.manual_seed(4)
    I = 3
    h = torch.randn(5, 2 * I, dtype=torch.float32)
    out = gated_activation(h, "geglu", concat=True)
    gate, up = h[..., :I], h[..., I:]
    ref = up * F.gelu(gate.to(torch.float32)).to(gate.dtype)
    torch.testing.assert_close(out, ref)
    # gelu is an alias for geglu.
    torch.testing.assert_close(gated_activation(h, "gelu", concat=True), ref)


def test_gated_activation_geglu_interleaved():
    torch.manual_seed(5)
    I = 4
    h = torch.randn(6, 2 * I, dtype=torch.float32)
    out = gated_activation(h, "geglu", concat=False)
    gate, up = h[..., 0::2], h[..., 1::2]
    ref = up * F.gelu(gate.to(torch.float32)).to(gate.dtype)
    torch.testing.assert_close(out, ref)


def test_gated_activation_gelu_tanh_and_reglu():
    torch.manual_seed(7)
    I = 4
    h = torch.randn(6, 2 * I, dtype=torch.float32)
    gate, up = h[..., :I], h[..., I:]

    # Gemma-style tanh-approx GeGLU (distinct from the erf gelu above).
    tanh_ref = up * F.gelu(gate, approximate="tanh")
    torch.testing.assert_close(gated_activation(h, "gelu_tanh", concat=True), tanh_ref)
    torch.testing.assert_close(
        gated_activation(h, "gelu_pytorch_tanh", concat=True), tanh_ref
    )
    # erf gelu and tanh gelu must not be the same op.
    assert not torch.allclose(gated_activation(h, "gelu", concat=True), tanh_ref)

    reglu_ref = up * F.relu(gate)
    torch.testing.assert_close(gated_activation(h, "reglu", concat=True), reglu_ref)

    with pytest.raises(ValueError, match="unsupported gated activation"):
        gated_activation(h, "mystery", concat=True)


def test_gated_activation_clamped_swiglu_limit():
    torch.manual_seed(8)
    I = 5
    # Force values outside [-L, L] so the clamp is exercised.
    h = torch.randn(7, 2 * I, dtype=torch.float64) * 20.0
    L = 10.0

    out = gated_activation(h, "swiglu", concat=True, limit=L)

    gate, up = h[..., :I], h[..., I:]
    gate_c = gate.clamp(max=L)
    up_c = up.clamp(min=-L, max=L)
    ref = up_c * F.silu(gate_c)
    torch.testing.assert_close(out, ref)
    # limit=None must not clamp.
    assert not torch.allclose(gated_activation(h, "swiglu", concat=True), ref)


def test_resolve_gated_activation_prefers_hidden_activation():
    # Gemma-style config: hidden_activation set, no usable hidden_act.
    gemma = SimpleNamespace(hidden_activation="gelu_pytorch_tanh")
    assert resolve_gated_activation(gemma) == "gelu_pytorch_tanh"
    # hidden_activation wins even when hidden_act is also present.
    both = SimpleNamespace(hidden_activation="gelu_pytorch_tanh", hidden_act="silu")
    assert resolve_gated_activation(both) == "gelu_pytorch_tanh"
    # Most models: only hidden_act.
    assert resolve_gated_activation(SimpleNamespace(hidden_act="SiLU")) == "silu"
    # Neither present -> safe default.
    assert resolve_gated_activation(SimpleNamespace()) == "silu"


def test_full_path_gradcheck_wrt_input():
    torch.manual_seed(6)
    E, H, I = 3, 4, 3
    counts = [2, 1, 2]
    T = sum(counts)
    offsets = _random_offsets(E, counts)

    # Keep magnitudes small so the composed swiglu is not stiff for finite differencing.
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float64) * 0.3
    w2 = torch.randn(E, H, I, dtype=torch.float64) * 0.3
    x = (torch.randn(T, H, dtype=torch.float64) * 0.3).requires_grad_(True)

    def fn(x_in):
        h = grouped_up_gemm(x_in, w1, offsets, backend="torch", concat=True)
        a = gated_activation(h, "swiglu", concat=True)
        return grouped_down_gemm(a, w2, offsets, backend="torch")

    assert torch.autograd.gradcheck(fn, (x,))
