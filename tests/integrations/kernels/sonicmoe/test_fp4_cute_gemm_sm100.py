# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SM100 tests for the fp4_cute grouped NVFP4 GEMM (quack blockscaled, varlen_m).

Operands are quantized with the repo's NVFP4 reference so the expected output is
an fp32 grouped matmul over the exact dequantized operand values; the kernel only
adds fp32 accumulation-order error and the final bf16 round.
"""

import inspect

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute import fp4_cute_available
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
    dequantize_nvfp4_ref,
    quantize_nvfp4_ref,
    simulate_nvfp4_quant,
)
from axolotl.integrations.kernels.libs.sonicmoe.sf_layout import (
    build_varlen_sfa,
    gate_up_interleave_perm,
)

pytestmark = [
    pytest.mark.gpu_blackwell,
    pytest.mark.skipif(
        not fp4_cute_available(),
        reason="requires an SM100 (Blackwell) GPU with quack fp4 GEMM",
    ),
]

K = 128
NVFP4_FORMATS = {"bs_format_a": "nvfp4", "bs_format_b": "nvfp4"}


def _cu(counts):
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32
    )


def _quantize_activations(counts, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(sum(counts), K, generator=gen).cuda()
    cu = _cu(counts).cuda()
    a_q, a_s, _ = quantize_nvfp4_ref(x)
    sfa = build_varlen_sfa(a_s.view(torch.uint8), cu).view(torch.float8_e4m3fn)
    return a_q, sfa, cu, dequantize_nvfp4_ref(a_q, a_s)


def _quantize_weights(e, n, seed):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(e, n, K, generator=gen) * K**-0.5).cuda()
    w_q, w_s, _ = quantize_nvfp4_ref(w)
    return w_q, w_s, dequantize_nvfp4_ref(w_q, w_s)


def _grouped_ref(a_deq, w_deq, counts):
    cu = _cu(counts).tolist()
    return torch.cat(
        [a_deq[cu[e] : cu[e + 1]] @ w_deq[e].T for e in range(len(counts))]
    )


def _row_pts(pts, counts):
    return torch.repeat_interleave(pts, torch.tensor(counts, device=pts.device))


def test_quack_gemm_gets_nvfp4_block_formats():
    from quack.gemm import gemm

    from axolotl.integrations.kernels.libs.sonicmoe import fp4_cute

    fp4_cute._blockscaled_format_kwargs.cache_clear()
    if "bs_format_a" in inspect.signature(gemm).parameters:
        assert fp4_cute._blockscaled_format_kwargs() == NVFP4_FORMATS
    else:
        assert fp4_cute._blockscaled_format_kwargs() == {}


@pytest.mark.parametrize(
    ("counts", "n"),
    [
        ([40, 0, 300, 200], 256),  # total_m >= 512: 256x256 tile, 2-CTA cluster
        ([40, 0, 300, 200], 128),  # 256x128 tile, 2-CTA cluster
        ([3, 17, 64, 0], 256),  # 128x128 tile, trailing empty expert
    ],
)
def test_grouped_nvfp4_gemm_matches_dequant_reference(counts, n):
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute import (
        GroupedNvfp4Gemm,
    )

    e = len(counts)
    a_q, sfa, cu, a_deq = _quantize_activations(counts, seed=0)
    w_q, w_s, w_deq = _quantize_weights(e, n, seed=1)
    pts = torch.tensor([0.5, 1.0, 2.0, 1.5], device="cuda")[:e]
    colvec = _row_pts(pts, counts)

    engine = GroupedNvfp4Gemm(n, K, e)
    engine.set_weights(w_q, w_s)
    out = engine.forward(a_q, sfa, cu, colvec=colvec)
    torch.cuda.synchronize()

    ref = _grouped_ref(a_deq, w_deq, counts) * colvec.unsqueeze(1)
    assert out.shape == (sum(counts), n) and out.dtype == torch.bfloat16
    # Operands are exact FP4/e4m3 values, so only bf16 output rounding (~2^-8
    # relative) and fp32 accumulation order separate kernel and reference.
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)

    gen = torch.Generator().manual_seed(2)
    delta = torch.randn(sum(counts), n, generator=gen).cuda().to(torch.bfloat16)
    summed = engine.forward(
        a_q, sfa, cu, colvec=colvec, out=delta.clone(), add_to_output=True
    )
    torch.testing.assert_close(
        summed.float(), ref + delta.float(), rtol=1e-2, atol=2e-2
    )


def test_gated_nvfp4_gemm_matches_dequant_reference():
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute import (
        GroupedNvfp4Gemm,
    )

    counts, n = [40, 0, 300, 200], 256
    e, total_m = len(counts), sum(counts)
    a_q, sfa, cu, a_deq = _quantize_activations(counts, seed=2)
    w_q, w_s, w_deq = _quantize_weights(e, n, seed=3)
    colvec = _row_pts(torch.tensor([0.5, 1.0, 2.0, 1.5], device="cuda"), counts)
    gen = torch.Generator().manual_seed(4)
    aux = (0.1 * torch.randn(total_m, n, generator=gen)).cuda().to(torch.bfloat16)

    engine = GroupedNvfp4Gemm(n, K, e, gated=True, activation="swiglu", concat_b=True)
    engine.set_weights(w_q, w_s)
    postact, preact = engine.forward(a_q, sfa, cu, colvec=colvec, aux=aux)
    torch.cuda.synchronize()

    pre_concat = _grouped_ref(a_deq, w_deq, counts) * colvec.unsqueeze(1)
    pre_ref = pre_concat[:, gate_up_interleave_perm(n, device="cuda")] + aux.float()
    post_ref = torch.nn.functional.silu(pre_ref[:, 0::2]) * pre_ref[:, 1::2]
    torch.testing.assert_close(preact.float(), pre_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(postact.float(), post_ref, rtol=1e-2, atol=1e-2)


def test_grouped_nvfp4_linear_forward_backward():
    nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        grouped_nvfp4_linear,
    )

    counts, n = [40, 0, 300, 200], 256
    e, total_m = len(counts), sum(counts)
    gen = torch.Generator().manual_seed(5)
    pts = torch.tensor([0.5, 1.0, 2.0, 1.5])
    w_dense = torch.randn(e, n, K, generator=gen) * K**-0.5
    qs, ss = zip(
        *(quantize_nvfp4_ref(w_dense[i], pts[i])[:2] for i in range(e)), strict=True
    )
    weight = nvfp4_mod.NVFP4Tensor(
        torch.stack(qs).cuda(),
        torch.stack(ss).cuda(),
        16,
        torch.bfloat16,
        per_tensor_scale=pts.view(-1, 1, 1).cuda(),
    )
    w_deq = weight.dequantize(torch.float32)
    x = torch.randn(total_m, K, generator=gen).cuda().to(torch.bfloat16)
    x.requires_grad_(True)
    end_offsets = _cu(counts)[1:].cuda()

    out = grouped_nvfp4_linear(x, weight, end_offsets)
    grad = torch.randn(total_m, n, generator=gen).cuda().to(torch.bfloat16)
    out.backward(grad)
    torch.cuda.synchronize()

    ref = _grouped_ref(simulate_nvfp4_quant(x.detach().float()), w_deq, counts)
    # Looser than the engine test: the triton activation quantizer may round a
    # few elements to a neighbouring FP4 code relative to the reference.
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=5e-2)

    cu = _cu(counts).tolist()
    dx_ref = torch.cat([grad[cu[i] : cu[i + 1]].float() @ w_deq[i] for i in range(e)])
    torch.testing.assert_close(x.grad.float(), dx_ref, rtol=1e-2, atol=2e-2)
