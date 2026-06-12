"""Dequantize-add for torchao FP4 tensors (enables PEFT target_parameters LoRA).

PEFT >= 0.19 registers a ``base + delta`` parametrization for ``target_parameters``
LoRA; when the base is a frozen torchao NVFP4/MXFP4 tensor, the add must dequantize.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.torchao_fp4_add import (  # noqa: E402
    patch_torchao_fp4_add,
)

DEV = "cuda"


def test_nvfp4_dispatch_add_matches_dequant():
    patch_torchao_fp4_add()
    w = torch.randn(4, 16, 32, device=DEV, dtype=torch.bfloat16)
    nv = NVFP4Tensor.to_nvfp4(w, block_size=16)
    delta = torch.randn(4, 16, 32, device=DEV, dtype=torch.bfloat16) * 0.01

    # The path PEFT's parametrization hits (__torch_dispatch__).
    out = torch.ops.aten.add.Tensor(nv, delta)
    ref = nv.dequantize(torch.bfloat16) + delta
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out, ref, atol=1e-3)

    # Commutativity: FP4 as the right operand too.
    out2 = torch.ops.aten.add.Tensor(delta, nv)
    assert torch.allclose(out2, ref, atol=1e-3)


def test_gradient_flows_to_delta():
    """The load-bearing property: grad reaches the LoRA delta through the FP4 add."""
    patch_torchao_fp4_add()
    base = NVFP4Tensor.to_nvfp4(
        torch.randn(4, 16, 32, device=DEV, dtype=torch.bfloat16), block_size=16
    )
    delta = torch.zeros(4, 16, 32, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    merged = torch.ops.aten.add.Tensor(base, delta)  # what the parametrization computes
    merged.float().pow(2).sum().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_patch_is_idempotent():
    patch_torchao_fp4_add()
    op = torch.ops.aten.add.Tensor
    before = id(NVFP4Tensor._ATEN_OP_TABLE[NVFP4Tensor][op])
    patch_torchao_fp4_add()
    after = id(NVFP4Tensor._ATEN_OP_TABLE[NVFP4Tensor][op])
    assert before == after  # not re-registered
    assert torch.ops.aten.add_.Tensor in NVFP4Tensor._ATEN_OP_TABLE[NVFP4Tensor]


def test_mxfp4_add_if_available():
    mx_mod = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    MXTensor = mx_mod.MXTensor
    patch_torchao_fp4_add()
    w = torch.randn(4, 16, 32, device=DEV, dtype=torch.bfloat16)
    mx = MXTensor.to_mx(w, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
    delta = torch.randn(4, 16, 32, device=DEV, dtype=torch.bfloat16) * 0.01
    out = torch.ops.aten.add.Tensor(mx, delta)
    ref = mx.dequantize(torch.bfloat16) + delta
    assert torch.allclose(out, ref, atol=2e-2)
