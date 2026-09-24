"""CUDA correctness coverage for native TorchAO NVFP4 ZeRO-3 components."""

import pytest
import torch

from tests.e2e.utils import requires_sm_ge_100


@pytest.mark.parametrize("dynamic_amax", [True, False])
@requires_sm_ge_100
def test_native_nvfp4_zero3_dynamic_forward_and_ste_gradient(dynamic_amax):
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
    )

    from axolotl.monkeypatch.torchao_deepspeed import prepare_native_nvfp4_zero3

    torch.manual_seed(0)
    device = torch.device("cuda")
    act_scale = (
        None
        if dynamic_amax
        else torch.tensor(1.0012345, device=device, dtype=torch.float32)
    )
    weight = torch.nn.Parameter(
        NVFP4Tensor.to_nvfp4(
            torch.randn(128, 128, device=device, dtype=torch.bfloat16),
            act_per_tensor_scale=act_scale,
            act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
                use_dynamic_per_tensor_scale=dynamic_amax
            ),
        ),
        requires_grad=False,
    )
    inputs = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(inputs, weight)

    model = torch.nn.Module()
    model.layer = torch.nn.Linear(128, 128, bias=False, device=device, dtype=torch.bfloat16)
    model.layer.weight = weight
    assert prepare_native_nvfp4_zero3(model, device)
    model.to(dtype=torch.bfloat16)

    actual_inputs = inputs.detach().clone().requires_grad_()
    actual = model.layer(actual_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    expected_gradient = torch.ones_like(actual).matmul(weight.dequantize())
    torch.testing.assert_close(actual_inputs.grad, expected_gradient, rtol=0, atol=0)

    if act_scale is None:
        assert model.layer._axolotl_nvfp4_act_per_tensor_scale_bytes is None
    else:
        assert torch.equal(
            model.layer._axolotl_nvfp4_act_per_tensor_scale_bytes,
            act_scale.reshape(-1).view(torch.uint8),
        )
