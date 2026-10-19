# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-layer branch selection in the non-routed linear dequant converter.

modelopt MIXED_PRECISION checkpoints quantize per layer, not per module name, while
converters are registered per suffix.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("triton")

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (  # noqa: E402
    Nvfp4LinearDequantize,
)


def _convert(input_dict, out_features=8, in_features=4):
    model = nn.Module()
    model.lin = nn.Linear(in_features, out_features, bias=False)
    Nvfp4LinearDequantize().convert(
        input_dict, full_layer_name="lin.weight", model=model, missing_keys=set()
    )
    return model.lin


def _to_fp8(w, scale):
    return (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def test_nvfp4_layer_dequantizes_and_keeps_pts():
    torchao = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")

    w = torch.randn(8, 64, dtype=torch.bfloat16)
    pts = (w.abs().amax().float() / (6.0 * 448.0)).reshape(1, 1, 1)
    ref = torchao.NVFP4Tensor.to_nvfp4(w, 16, per_tensor_scale=pts)

    # the loader casts sources to the skeleton dtype; qdata/scale must survive the round trip
    lin = _convert(
        {
            "lin.weight_scale_2": [pts.reshape(())],
            "lin.weight_scale": [ref.scale.to(torch.bfloat16)],
            "lin.weight": [ref.qdata.to(torch.bfloat16)],
        },
        in_features=64,
    )

    assert tuple(lin.weight.shape) == (8, 64)
    assert torch.equal(lin.weight.data, ref.dequantize(torch.bfloat16))
    assert torch.equal(lin._nvfp4_pts, pts.reshape(()))


def test_static_fp8_per_tensor_scale():
    w = torch.randn(8, 4)
    scale = (w.abs().amax() / 448.0).clamp(min=1e-8)
    w8 = _to_fp8(w, scale)

    lin = _convert(
        {
            "lin.weight": [w8],
            "lin.weight_scale": [scale.reshape(())],
            "lin.input_scale": [torch.tensor(1.0)],
        }
    )

    assert lin.weight.dtype == torch.bfloat16
    assert torch.equal(lin.weight.data, (w8.float() * scale).to(torch.bfloat16))
    assert not hasattr(lin, "_nvfp4_pts")


def test_static_fp8_per_row_scale():
    w = torch.randn(8, 4)
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-8)
    w8 = _to_fp8(w, scale)

    lin = _convert({"lin.weight": [w8], "lin.weight_scale": [scale.reshape(-1)]})

    assert torch.equal(lin.weight.data, (w8.float() * scale).to(torch.bfloat16))


def test_unquantized_layer_passes_through():
    """A suffix quantized in SOME layers still claims the bf16 ones; they must load as-is."""
    w = torch.randn(8, 4, dtype=torch.bfloat16)

    assert torch.equal(_convert({"lin.weight": [w]}).weight.data, w)


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(_to_fp8(torch.randn(8, 4), torch.tensor(0.01)), id="fp8"),
        # packed [out, in/2] qdata, already cast to the skeleton dtype: only shape gives it away
        pytest.param(torch.randn(8, 2, dtype=torch.bfloat16), id="nvfp4_qdata"),
    ],
)
def test_quantized_weight_without_scale_raises(weight):
    """Never load a scaleless quantized weight: it would train on unscaled values."""
    with pytest.raises(KeyError, match="unscaled"):
        _convert({"lin.weight": [weight]})
