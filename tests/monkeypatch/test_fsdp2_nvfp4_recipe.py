# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""NVFP4 recipe preservation during FSDP2 shard reconstruction."""

import pytest
import torch

from axolotl.monkeypatch.accelerate.fsdp2 import _rebuild_nvfp4_like

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor


def test_rebuild_nvfp4_like_preserves_complete_native_recipe():
    qdata = torch.randint(0, 255, (2, 16, 8), dtype=torch.uint8)
    scale = torch.ones((2, 16, 1), dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.tensor([[[0.3]], [[0.7]]])
    act_per_tensor_scale = torch.tensor([[[0.125]], [[0.25]]])
    recipe = NVFP4Tensor(
        qdata,
        scale,
        16,
        torch.float32,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=False,
        use_triton_kernel=False,
        act_quant_kwargs={"block_size": 16},
    )

    rebuilt = _rebuild_nvfp4_like(
        recipe,
        recipe.qdata[:1].clone(),
        recipe.scale[:1].clone(),
        recipe.per_tensor_scale[1:].clone(),
        recipe.act_per_tensor_scale[1:].clone(),
    )

    assert rebuilt.block_size == recipe.block_size
    assert rebuilt.orig_dtype == recipe.orig_dtype
    assert torch.equal(rebuilt.per_tensor_scale, recipe.per_tensor_scale[1:])
    assert torch.equal(rebuilt.act_per_tensor_scale, recipe.act_per_tensor_scale[1:])
    assert rebuilt.is_swizzled_scales is recipe.is_swizzled_scales
    assert rebuilt.use_triton_kernel is recipe.use_triton_kernel
    assert rebuilt.act_quant_kwargs == recipe.act_quant_kwargs
