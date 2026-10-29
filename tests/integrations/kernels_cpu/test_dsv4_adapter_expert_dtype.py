# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The DeepSeek-V4 adapter refuses use_scattermoe for fp8 checkpoints whose routed experts are
not NVFP4, before the NVFP4-aware quantizer is installed."""

from types import SimpleNamespace

import pytest

import axolotl.integrations.kernels.adapters.dsv4 as dsv4_mod
from axolotl.integrations.kernels.adapters.dsv4 import (
    DSV4Adapter,
    scattermoe_unsupported_expert_dtype,
)
from axolotl.utils.dict import DictDefault

NVFP4 = {
    "quant_method": "fp8",
    "quant_algo": "MIXED_PRECISION",
    "moe_quant_algo": "NVFP4",
    "expert_dtype": "fp4",
}
DEEPSEEK_FP4 = {"quant_method": "fp8", "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
FP8_EXPERTS = {"quant_method": "fp8", "weight_block_size": [128, 128]}


def _cfg(monkeypatch, quantization_config):
    model_config = SimpleNamespace(model_type="deepseek_v4")
    if quantization_config is not None:
        model_config.quantization_config = quantization_config
    monkeypatch.setattr(dsv4_mod, "load_base_model_config", lambda cfg: model_config)
    return DictDefault(
        base_model="some/dsv4", use_dsv4_kernels=True, use_scattermoe=True
    )


@pytest.mark.parametrize(
    "quantization_config", [NVFP4, None], ids=["nvfp4_experts", "unquantized"]
)
def test_supported_experts_pass(monkeypatch, quantization_config):
    assert (
        scattermoe_unsupported_expert_dtype(_cfg(monkeypatch, quantization_config))
        is None
    )


@pytest.mark.parametrize(
    "quantization_config, expert_dtype",
    [(DEEPSEEK_FP4, "fp4"), (FP8_EXPERTS, "fp8")],
    ids=["deepseek_fp4_experts", "fp8_experts"],
)
def test_blockwise_experts_are_refused(monkeypatch, quantization_config, expert_dtype):
    cfg = _cfg(monkeypatch, quantization_config)
    with pytest.raises(ValueError, match=f"expert_dtype='{expert_dtype}'"):
        DSV4Adapter().pre_model_load(cfg)
