# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The DeepSeek-V4 adapter refuses use_scattermoe for DeepSeek's packed-FP4 experts and warns on
block-FP8 experts, before the quantizer is installed."""

from types import SimpleNamespace

import pytest

import axolotl.integrations.kernels.adapters.dsv4 as dsv4_mod
from axolotl.integrations.kernels.adapters.dsv4 import (
    DSV4Adapter,
    check_scattermoe_expert_dtype,
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


def _cfg(monkeypatch, quantization_config, **extra):
    model_config = SimpleNamespace(model_type="deepseek_v4")
    if quantization_config is not None:
        model_config.quantization_config = quantization_config
    monkeypatch.setattr(dsv4_mod, "load_base_model_config", lambda cfg: model_config)
    return DictDefault(
        base_model="some/dsv4", use_dsv4_kernels=True, use_scattermoe=True, **extra
    )


def _record_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        dsv4_mod.LOG, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    return warnings


@pytest.mark.parametrize(
    "quantization_config", [NVFP4, None], ids=["nvfp4_experts", "unquantized"]
)
def test_supported_experts_pass(monkeypatch, quantization_config):
    assert (
        scattermoe_unsupported_expert_dtype(_cfg(monkeypatch, quantization_config))
        is None
    )


def test_deepseek_fp4_experts_are_refused(monkeypatch):
    cfg = _cfg(monkeypatch, DEEPSEEK_FP4)
    with pytest.raises(ValueError, match="expert_dtype='fp4'"):
        DSV4Adapter().pre_model_load(cfg)


@pytest.mark.parametrize(
    "grouped_mode",
    [{}, {"moe_grouped_mode": "nvfp4"}, {"dsv4_fp4_grouped_mode": "nvfp4"}],
    ids=["no_grouped_mode", "moe_grouped_mode", "dsv4_fp4_grouped_mode"],
)
def test_block_fp8_experts_warn(monkeypatch, grouped_mode):
    warnings = _record_warnings(monkeypatch)
    check_scattermoe_expert_dtype(_cfg(monkeypatch, FP8_EXPERTS, **grouped_mode))
    assert len(warnings) == 1 and "expert_dtype='fp8'" in warnings[0]
