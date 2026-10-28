# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The NVFP4-aware FP8 quantizer routes DeepSeek-V4 NVFP4 expert keys through transformers'
``get_model_conversion_mapping`` into the in-place ``*_scale`` params, with no FP8 dequant."""

import re

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("transformers.models.deepseek_v4")

from transformers import AutoModelForCausalLM, DeepseekV4Config  # noqa: E402
from transformers.conversion_mapping import get_model_conversion_mapping  # noqa: E402
from transformers.core_model_loading import (  # noqa: E402
    WeightConverter,
    WeightRenaming,
    rename_source_key,
)
from transformers.utils.quantization_config import FineGrainedFP8Config  # noqa: E402

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fp8_quantizer import (  # noqa: E402
    make_nvfp4_fp8_quantizer,
)

EXPERTS = "model.layers.0.mlp.experts"
CKPT_EXPERT = "layers.0.ffn.experts.3"


@pytest.fixture(scope="module")
def loaded():
    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        num_nextn_predict_layers=0,
        expert_dtype="fp4",
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    quantizer = make_nvfp4_fp8_quantizer()(
        FineGrainedFP8Config.from_dict(dict(cfg.quantization_config)),
        pre_quantized=True,
    )
    # same order as from_pretrained: the module swap precedes the conversion mapping
    quantizer.preprocess_model(model, dtype=torch.bfloat16)
    conversions = get_model_conversion_mapping(model, hf_quantizer=quantizer)
    return model, conversions


def _rename(loaded, key):
    model, conversions = loaded
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    return rename_source_key(
        key, renamings, converters, model.base_model_prefix, model.state_dict()
    )


@pytest.mark.parametrize(
    "key, target",
    [
        (f"{CKPT_EXPERT}.w1.weight", f"{EXPERTS}.gate_up_proj"),
        (f"{CKPT_EXPERT}.w3.weight", f"{EXPERTS}.gate_up_proj"),
        (f"{CKPT_EXPERT}.w1.weight_scale", f"{EXPERTS}.gate_up_proj_scale"),
        (f"{CKPT_EXPERT}.w3.weight_scale", f"{EXPERTS}.gate_up_proj_scale"),
        (f"{CKPT_EXPERT}.w2.weight", f"{EXPERTS}.down_proj"),
        (f"{CKPT_EXPERT}.w2.weight_scale", f"{EXPERTS}.down_proj_scale"),
    ],
)
def test_expert_keys_fuse_into_nvfp4_params(loaded, key, target):
    model, _ = loaded
    renamed, pattern = _rename(loaded, key)
    assert pattern is not None
    assert renamed == target
    assert renamed in model.state_dict()


@pytest.mark.parametrize("suffix", ["w1.weight_scale_2", "w2.input_scale"])
def test_scalar_keys_are_left_to_the_ignore_list(loaded, suffix):
    model, _ = loaded
    renamed, pattern = _rename(loaded, f"{CKPT_EXPERT}.{suffix}")
    assert pattern is None
    assert any(re.search(p, renamed) for p in model._keys_to_ignore_on_load_unexpected)


def test_expert_converters_skip_fp8_dequantize(loaded):
    _, conversions = loaded
    experts = [
        c
        for c in conversions
        if isinstance(c, WeightConverter)
        and any("experts" in t for t in c._original_target_patterns)
    ]
    assert len(experts) == 4
    for conv in experts:
        assert not any(
            getattr(op, "hf_quantizer", None) is not None for op in conv.operations
        )
        assert not any(
            "scale_inv" in p or "activation_scale" in p
            for p in conv._original_source_patterns
        )
