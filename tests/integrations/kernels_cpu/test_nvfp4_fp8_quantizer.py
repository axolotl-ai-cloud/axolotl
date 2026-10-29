# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The NVFP4-aware FP8 quantizer routes DeepSeek-V4 NVFP4 expert keys through transformers'
``get_model_conversion_mapping`` into the in-place ``*_scale`` params, with no FP8 dequant, and
loads every other DeepSeek-V4 fp8 checkpoint exactly like the stock quantizer."""

import re

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("transformers.models.deepseek_v4")

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM, DeepseekV4Config  # noqa: E402
from transformers.conversion_mapping import get_model_conversion_mapping  # noqa: E402
from transformers.core_model_loading import (  # noqa: E402
    WeightConverter,
    WeightRenaming,
    rename_source_key,
)
from transformers.quantizers.quantizer_finegrained_fp8 import (  # noqa: E402
    FineGrainedFP8HfQuantizer,
)
from transformers.utils.quantization_config import FineGrainedFP8Config  # noqa: E402

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fp8_quantizer import (  # noqa: E402
    make_nvfp4_fp8_quantizer,
)

EXPERTS = "model.layers.0.mlp.experts"
CKPT_EXPERT = "layers.0.ffn.experts.3"
PROJS = ("w1", "w2", "w3")

# quantization_config + per-expert checkpoint leaves of each DeepSeek-V4 fp8 release
NVFP4_LAYOUT = (
    "fp4",
    {
        "quant_method": "fp8",
        "quant_algo": "MIXED_PRECISION",
        "moe_quant_algo": "NVFP4",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
    },
    ("weight", "weight_scale", "weight_scale_2", "input_scale"),
)
DEEPSEEK_FP4_LAYOUT = (
    "fp4",
    {"quant_method": "fp8", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"},
    ("weight", "scale"),
)
FP8_EXPERTS_LAYOUT = (
    "fp8",
    {"quant_method": "fp8", "weight_block_size": [128, 128]},
    ("weight", "scale"),
)


def _write_checkpoint(path, leaves):
    # the quantizer reads only the safetensors header, so the tensors just need the right names
    keys = [
        f"layers.{layer}.ffn.experts.{e}.{proj}.{leaf}"
        for layer in range(2)
        for e in range(4)
        for proj in PROJS
        for leaf in leaves
    ]
    save_file({k: torch.zeros(1) for k in keys}, str(path))
    return [str(path)]


def _build(quantizer_cls, layout, checkpoint_files):
    expert_dtype, quantization_config, _ = layout
    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=128,
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
        expert_dtype=expert_dtype,
        quantization_config=dict(quantization_config),
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    quantizer = quantizer_cls(
        FineGrainedFP8Config.from_dict(dict(cfg.quantization_config)),
        pre_quantized=True,
    )
    # same order and kwargs as from_pretrained: the module swap precedes the conversion mapping
    quantizer.preprocess_model(
        model, dtype=torch.bfloat16, checkpoint_files=checkpoint_files
    )
    conversions = get_model_conversion_mapping(model, hf_quantizer=quantizer)
    return model, quantizer, conversions


def _rename(model, conversions, key):
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    return rename_source_key(
        key, renamings, converters, model.base_model_prefix, model.state_dict()
    )


def _describe_conversions(conversions):
    return [
        (
            type(c).__name__,
            list(c._original_source_patterns),
            list(c._original_target_patterns),
            [type(op).__name__ for op in getattr(c, "operations", ())],
        )
        for c in conversions
    ]


def _describe_params(model):
    return {
        name: (type(p.data).__name__, p.dtype, tuple(p.shape))
        for name, p in model.named_parameters()
    }


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    ckpt = _write_checkpoint(
        tmp_path_factory.mktemp("nvfp4") / "model.safetensors", NVFP4_LAYOUT[2]
    )
    model, _, conversions = _build(make_nvfp4_fp8_quantizer(), NVFP4_LAYOUT, ckpt)
    return model, conversions


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
    model, conversions = loaded
    renamed, pattern = _rename(model, conversions, key)
    assert pattern is not None
    assert renamed == target
    assert renamed in model.state_dict()


@pytest.mark.parametrize("suffix", ["w1.weight_scale_2", "w2.input_scale"])
def test_scalar_keys_are_left_to_the_ignore_list(loaded, suffix):
    model, conversions = loaded
    renamed, pattern = _rename(model, conversions, f"{CKPT_EXPERT}.{suffix}")
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


NON_NVFP4_LAYOUTS = [FP8_EXPERTS_LAYOUT, DEEPSEEK_FP4_LAYOUT]
NON_NVFP4_IDS = ["fp8_experts", "deepseek_fp4_experts"]


def _build_stock_and_ours(layout, tmp_dir):
    ckpt = _write_checkpoint(tmp_dir / "model.safetensors", layout[2])
    return (
        _build(FineGrainedFP8HfQuantizer, layout, ckpt),
        _build(make_nvfp4_fp8_quantizer(), layout, ckpt),
    )


@pytest.fixture(scope="module", params=NON_NVFP4_LAYOUTS, ids=NON_NVFP4_IDS)
def stock_and_ours(request, tmp_path_factory):
    return _build_stock_and_ours(request.param, tmp_path_factory.mktemp("ckpt"))


def test_non_nvfp4_conversions_match_stock(stock_and_ours):
    (_, _, stock_conv), (_, _, ours_conv) = stock_and_ours
    assert _describe_conversions(ours_conv) == _describe_conversions(stock_conv)


@pytest.mark.parametrize("proj", PROJS)
@pytest.mark.parametrize("leaf", ["weight", "scale", "weight_scale_inv"])
def test_non_nvfp4_expert_keys_route_like_stock(stock_and_ours, proj, leaf):
    (stock_model, _, stock_conv), (ours_model, _, ours_conv) = stock_and_ours
    key = f"{CKPT_EXPERT}.{proj}.{leaf}"
    expected = _rename(stock_model, stock_conv, key)
    assert expected[1] is not None and expected[0] in stock_model.state_dict()
    assert _rename(ours_model, ours_conv, key) == expected


@pytest.mark.parametrize("layout", NON_NVFP4_LAYOUTS, ids=NON_NVFP4_IDS)
def test_non_nvfp4_model_matches_stock_before_and_after_load(layout, tmp_path):
    (stock_model, stock_q, _), (ours_model, ours_q, _) = _build_stock_and_ours(
        layout, tmp_path
    )
    assert _describe_params(ours_model) == _describe_params(stock_model)
    assert set(ours_model._keys_to_ignore_on_load_unexpected or ()) == set(
        stock_model._keys_to_ignore_on_load_unexpected or ()
    )
    stock_q.postprocess_model(stock_model)
    ours_q.postprocess_model(ours_model)
    assert _describe_params(ours_model) == _describe_params(stock_model)
