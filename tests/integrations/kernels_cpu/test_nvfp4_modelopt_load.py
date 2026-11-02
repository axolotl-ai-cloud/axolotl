# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""A modelopt NVFP4 MoE checkpoint loads bit-exactly through transformers' conversion loader.

``quant_method: modelopt`` has no transformers quantizer, so the loader would cast every source
to the bf16 skeleton dtype before our converters run, rounding the fp32 ``weight_scale_2`` that
the experts' ``per_tensor_scale``, the dequantized non-routed linears and ``_nvfp4_pts`` use.
"""

import json
import re

import pytest
import torch

pytest.importorskip("triton")
NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor"
).NVFP4Tensor
pytest.importorskip("transformers.models.beam")

import transformers.conversion_mapping as conversion_mapping  # noqa: E402
import transformers.core_model_loading as cml  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import BeamConfig, BeamForCausalLM  # noqa: E402

from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    nvfp4_weight_converter,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (  # noqa: E402
    fuse_nvfp4_experts,
    inspect_nvfp4_layout,
)

NUM_EXPERTS = 8
EXPERTS = "model.layers.1.mlp.experts"
QUANTIZED_LINEAR = re.compile(
    r"\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
)
# unquantized layer under a converter suffix: the converter passes it through
PASSTHROUGH = "model.layers.1.self_attn.o_proj.weight"


def _nvfp4(gen, n, k, pts):
    return {
        "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen),
        "weight_scale": (torch.rand(n, k // 16, generator=gen) * 4 + 0.25).to(
            torch.float8_e4m3fn
        ),
        "weight_scale_2": torch.tensor(pts, dtype=torch.float32),
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = BeamConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        mlp_layer_types=["dense", "sparse"],
    )
    with torch.device("meta"):
        skeleton = BeamForCausalLM(cfg)

    gen = torch.Generator().manual_seed(0)
    sd = {}
    for i, (name, param) in enumerate(skeleton.state_dict().items()):
        if name == f"{EXPERTS}.gate_up_proj":
            n, k = param.shape[1] // 2, param.shape[2]
            for e in range(NUM_EXPERTS):
                # gate and up differ so the fused max-and-fold path runs
                for proj, pts in (("gate_proj", 0.01), ("up_proj", 0.005)):
                    for leaf, t in _nvfp4(gen, n, k, pts * (e + 1)).items():
                        sd[f"{EXPERTS}.{e}.{proj}.{leaf}"] = t
        elif name == f"{EXPERTS}.down_proj":
            n, k = param.shape[1], param.shape[2]
            for e in range(NUM_EXPERTS):
                for leaf, t in _nvfp4(gen, n, k, 0.02 * (e + 1)).items():
                    sd[f"{EXPERTS}.{e}.down_proj.{leaf}"] = t
        elif QUANTIZED_LINEAR.search(name) and name != PASSTHROUGH:
            base = name[: -len(".weight")]
            for leaf, t in _nvfp4(gen, *param.shape, 0.03 + 0.001 * i).items():
                sd[f"{base}.{leaf}"] = t
            sd[f"{base}.input_scale"] = torch.tensor(1.0)
        else:
            sd[name] = torch.randn(param.shape, generator=gen)

    path = tmp_path_factory.mktemp("beam_modelopt_nvfp4")
    save_file(sd, str(path / "model.safetensors"))
    config = cfg.to_dict()
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "ignore": ["lm_head", "model.layers.*.mlp.gate"],
    }
    (path / "config.json").write_text(json.dumps(config))
    return path, sd


@pytest.fixture
def isolated_loader(monkeypatch):
    """Undo the conversion registration and loader patches after each test."""
    conversion_mapping.get_checkpoint_conversion_mapping("beam")
    monkeypatch.setattr(
        conversion_mapping,
        "_checkpoint_conversion_mapping_cache",
        dict(conversion_mapping._checkpoint_conversion_mapping_cache),
    )
    monkeypatch.setattr(
        conversion_mapping,
        "USER_REGISTERED_MAPPINGS",
        set(conversion_mapping.USER_REGISTERED_MAPPINGS),
    )
    for fn in ("rename_source_key", "spawn_materialize", "_materialize_copy"):
        monkeypatch.setattr(cml, fn, getattr(cml, fn))
    monkeypatch.setattr(
        nvfp4_weight_converter,
        "_RANK0_ONLY_ACTIVE",
        nvfp4_weight_converter._RANK0_ONLY_ACTIVE,
    )
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def _expert_parts(sd, proj):
    leaves = {"qd": "weight", "sc": "weight_scale", "pts": "weight_scale_2"}
    return {
        key: [sd[f"{EXPERTS}.{e}.{proj}.{leaf}"] for e in range(NUM_EXPERTS)]
        for key, leaf in leaves.items()
    }


@pytest.mark.parametrize("async_load", [False, True], ids=["sync", "threaded"])
@pytest.mark.parametrize("rank0_only", [False, True], ids=["plain", "rank0_only"])
def test_modelopt_nvfp4_scales_load_exactly(
    checkpoint, isolated_loader, monkeypatch, rank0_only, async_load
):
    path, sd = checkpoint
    scales_2 = torch.stack([t for k, t in sd.items() if k.endswith("weight_scale_2")])
    # a scale that survives the bf16 round trip could not catch the cast
    assert (scales_2.to(torch.bfloat16).float() != scales_2).all()

    if async_load:
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    else:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    if rank0_only:
        nvfp4_weight_converter.patch_conversion_loader_rank0_only()
    nvfp4_weight_converter.register_nvfp4_converters_for_layout(
        "beam", inspect_nvfp4_layout(str(path))
    )

    model, info = BeamForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, output_loading_info=True
    )
    assert not info["missing_keys"]

    experts = model.get_submodule(EXPERTS)
    for attr, projs in (
        ("gate_up_proj", ("gate_proj", "up_proj")),
        ("down_proj", ("down_proj",)),
    ):
        parts = [_expert_parts(sd, proj) for proj in projs]
        pts = torch.stack(parts[0]["pts"])
        for part in parts[1:]:
            pts = torch.maximum(pts, torch.stack(part["pts"]))
        got = getattr(experts, attr)
        want = fuse_nvfp4_experts(parts)

        assert isinstance(got, NVFP4Tensor)
        assert got.per_tensor_scale.dtype == torch.float32
        assert torch.equal(got.per_tensor_scale.flatten(), pts)
        assert torch.equal(got.qdata, want.qdata)
        assert torch.equal(got.scale.view(torch.uint8), want.scale.view(torch.uint8))

    nonrouted = [
        k[: -len(".weight_scale_2")]
        for k in sd
        if k.endswith(".weight_scale_2") and not k.startswith(EXPERTS)
    ]
    assert len(nonrouted) == 10
    for base in nonrouted:
        pts = sd[f"{base}.weight_scale_2"]
        want = NVFP4Tensor(
            sd[f"{base}.weight"],
            sd[f"{base}.weight_scale"],
            16,
            torch.bfloat16,
            per_tensor_scale=pts.view(1, 1, 1),
        ).dequantize(torch.bfloat16)
        module = model.get_submodule(base)
        assert torch.equal(module.weight, want), base
        assert module._nvfp4_pts.dtype == torch.float32
        assert torch.equal(module._nvfp4_pts, pts), base

    # fp32 checkpoint params keep the dtype the loader gave them before
    for name, dtype in (
        (PASSTHROUGH, torch.bfloat16),
        ("model.embed_tokens.weight", torch.bfloat16),
        ("model.layers.0.input_layernorm.weight", torch.bfloat16),
        ("model.layers.1.mlp.gate.weight", torch.float32),
    ):
        param = model.get_parameter(name)
        assert param.dtype == dtype, name
        assert torch.equal(param, sd[name].to(dtype)), name
