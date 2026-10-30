# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""``experts_implementation: scattermoe`` on a fp8-quantized checkpoint: transformers'
``FP8Experts`` dispatch through their own registry, which gets a ``"scattermoe"`` entry that
picks the forward by expert format. Unquantized experts keep the generic registry.

The FP8 and scattermoe kernels need CUDA, so both forwards are replaced by recorders and only
the choice is asserted."""

import pytest
import torch
from torch import nn

pytest.importorskip("triton")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    DeepseekV4Config,
    MistralLarge4TextConfig,
    MistralLarge4TextModel,
)
from transformers.integrations import finegrained_fp8  # noqa: E402
from transformers.integrations.moe import (  # noqa: E402
    ExpertsInterface,
    use_experts_implementation,
)
from transformers.modeling_utils import PreTrainedModel  # noqa: E402
from transformers.quantizers.quantizer_finegrained_fp8 import (  # noqa: E402
    FineGrainedFP8HfQuantizer,
)
from transformers.utils.quantization_config import FineGrainedFP8Config  # noqa: E402

import axolotl.integrations.kernels.libs.scattermoe_lora.experts as experts_mod  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    experts_lora_fastpath,
    nvfp4_fsdp,
    torchao_fp4_add,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    register_scattermoe_experts,
    register_scattermoe_interface,
    register_scattermoe_model_interfaces,
)

FP8_REGISTRY = finegrained_fp8.ALL_FP8_EXPERTS_FUNCTIONS
FP8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
E, H, I = 4, 128, 128  # noqa: E741


@pytest.fixture(autouse=True)
def isolated_registration(monkeypatch):
    """Keep the registrations and global patches of ``register_scattermoe_experts`` local."""
    for cls in (ExpertsInterface, type(FP8_REGISTRY)):
        # an earlier test in the process may already have registered scattermoe
        mapping = {k: v for k, v in cls._global_mapping.items() if k != "scattermoe"}
        monkeypatch.setattr(cls, "_global_mapping", mapping)
    monkeypatch.setattr(
        PreTrainedModel,
        "get_correct_experts_implementation",
        PreTrainedModel.get_correct_experts_implementation,
    )
    monkeypatch.setattr(experts_mod, "_SCATTERMOE_PATCHED", False)
    for mod, name in (
        (experts_lora_fastpath, "patch_paramwrapper_fastpath"),
        (torchao_fp4_add, "patch_torchao_fp4_add"),
        (nvfp4_fsdp, "patch_nvfp4_fsdp"),
    ):
        monkeypatch.setattr(mod, name, lambda: None)


@pytest.fixture
def calls(monkeypatch):
    """Record which experts forward runs."""
    seen = []

    def recorder(name):
        def forward(self, hidden_states, top_k_index, top_k_weights):
            seen.append(name)
            return torch.zeros_like(hidden_states)

        return forward

    monkeypatch.setattr(
        experts_mod, "scattermoe_experts_forward", recorder("scattermoe")
    )
    for impl in ("grouped_mm", "batched_mm"):
        monkeypatch.setitem(type(FP8_REGISTRY)._global_mapping, impl, recorder(impl))
    return seen


def _dsv4(expert_dtype, quantization_config):
    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=H,
        moe_intermediate_size=I,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        n_routed_experts=E,
        num_experts_per_tok=2,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        num_nextn_predict_layers=0,
        expert_dtype=expert_dtype,
        quantization_config=quantization_config,
    )
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg), "model.layers.0.mlp.experts"


def _mistral_large4(expert_dtype, quantization_config):
    cfg = MistralLarge4TextConfig(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=I,
        moe_intermediate_size=I,
        num_hidden_layers=1,
        first_k_dense_replace=0,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=E,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        head_dim=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        quantization_config=quantization_config,
    )
    with torch.device("meta"):
        return MistralLarge4TextModel(cfg), "layers.0.mlp.experts"


ARCHS = [_dsv4, _mistral_large4]
ARCH_IDS = ["deepseek_v4", "mistral_large4"]


def _load(arch, expert_dtype="fp8", quantization_config=FP8):
    register_scattermoe_experts()
    model, path = arch(expert_dtype, quantization_config)
    if quantization_config is not None:
        quantizer = FineGrainedFP8HfQuantizer(
            FineGrainedFP8Config.from_dict(dict(quantization_config)),
            pre_quantized=True,
        )
        quantizer.preprocess_model(model, dtype=torch.bfloat16, checkpoint_files=[])
    model.set_experts_implementation("scattermoe")
    register_scattermoe_model_interfaces(model)
    experts = model.get_submodule(path)
    experts.to_empty(device="cpu")
    return experts


def _set_weights(experts, fmt):
    """Give the experts the weights each checkpoint format loads as."""
    for name in ("gate_up_proj", "down_proj"):
        shape = getattr(experts, name).shape
        if fmt == "block_fp8":
            weight = torch.randn(shape).to(torch.float8_e4m3fn)
        elif fmt == "packed_fp4":
            weight = torch.randint(-128, 127, shape, dtype=torch.int8)
        else:
            # full-width K: the fp4 allocation halves it for the packed layout
            in_features = I if name == "down_proj" else H
            weight = torch.randn(E, shape[1], in_features, dtype=torch.bfloat16)
            if fmt == "nvfp4":
                from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

                weight = NVFP4Tensor.to_nvfp4(weight)
            # the NVFP4 quantizer drops the block-scale placeholders
            experts._parameters.pop(f"{name}_scale_inv", None)
        setattr(experts, name, nn.Parameter(weight, requires_grad=False))


def _route(experts):
    hidden = torch.randn(3, H, dtype=torch.bfloat16)
    top_k_index = torch.tensor([[0, 1], [2, 3], [1, 2]])
    top_k_weights = torch.full((3, 2), 0.5)
    return experts(hidden, top_k_index, top_k_weights)


@pytest.mark.parametrize("arch", ARCHS, ids=ARCH_IDS)
@pytest.mark.parametrize(
    "expert_dtype, fmt, expected",
    [
        ("fp8", "block_fp8", "grouped_mm"),
        ("fp4", "packed_fp4", "grouped_mm"),
        ("fp4", "nvfp4", "scattermoe"),
        ("fp8", "bf16", "scattermoe"),
    ],
)
def test_fp8_experts_dispatch_by_format(calls, arch, expert_dtype, fmt, expected):
    experts = _load(arch, expert_dtype)
    assert isinstance(experts, finegrained_fp8.FP8Experts)
    assert experts.config._experts_implementation == "scattermoe"
    _set_weights(experts, fmt)
    _route(experts)
    assert calls == [expected]


@pytest.mark.parametrize("arch", ARCHS, ids=ARCH_IDS)
def test_unquantized_experts_keep_the_generic_registry(calls, arch):
    experts = _load(arch, quantization_config=None)
    assert not isinstance(experts, finegrained_fp8.FP8Experts)
    _set_weights(experts, "bf16")
    _route(experts)
    assert calls == ["scattermoe"]


@pytest.mark.parametrize("fmt", ["block_fp8", "packed_fp4"])
def test_block_quantized_experts_with_expert_lora_go_to_scattermoe(calls, fmt):
    experts = _load(_dsv4, "fp8" if fmt == "block_fp8" else "fp4")
    _set_weights(experts, fmt)
    experts._scattermoe_lora = {"gate_up_proj": object()}
    _route(experts)
    assert calls == ["scattermoe"]


@pytest.mark.parametrize("fmt", ["block_fp8", "packed_fp4"])
def test_scattermoe_refuses_block_quantized_experts(monkeypatch, fmt):
    # its torch.compile'd bincount needs a compiler; the refusal comes before the sort is used
    monkeypatch.setattr(
        experts_mod, "flatten_sort_count", lambda idx, num_experts: (None,) * 3
    )
    experts = _load(_dsv4, "fp8" if fmt == "block_fp8" else "fp4")
    _set_weights(experts, fmt)
    experts._scattermoe_lora = {"gate_up_proj": None, "down_proj": None}
    with pytest.raises(NotImplementedError, match="block-quantized"):
        _route(experts)


def test_registration_is_idempotent_and_kept():
    register_scattermoe_experts()
    dispatch = FP8_REGISTRY["scattermoe"]
    register_scattermoe_experts()
    # the expert-LoRA fast path registers lazily into the interface it finds
    register_scattermoe_interface(FP8_REGISTRY)
    assert FP8_REGISTRY["scattermoe"] is dispatch


def test_replaces_a_bare_scattermoe_entry():
    FP8_REGISTRY.register("scattermoe", experts_mod.scattermoe_experts_forward)
    register_scattermoe_experts()
    assert FP8_REGISTRY["scattermoe"]._scattermoe_fp8_dispatch


def test_registers_into_the_interface_the_experts_bind(calls):
    """A second transformers import leaves a separate ``FP8ExpertsInterface`` class."""
    twin_cls = type(
        "FP8ExpertsInterface",
        (ExpertsInterface,),
        {"_global_mapping": dict(type(FP8_REGISTRY)._global_mapping)},
    )
    twin = twin_cls()
    experts_cls = use_experts_implementation(
        type("TwinFP8Experts", (finegrained_fp8.FP8Experts,), {}),
        experts_interface=twin,
    )
    model, _ = _dsv4("fp8", FP8)
    experts = experts_cls(model.config, block_size=[128, 128])
    experts.config._experts_implementation = "scattermoe"
    _set_weights(experts, "block_fp8")
    container = nn.ModuleDict({"experts": experts})

    register_scattermoe_model_interfaces(container)
    assert "scattermoe" not in FP8_REGISTRY
    _route(experts)
    assert calls == ["grouped_mm"]
