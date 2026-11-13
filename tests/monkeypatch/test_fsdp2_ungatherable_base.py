"""PEFT saves under FSDP2 full state dicts skip the base when it cannot be all-gathered."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import _model_needs_ownership


class FakeNVFP4Tensor(torch.Tensor):
    pass


FakeNVFP4Tensor.__module__ = "torchao.prototype.mx_formats.nvfp4_tensor"


def _peft_model():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    return get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"]))


def test_a_torchao_base_weight_saves_the_adapter_only():
    model = _peft_model()
    assert not _model_needs_ownership(model)
    mlp = model.base_model.model.model.layers[0].mlp
    weight = torch.Tensor._make_subclass(FakeNVFP4Tensor, mlp.down_proj.weight.data)
    mlp.down_proj._parameters["weight"] = torch.nn.Parameter(
        weight, requires_grad=False
    )
    assert _model_needs_ownership(model)
    assert not _model_needs_ownership(model.base_model.model)
