"""Model-support providers for LoRA kernel attention classes and layer discovery."""

from types import SimpleNamespace

import pytest
from peft import PeftModel
from torch import nn

from axolotl.model_support import (
    ModelStrategies,
    ModelStrategyOverrides,
    get_model_support_for_cfg,
)
from axolotl.monkeypatch import lora_kernels
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

NEMOTRON = "nvidia/Nemotron-Labs-Diffusion-3B"


class _Attention(nn.Module):
    pass


def test_strategy_provider_supplies_the_attention_class(monkeypatch):
    seen = []

    def provider(cfg):
        seen.append(cfg.base_model)
        return _Attention

    strategies = ModelStrategies(lora_attention_cls=provider)
    monkeypatch.setattr(
        lora_kernels, "AutoConfig", SimpleNamespace(from_pretrained=None)
    )
    monkeypatch.setattr(
        "axolotl.model_support.get_model_support", lambda model_type: object()
    )
    monkeypatch.setattr(
        "axolotl.model_support.resolve_model_support",
        lambda support: SimpleNamespace(strategies=strategies),
    )

    cfg = DictDefault(base_model="remote/model", model_config_type="remote")

    assert lora_kernels.get_attention_cls_from_config(cfg) is _Attention
    assert seen == ["remote/model"]


def test_strategy_overrides_inherit_and_replace_the_provider():
    base = ModelStrategies(lora_attention_cls=lambda cfg: _Attention)

    assert base.with_overrides(ModelStrategyOverrides()).lora_attention_cls is not None
    assert (
        base.with_overrides(
            ModelStrategyOverrides(lora_attention_cls=None)
        ).lora_attention_cls
        is None
    )


def test_get_layers_falls_back_to_an_encoder_stack():
    layers = nn.ModuleList([nn.Linear(2, 2)])
    pretrained = SimpleNamespace(encoder=SimpleNamespace(layers=layers))

    assert lora_kernels.get_layers(SimpleNamespace(model=pretrained)) is layers


def test_apply_lora_kernel_patches_requires_a_peft_model():
    with pytest.raises(TypeError, match="PeftModel"):
        lora_kernels.apply_lora_kernel_patches(nn.Linear(2, 2), DictDefault())
    assert PeftModel is not None


def test_nemotron_descriptor_matches_before_the_model_type_is_known():
    support = get_model_support_for_cfg(DictDefault(base_model=NEMOTRON))

    assert support is not None
    assert "nemotron_labs_diffusion" in support.model_types
    assert get_model_support_for_cfg(DictDefault(base_model="dummy_model")) is None


def _kernel_cfg(base_model: str) -> DictDefault:
    return DictDefault(
        {
            "base_model": base_model,
            "trust_remote_code": True,
            "adapter": "lora",
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
            "lora_mlp_kernel": True,
            "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
        }
    )


def test_remote_code_kernels_allowed_only_with_a_model_support_provider():
    validated = validate_config(_kernel_cfg(NEMOTRON))

    assert validated.lora_qkv_kernel is True
    with pytest.raises(ValueError, match="not compatible with trust_remote_code"):
        validate_config(_kernel_cfg("dummy_model"))
