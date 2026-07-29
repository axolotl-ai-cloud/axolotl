"""Unit tests for the vision patch_embed Conv3d -> F.linear patch."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.models.vision_patch_embed_linear import (
    SUPPORTED_PATCH_EMBEDS,
    patch_vision_patch_embed_linear,
    unpatch_vision_patch_embed_linear,
)

# families whose patch-embed __init__ takes kwargs instead of a config object
_KWARGS_FAMILIES = {"qwen2_vl", "qwen2_5_vl", "qwen2_5_omni"}


def _resolve(model_config_type):
    module_path, class_name = SUPPORTED_PATCH_EMBEDS[model_config_type]
    module = pytest.importorskip(module_path)
    return getattr(module, class_name)


@pytest.fixture
def clean_patch_slate(model_config_type):
    cls = _resolve(model_config_type)
    saved = {
        name: cls.__dict__.get(name)
        for name in (
            "forward",
            "_axolotl_patch_embed_linear",
            "_axolotl_patch_embed_original_forward",
        )
    }
    try:
        yield cls
    finally:
        for name, value in saved.items():
            if value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)


def _make_module(model_config_type):
    cls = _resolve(model_config_type)
    torch.manual_seed(0)
    if model_config_type in _KWARGS_FAMILIES:
        return cls(patch_size=4, temporal_patch_size=2, in_channels=3, embed_dim=8)
    config = SimpleNamespace(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
    )
    return cls(config)


@pytest.mark.parametrize("model_config_type", sorted(SUPPORTED_PATCH_EMBEDS))
def test_linear_forward_matches_conv3d(model_config_type, clean_patch_slate):
    module = _make_module(model_config_type)
    x = torch.randn(16, 3 * 2 * 4 * 4)
    expected = module(x)

    patch_vision_patch_embed_linear(model_config_type)
    actual = module(x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_config_type", ["qwen3_vl"])
def test_grads_flow_to_conv_weight(model_config_type, clean_patch_slate):
    patch_vision_patch_embed_linear(model_config_type)
    module = _make_module(model_config_type)
    module(torch.randn(16, 3 * 2 * 4 * 4)).sum().backward()

    assert module.proj.weight.grad is not None
    assert module.proj.bias.grad is not None
    assert module.proj.weight.grad.shape == module.proj.weight.shape


@pytest.mark.parametrize("model_config_type", ["qwen3_vl"])
def test_patch_is_idempotent(model_config_type, clean_patch_slate):
    patch_vision_patch_embed_linear(model_config_type)
    first = clean_patch_slate.forward
    patch_vision_patch_embed_linear(model_config_type)
    assert clean_patch_slate.forward is first


@pytest.mark.parametrize("model_config_type", ["qwen3_vl"])
def test_unpatch_restores_stock_forward(model_config_type, clean_patch_slate):
    stock = clean_patch_slate.forward
    patch_vision_patch_embed_linear(model_config_type)
    assert clean_patch_slate.forward is not stock

    unpatch_vision_patch_embed_linear(model_config_type)
    assert clean_patch_slate.forward is stock
    assert not getattr(clean_patch_slate, "_axolotl_patch_embed_linear", False)

    unpatch_vision_patch_embed_linear(model_config_type)
    assert clean_patch_slate.forward is stock


@pytest.mark.parametrize("model_config_type", ["qwen3_vl"])
def test_peft_wrapped_proj_keeps_wrapper_in_path(model_config_type, clean_patch_slate):
    class FakeAdapter(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.delta = nn.Parameter(torch.tensor(0.5))

        @property
        def weight(self):
            return self.base_layer.weight

        @property
        def bias(self):
            return self.base_layer.bias

        def forward(self, x):
            return self.base_layer(x) + self.delta

    module = _make_module(model_config_type)
    x = torch.randn(16, 3 * 2 * 4 * 4)
    base_out = module(x)

    patch_vision_patch_embed_linear(model_config_type)
    module.proj = FakeAdapter(module.proj)
    wrapped_out = module(x)

    torch.testing.assert_close(wrapped_out, base_out + 0.5, rtol=1e-5, atol=1e-5)
    wrapped_out.sum().backward()
    assert module.proj.delta.grad is not None


def test_unsupported_model_type_is_a_noop():
    patch_vision_patch_embed_linear("llama")
    patch_vision_patch_embed_linear(None)
    unpatch_vision_patch_embed_linear("llama")
    unpatch_vision_patch_embed_linear(None)


def test_registry_classes_are_distinct():
    resolved = set()
    for model_config_type in SUPPORTED_PATCH_EMBEDS:
        module_path, class_name = SUPPORTED_PATCH_EMBEDS[model_config_type]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        key = f"{cls.__module__}.{cls.__qualname__}"
        assert key not in resolved
        resolved.add(key)
