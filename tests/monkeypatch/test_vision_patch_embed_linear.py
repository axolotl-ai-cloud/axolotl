"""Unit tests for the vision patch_embed Conv3d -> F.linear patch."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from axolotl.monkeypatch.models.vision_patch_embed_linear import (
    SUPPORTED_PATCH_EMBEDS,
    patch_vision_patch_embed_linear,
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
    original_forward = cls.forward
    had_sentinel = getattr(cls, "_axolotl_patch_embed_linear", False)
    try:
        yield cls
    finally:
        cls.forward = original_forward
        if had_sentinel:
            cls._axolotl_patch_embed_linear = True
        elif hasattr(cls, "_axolotl_patch_embed_linear"):
            del cls._axolotl_patch_embed_linear


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


def test_unsupported_model_type_is_a_noop():
    patch_vision_patch_embed_linear("llama")
    patch_vision_patch_embed_linear(None)


def test_registry_classes_are_distinct():
    resolved = set()
    for model_config_type in SUPPORTED_PATCH_EMBEDS:
        module_path, class_name = SUPPORTED_PATCH_EMBEDS[model_config_type]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        key = f"{cls.__module__}.{cls.__qualname__}"
        assert key not in resolved
        resolved.add(key)
