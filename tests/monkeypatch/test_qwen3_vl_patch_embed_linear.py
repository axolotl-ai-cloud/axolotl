"""Unit tests for the Qwen3-VL patch_embed Conv3d -> F.linear patch."""

from types import SimpleNamespace

import pytest
import torch

qwen3_vl_modeling = pytest.importorskip(
    "transformers.models.qwen3_vl.modeling_qwen3_vl"
)


@pytest.fixture
def clean_patch_embed_slate():
    cls = qwen3_vl_modeling.Qwen3VLVisionPatchEmbed
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


def _make_module():
    config = SimpleNamespace(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
    )
    torch.manual_seed(0)
    return qwen3_vl_modeling.Qwen3VLVisionPatchEmbed(config)


def test_linear_forward_matches_conv3d(clean_patch_embed_slate):
    from axolotl.monkeypatch.models.qwen3_vl.patch_embed_linear import (
        patch_qwen3_vl_patch_embed_linear,
    )

    module = _make_module()
    x = torch.randn(16, 3 * 2 * 4 * 4)
    expected = module(x)

    patch_qwen3_vl_patch_embed_linear()
    actual = module(x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_grads_flow_to_conv_weight(clean_patch_embed_slate):
    from axolotl.monkeypatch.models.qwen3_vl.patch_embed_linear import (
        patch_qwen3_vl_patch_embed_linear,
    )

    patch_qwen3_vl_patch_embed_linear()
    module = _make_module()
    module(torch.randn(16, 3 * 2 * 4 * 4)).sum().backward()

    assert module.proj.weight.grad is not None
    assert module.proj.bias.grad is not None
    assert module.proj.weight.grad.shape == module.proj.weight.shape


def test_patch_is_idempotent(clean_patch_embed_slate):
    from axolotl.monkeypatch.models.qwen3_vl.patch_embed_linear import (
        patch_qwen3_vl_patch_embed_linear,
    )

    patch_qwen3_vl_patch_embed_linear()
    first = clean_patch_embed_slate.forward
    patch_qwen3_vl_patch_embed_linear()
    assert clean_patch_embed_slate.forward is first
