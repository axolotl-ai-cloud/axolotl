"""Tests for the Qwen3.5 packing monkeypatch (decoder-layer dispatch)."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers.models.qwen3_5")


@pytest.fixture
def patched_modeling():
    """Apply the packing patch and restore the stock forwards after."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    from axolotl.monkeypatch.models.qwen3_5.modeling import (
        patch_qwen3_5_modeling_packing,
    )

    saved = (
        modeling.Qwen3_5DecoderLayer.forward,
        modeling.Qwen3_5GatedDeltaNet.forward,
    )
    patch_qwen3_5_modeling_packing()
    yield modeling
    (
        modeling.Qwen3_5DecoderLayer.forward,
        modeling.Qwen3_5GatedDeltaNet.forward,
    ) = saved


class _Recorder(nn.Module):
    """Stands in for linear_attn / self_attn so dispatch is tested without FLA kernels."""

    def __init__(self, returns_tuple=False):
        super().__init__()
        self.returns_tuple = returns_tuple
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        out = torch.ones_like(kwargs["hidden_states"])
        return (out, None) if self.returns_tuple else out


def _build_layer(modeling, block_type):
    """A single decoder layer with its token mixer replaced by a recorder."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=[block_type],
    )
    torch.manual_seed(0)
    layer = modeling.Qwen3_5DecoderLayer(cfg, 0).eval()

    mixer = _Recorder(returns_tuple=block_type == "full_attention")
    attr = "linear_attn" if block_type == "linear_attention" else "self_attn"
    setattr(layer, attr, mixer)
    return layer, mixer


def _inputs(batch=2, seq_len=6, hidden_size=32):
    hidden_states = torch.randn(batch, seq_len, hidden_size)
    # MRoPE: transformers hands the decoder layer [axes, B, T] position_ids
    position_ids = torch.arange(seq_len).expand(3, batch, seq_len).contiguous()
    cos = torch.ones(batch, seq_len, 8)
    return hidden_states, position_ids, (cos, cos.clone())


def test_linear_attention_receives_position_ids(patched_modeling):
    layer, mixer = _build_layer(patched_modeling, "linear_attention")
    hidden_states, position_ids, position_embeddings = _inputs()

    out = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
    )

    assert len(mixer.calls) == 1
    assert torch.equal(mixer.calls[0]["position_ids"], position_ids)
    assert out.shape == hidden_states.shape


def test_full_attention_receives_position_embeddings(patched_modeling):
    layer, mixer = _build_layer(patched_modeling, "full_attention")
    hidden_states, position_ids, position_embeddings = _inputs()

    out = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
    )

    assert len(mixer.calls) == 1
    assert mixer.calls[0]["position_embeddings"] is position_embeddings
    assert out.shape == hidden_states.shape
