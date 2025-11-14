"""Tests for attention instrumentation helpers."""

import torch
import torch.nn as nn

import torch
import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)

from axolotl.muonclip import (
    MuonClipController,
    ensure_llama_attention_instrumentation,
    ensure_qwen_attention_instrumentation,
)
from axolotl.muonclip.attention import (
    BUFFER_NAME,
    auto_register_llama_attention,
    register_attention_module,
    record_attention_logits,
)
from axolotl.utils.schemas.muon import MuonClipConfig


class _ToyAttention(nn.Module):
    def __init__(self, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(4, 4)

    def forward(self, logits):
        record_attention_logits(self, logits)
        return logits


def test_register_attention_module_adds_buffer():
    attn = _ToyAttention()

    tracker = register_attention_module(attn, name="toy", num_heads=attn.num_heads)

    assert tracker.name == "toy"
    assert hasattr(attn, BUFFER_NAME)
    assert getattr(attn, BUFFER_NAME).shape[0] == attn.num_heads


def test_record_attention_logits_updates_buffer():
    attn = _ToyAttention()
    register_attention_module(attn, name="toy", num_heads=attn.num_heads)

    logits = torch.tensor(
        [
            [  # batch 0
                [[1.0, 2.0], [3.0, 4.0]],  # head 0
                [[0.5, 0.1], [0.2, 0.3]],  # head 1
            ]
        ]
    )
    attn(logits)
    buffer = getattr(attn, BUFFER_NAME)
    assert torch.allclose(buffer, torch.tensor([4.0, 0.5]))

    logits2 = torch.tensor(
        [
            [
                [[5.0, 1.0], [0.0, 2.0]],
                [[0.4, 0.6], [0.8, 0.7]],
            ]
        ]
    )
    attn(logits2)
    buffer = getattr(attn, BUFFER_NAME)
    assert torch.allclose(buffer, torch.tensor([5.0, 0.8]))


def test_record_attention_logits_accepts_vector():
    attn = _ToyAttention()
    register_attention_module(attn, name="toy", num_heads=attn.num_heads)

    head_max = torch.tensor([1.0, 2.5])
    record_attention_logits(attn, head_max)
    buffer = getattr(attn, BUFFER_NAME)
    assert torch.allclose(buffer, head_max)


def test_record_attention_logits_accepts_batch_head_matrix():
    attn = _ToyAttention()
    register_attention_module(attn, name="toy", num_heads=attn.num_heads)

    batch_head = torch.tensor([[1.0, 0.5], [0.1, 3.2]])
    record_attention_logits(attn, batch_head)
    buffer = getattr(attn, BUFFER_NAME)
    assert torch.allclose(buffer, torch.tensor([1.0, 3.2]))


class LlamaSdpaAttention(nn.Module):
    def __init__(self, heads=2):
        super().__init__()
        self.num_heads = heads
        self.weight = nn.Parameter(torch.ones(1))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = LlamaSdpaAttention()


def test_auto_register_llama_attention_tracks_modules():
    model = _ToyModel()
    controller = MuonClipController(model, MuonClipConfig(enabled=True))

    count = auto_register_llama_attention(model, controller)

    assert count == 1
    assert hasattr(model.attn, BUFFER_NAME)


def test_llama_attention_patch_installs():
    ensure_llama_attention_instrumentation()
    from transformers.models.llama.modeling_llama import LlamaAttention

    assert getattr(LlamaAttention, "_muonclip_llama_patched", False)


def test_qwen_attention_patch_installs():
    ensure_qwen_attention_instrumentation()
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    assert getattr(Qwen3Attention, "_muonclip_qwen_patched", False)


def test_llama_qk_clip_handles_grouped_query_attention():
    ensure_llama_attention_instrumentation()
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    attn = LlamaAttention(config, layer_idx=0)
    register_attention_module(attn, name="llama", num_heads=config.num_attention_heads)

    rotary = LlamaRotaryEmbedding(config)
    seq_len = 4
    hidden_states = torch.arange(1 * seq_len * config.hidden_size, dtype=torch.float32).reshape(
        1, seq_len, config.hidden_size
    )
    position_ids = torch.zeros(1, seq_len, dtype=torch.long)
    position_embeddings = rotary(hidden_states, position_ids)

    attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
    )
    buffer = getattr(attn, BUFFER_NAME)
    assert buffer.numel() == config.num_attention_heads
    assert torch.all(torch.isfinite(buffer))


def test_qwen_qk_clip_handles_grouped_query_attention():
    ensure_qwen_attention_instrumentation()
    config = Qwen3Config(
        hidden_size=48,
        intermediate_size=64,
        num_attention_heads=6,
        num_key_value_heads=3,
        num_hidden_layers=1,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    attn = Qwen3Attention(config, layer_idx=0)
    register_attention_module(attn, name="qwen3", num_heads=config.num_attention_heads)

    rotary = Qwen3RotaryEmbedding(config)
    seq_len = 4
    hidden_states = torch.arange(1 * seq_len * config.hidden_size, dtype=torch.float32).reshape(
        1, seq_len, config.hidden_size
    )
    position_ids = torch.zeros(1, seq_len, dtype=torch.long)
    position_embeddings = rotary(hidden_states, position_ids)

    attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
    )
    buffer = getattr(attn, BUFFER_NAME)
    assert buffer.numel() == config.num_attention_heads
    assert torch.all(torch.isfinite(buffer))
