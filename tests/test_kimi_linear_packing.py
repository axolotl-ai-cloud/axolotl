"""Tests that Kimi-Linear's KDA layer isolates packed documents from position_ids."""

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from axolotl.model_support.kimi_linear import modeling_kimi  # noqa: E402
from axolotl.model_support.kimi_linear.configuration_kimi import (  # noqa: E402
    KimiLinearConfig,
)

HIDDEN, HEADS, HEAD_DIM = 16, 2, 4
# training asserts the chunk kernel, which Kimi only picks above 64 tokens
SEQ = 96


class _Conv(nn.Module):
    """Stands in for fla's ShortConvolution and records the cu_seqlens it was given."""

    def __init__(self):
        super().__init__()
        self.cu_seqlens = []

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        self.cu_seqlens.append(cu_seqlens)
        return x, None


class _Norm(nn.Module):
    def forward(self, o, g):
        return o


@pytest.fixture
def layer(monkeypatch):
    config = KimiLinearConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )
    torch.manual_seed(0)
    kda = modeling_kimi.KimiDeltaAttention(config, layer_idx=0)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(kda, name, _Conv())
    kda.o_norm = _Norm()

    seen = {}

    def fake_chunk_kda(q, k, v, g, beta, cu_seqlens=None, **_):
        seen["cu_seqlens"] = cu_seqlens
        seen["batch"] = v.shape[0]
        return v, None

    monkeypatch.setattr(modeling_kimi, "chunk_kda", fake_chunk_kda)
    monkeypatch.setattr(modeling_kimi, "fused_kda_gate", lambda g, A_log, dt_bias: g)
    kda.train()
    return kda, seen


def test_packed_row_yields_document_boundaries(layer):
    kda, seen = layer
    hidden = torch.randn(1, SEQ, HIDDEN)
    position_ids = torch.cat([torch.arange(64), torch.arange(SEQ - 64)])[None]

    out = kda(hidden, position_ids=position_ids)

    assert seen["cu_seqlens"].tolist() == [0, 64, SEQ]
    assert kda.q_conv1d.cu_seqlens[0].tolist() == [0, 64, SEQ]
    assert out.shape == hidden.shape


def test_packed_batch_is_flattened_and_restored(layer):
    kda, seen = layer
    hidden = torch.randn(2, SEQ, HIDDEN)
    row0 = torch.cat([torch.arange(80), torch.arange(SEQ - 80)])
    row1 = torch.cat([torch.arange(32), torch.arange(SEQ - 32)])

    out = kda(hidden, position_ids=torch.stack([row0, row1]))

    assert seen["batch"] == 1
    assert seen["cu_seqlens"].tolist() == [0, 80, SEQ, SEQ + 32, 2 * SEQ]
    assert out.shape == hidden.shape


def test_unpacked_input_runs_dense(layer):
    kda, seen = layer

    kda(torch.randn(2, SEQ, HIDDEN), position_ids=torch.arange(SEQ).expand(2, SEQ))

    assert seen["cu_seqlens"] is None
    assert seen["batch"] == 2


def test_multipack_attention_mask_still_splits_documents(layer):
    kda, seen = layer
    mask = torch.tensor([[1] * 50 + [2] * 40 + [0] * (SEQ - 90)])

    out = kda(torch.randn(1, SEQ, HIDDEN), attention_mask=mask)

    assert seen["cu_seqlens"].tolist() == [0, 50, 90]
    assert out.shape == (1, SEQ, HIDDEN)


def test_decoder_layer_hands_position_ids_to_the_kda_branch():
    calls = []

    class _Recorder(nn.Module):
        def forward(self, hidden_states, **kwargs):
            calls.append(kwargs)
            return hidden_states

    decoder = nn.Module.__new__(modeling_kimi.KimiDecoderLayer)
    nn.Module.__init__(decoder)
    decoder.is_linear_attn = True
    decoder.self_attn = _Recorder()
    decoder.input_layernorm = nn.Identity()
    decoder.post_attention_layernorm = nn.Identity()
    decoder.mlp = nn.Identity()

    position_ids = torch.tensor([[0, 1, 0, 1]])
    decoder(torch.randn(1, 4, HIDDEN), position_ids=position_ids)

    assert calls[0]["position_ids"] is position_ids


def test_model_hands_position_ids_to_every_decoder_layer():
    config = KimiLinearConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )
    model = modeling_kimi.KimiLinearModel(config).eval()
    calls = []

    class _Recorder(nn.Module):
        is_linear_attn = True

        def forward(self, hidden_states, **kwargs):
            calls.append(kwargs)
            return hidden_states

    model.layers[0] = _Recorder()
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    model(
        input_ids=torch.randint(0, 64, (1, 5)),
        position_ids=position_ids,
        use_cache=False,
    )

    assert calls[0]["position_ids"] is position_ids
