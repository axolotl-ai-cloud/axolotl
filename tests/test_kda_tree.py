"""Kimi Delta Attention on tree rows: every sequence's output and gradients match
the sequence run alone, forking the KDA state or rescanning shared paths."""

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("fla.ops.kda", reason="the KDA layers import fla-core")

from fla.ops.kda.gate import naive_kda_gate  # noqa: E402
from fla.ops.kda.naive import naive_recurrent_kda  # noqa: E402
from tree_recurrence import (  # noqa: E402
    Policy,
    driver as tree_driver,  # noqa: E402
    follows_trees,
    tree_plan,
)

from axolotl.core.trainers.async_grpo.scoring import pack_sequences  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, WIDTH = 32, 4, 8, 4


class ReferenceShortConvolution(nn.Conv1d):
    """fla's ShortConvolution (silu) as a per-sequence depthwise conv."""

    def __init__(self, hidden_size):
        super().__init__(
            hidden_size, hidden_size, WIDTH, groups=hidden_size, bias=False
        )

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            seq = nn.functional.pad(x[:, start:end].transpose(1, 2), (WIDTH - 1, 0))
            out = nn.functional.conv1d(seq, self.weight, groups=self.groups)
            outs.append(nn.functional.silu(out).transpose(1, 2))
        return torch.cat(outs, dim=1), None


class ReferenceNormGated(nn.Module):
    """fla's FusedRMSNormGated with a sigmoid gate."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(HEAD_DIM))

    def forward(self, x, g):
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x * self.weight * torch.sigmoid(g.float())).to(g.dtype)


def reference_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
):
    """fla's varlen chunk_kda, sequence by sequence through its naive recurrence."""
    if use_qk_l2norm_in_kernel:
        q, k = (nn.functional.normalize(x.float(), dim=-1, eps=1e-6) for x in (q, k))
    bounds = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outs, finals = [], []
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        out, final = naive_recurrent_kda(
            *(x[:, start:end] for x in (q, k, v, g, beta)),
            initial_state=None if initial_state is None else initial_state[i : i + 1],
            output_final_state=True,
        )
        outs.append(out)
        finals.append(final)
    return torch.cat(outs, dim=1), torch.cat(finals)


def _kimi(monkeypatch):
    from axolotl.model_support.kimi_linear import modeling_kimi
    from axolotl.model_support.kimi_linear.configuration_kimi import (
        KimiLinearConfig,
    )

    monkeypatch.setattr(modeling_kimi, "chunk_kda", reference_chunk_kda)
    monkeypatch.setattr(modeling_kimi, "fused_kda_gate", naive_kda_gate)
    config = KimiLinearConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        linear_attn_config={
            "short_conv_kernel_size": WIDTH,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )
    layer = modeling_kimi.KimiDeltaAttention(config, layer_idx=0)
    with torch.no_grad():
        layer.dt_bias.normal_()
    return layer


def test_kimi_linear_switches_to_tree_attention():
    from trl.experimental.async_grpo import packing

    from axolotl.model_support.kimi_linear.configuration_kimi import (
        KimiLinearConfig,
    )
    from axolotl.model_support.kimi_linear.modeling_kimi import KimiLinearForCausalLM

    packing.register_tree_attention()
    config = KimiLinearConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        linear_attn_config={
            "short_conv_kernel_size": WIDTH,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "kda_layers": [1],
            "full_attn_layers": [2],
        },
    )
    model = KimiLinearForCausalLM(config)
    model.set_attn_implementation(packing.TREE_ATTENTION)
    assert model.config._attn_implementation == packing.TREE_ATTENTION


def _bailing(monkeypatch):
    from axolotl.model_support.bailing_hybrid import modeling_bailing_moe_v3
    from axolotl.model_support.bailing_hybrid.configuration_bailing_moe_v3 import (
        BailingMoeV3Config,
    )

    monkeypatch.setattr(modeling_bailing_moe_v3, "chunk_kda", reference_chunk_kda)
    config = BailingMoeV3Config(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HEAD_DIM,
        short_conv_kernel_size=WIDTH,
        layer_group_size=2,
    )
    layer = modeling_bailing_moe_v3.BailingMoeV3KimiDeltaAttention(config, 0)
    with torch.no_grad():
        layer.A_log.uniform_(1, 16).log_()
        layer.dt_bias.normal_()
    return layer


def _sequences():
    rng = np.random.default_rng(0)
    system = rng.integers(1, 64, size=5).tolist()
    sequences = []
    for _ in range(3):
        prompt = system + rng.integers(1, 64, size=int(rng.integers(1, 4))).tolist()
        for _ in range(3):
            completion = rng.integers(1, 64, size=int(rng.integers(1, 6))).tolist()
            sequences.append(prompt + completion)
        sequences.append(sequences[-1][: len(prompt) + 1] + [7, 8])
    return [np.asarray(s) for s in sequences]


@pytest.mark.parametrize("build", [_kimi, _bailing], ids=["kimi_linear", "bailing"])
@pytest.mark.parametrize("forks", [True, False], ids=["fork", "paths"])
def test_tree_rows_match_each_sequence_alone(monkeypatch, build, forks):
    policy = Policy(fork_min_saved_tokens=0 if forks else 10**9)
    monkeypatch.setattr(tree_driver, "DEFAULT_POLICY", policy)
    torch.manual_seed(0)
    layer = build(monkeypatch)
    assert follows_trees(layer)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = getattr(layer, name)
        setattr(layer, name, ReferenceShortConvolution(conv.weight.shape[0]))
    layer.o_norm = ReferenceNormGated()
    layer.train()
    embed = nn.Embedding(64, HIDDEN)
    params = [embed.weight, *layer.parameters()]

    def grads():
        found = [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        return found

    sequences = _sequences()
    alone = torch.cat([layer(embed(torch.as_tensor(s))[None])[0] for s in sequences])
    alone.square().sum().backward()
    alone_grads = grads()

    packed = pack_sequences(sequences)
    assert len(packed["input_ids"]) < sum(map(len, sequences))
    tree = {
        k: packed[k]
        for k in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    out = layer(embed(packed["input_ids"])[None], **tree)[0, packed["positions"]]
    torch.testing.assert_close(out, alone, rtol=1e-4, atol=1e-5)
    out.square().sum().backward()
    for grad, expected in zip(grads(), alone_grads, strict=True):
        torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-5)
    assert policy.forks(tree_plan(tree)) == forks
