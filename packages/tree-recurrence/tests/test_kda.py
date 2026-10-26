# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Kimi Delta Attention on tree rows: a layer following trees through its own
forward (as an in-tree model does), with FLA's API run by the PyTorch reference."""

import pytest
import torch
from torch import nn
from tree_recurrence import (
    Policy,
    TreeFamily,
    install,
    mark_tree_forward,
    reference,
    register_family,
    tree_conv,
    tree_plan,
    tree_scan,
    unregister_family,
)
from tree_recurrence.components import kda_scan, short_convolution
from tree_testing import branching_sequences, pack

HIDDEN, HEADS, HEAD_DIM, WIDTH = 32, 4, 8, 4


class ShortConvolution(nn.Conv1d):
    """FLA's ShortConvolution (silu) as a per-sequence depthwise conv."""

    def __init__(self, channels):
        super().__init__(channels, channels, WIDTH, groups=channels, bias=False)

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            seq = nn.functional.pad(x[:, start:end].transpose(1, 2), (WIDTH - 1, 0))
            out = nn.functional.conv1d(seq, self.weight, groups=self.groups)
            outs.append(nn.functional.silu(out).transpose(1, 2))
        return torch.cat(outs, dim=1), None


class TinyKimiDeltaAttention(nn.Module):
    def __init__(self, policy):
        super().__init__()
        size = HEADS * HEAD_DIM
        self.policy = policy
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(HIDDEN, size) for _ in range(3)
        )
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            ShortConvolution(size) for _ in range(3)
        )
        self.f_proj = nn.Linear(HIDDEN, size)
        self.b_proj = nn.Linear(HIDDEN, HEADS)
        self.o_proj = nn.Linear(size, HIDDEN)

    @mark_tree_forward
    def forward(self, hidden_states, **kwargs):
        tree = tree_plan(kwargs)
        heads = (*hidden_states.shape[:2], HEADS, HEAD_DIM)
        q, k, v = (
            conv(x=proj(hidden_states))[0]
            if tree is None
            else tree_conv(short_convolution(conv), proj(hidden_states), tree, WIDTH)
            for proj, conv in (
                (self.q_proj, self.q_conv1d),
                (self.k_proj, self.k_conv1d),
                (self.v_proj, self.v_conv1d),
            )
        )
        q, k, v = (x.reshape(heads) for x in (q, k, v))
        g = -nn.functional.softplus(self.f_proj(hidden_states)).reshape(heads)
        beta = self.b_proj(hidden_states).sigmoid()
        if tree is None:
            o, _ = reference.chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        else:
            scan = kda_scan(reference.chunk_kda)
            o = tree_scan(scan, [q, k, v, g, beta], tree, policy=self.policy)
        return self.o_proj(o.reshape(*hidden_states.shape[:2], -1))


class Config:
    model_type = "tiny_kimi"

    def get_text_config(self):
        return self


class TinyKimi(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.config = Config()
        self.embed = nn.Embedding(64, HIDDEN)
        self.mixer = TinyKimiDeltaAttention(policy)

    def forward(self, input_ids, **kwargs):
        return self.mixer(self.embed(input_ids), **kwargs)


@pytest.fixture
def family():
    register_family(TreeFamily("tiny_kda", ("tiny_kimi",), ("TinyKimiDeltaAttention",)))
    yield
    unregister_family("tiny_kda")


@pytest.mark.parametrize(
    "min_saved, forks", [(0, True), (10**9, False)], ids=["fork", "paths"]
)
def test_tree_rows_match_each_sequence_alone(family, min_saved, forks):
    torch.manual_seed(0)
    model = TinyKimi(Policy(fork_min_saved_tokens=min_saved))
    support = install(model)
    assert support.mode == "tree" and support.owners == {
        "TinyKimiDeltaAttention": "model"
    }
    params = list(model.parameters())

    def grads():
        found = [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        return found

    sequences = branching_sequences()
    alone = torch.cat([model(torch.as_tensor(s)[None])[0] for s in sequences])
    alone.square().sum().backward()
    alone_grads = grads()

    packed = pack(sequences)
    tree = packed.kwargs("trl-7686")
    out = model(packed.input_ids[None], **tree)[0, torch.cat(packed.positions)]
    torch.testing.assert_close(out, alone, rtol=1e-4, atol=1e-5)
    out.square().sum().backward()
    for grad, expected in zip(grads(), alone_grads, strict=True):
        torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-5)
    assert model.mixer.policy.forks(tree_plan(tree)) == forks


def test_an_unmarked_forward_is_refused(family, monkeypatch):
    def forward(self, hidden_states, **kwargs):
        return hidden_states

    monkeypatch.setattr(TinyKimiDeltaAttention, "forward", forward)
    support = install(TinyKimi(Policy()))
    assert support.mode == "none" and "not marked" in support.reason
