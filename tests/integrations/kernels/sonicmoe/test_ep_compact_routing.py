# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for SonicMoE EP local-pair compaction (the kernel call itself is mocked)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe import experts

E, K, H, INTER = 3, 4, 8, 6


@pytest.fixture(autouse=True)
def compact_on():
    experts.configure_ep_row_padding()
    experts.configure_ep_compact_routing(True, granule=8)
    experts._EP_COMPACT_WARNED.clear()
    yield
    experts.configure_ep_compact_routing()
    experts.configure_ep_row_padding()


def _grid(num_tokens, seed=0):
    """An EP-received grid: local ids in [0, E), remote slots as -1 or >= E, zero weight."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, E, (num_tokens, K), generator=gen)
    remote = torch.rand(num_tokens, K, generator=gen) < 0.6
    remote[1] = True  # a token with no local expert
    sentinel = torch.where(
        torch.rand(num_tokens, K, generator=gen) < 0.5,
        torch.full_like(idx, -1),
        torch.full_like(idx, E + 5),
    )
    idx = torch.where(remote, sentinel, idx)
    weights = torch.rand(num_tokens, K, generator=gen, dtype=torch.float64)
    weights = weights.masked_fill(remote, 0)
    return idx, weights


def test_pairs_are_sorted_typed_and_cover_every_local_slot():
    idx, weights = _grid(10)
    tokens, expert_ids, scores = experts.compact_local_pairs(
        idx, weights, E, True, torch.bfloat16
    )
    assert tokens.dtype == torch.int32 and expert_ids.dtype == torch.int32
    assert scores.dtype == torch.bfloat16
    assert torch.equal(tokens, tokens.sort().values)
    local = (idx >= 0) & (idx < E)
    real = expert_ids < E
    assert int(real.sum()) == int(local.sum())
    got = sorted(zip(tokens[real].tolist(), expert_ids[real].tolist(), strict=True))
    rows, cols = local.nonzero(as_tuple=True)
    want = sorted(zip(rows.tolist(), idx[rows, cols].tolist(), strict=True))
    assert got == want
    # padding and the no-local-expert token carry sentinel experts with zero score
    assert (expert_ids[~real] == E).all() and not scores[~real].float().any()
    assert 1 in tokens.tolist()
    counts = torch.bincount(tokens.long(), minlength=10)
    assert (counts >= 1).all() and (counts <= K).all()
    assert tokens.numel() % 8 == 0 and tokens.numel() >= int(local.sum()) + 1


def test_pair_padding_is_highwater_per_mode_and_capped_at_the_grid():
    experts.configure_ep_row_padding(headroom=1.0)
    experts.configure_ep_compact_routing(True, granule=8)
    idx, weights = _grid(10)
    first = experts.compact_local_pairs(idx, weights, E, True, torch.float32)[0].numel()
    fewer = idx.clone()
    fewer[2:] = E
    # within a step a smaller call pads by its own count (a recompute must match its forward)
    within = experts.compact_local_pairs(fewer, weights, E, True, torch.float32)
    assert within[0].numel() < first
    experts.promote_ep_padding_marks()
    assert (
        experts.compact_local_pairs(fewer, weights, E, True, torch.float32)[0].numel()
        == first
    )
    # eval keeps its own mark
    assert (
        experts.compact_local_pairs(fewer, weights, E, False, torch.float32)[0].numel()
        < first
    )
    full = torch.zeros((3, K), dtype=torch.long)
    experts.configure_ep_compact_routing(True, granule=1000)
    tokens = experts.compact_local_pairs(
        full, torch.ones(3, K), E, True, torch.float32
    )[0]
    assert tokens.numel() == 3 * K


def _act(x):
    return F.silu(x)


def _reference_grid(x, idx, weights, w1, w2):
    """The tokens x top_k grid semantics: remote slots contribute nothing."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            if 0 <= e < E:
                gate, up = (x[t] @ w1[e].T).chunk(2)
                out[t] = out[t] + weights[t, j] * ((_act(gate) * up) @ w2[e].T)
    return out


def _kernel_on_pairs(x, tokens, expert_ids, scores, w1, w2):
    """What moe_general_routing_inputs computes from flat pairs (sentinel experts dropped)."""
    out = torch.zeros_like(x)
    for p in range(tokens.numel()):
        e = int(expert_ids[p])
        if e >= E:
            continue
        t = int(tokens[p])
        gate, up = (x[t] @ w1[e].T).chunk(2)
        out = out.index_add(
            0,
            tokens[p : p + 1].long(),
            (scores[p] * ((_act(gate) * up) @ w2[e].T))[None],
        )
    return out


def test_compacted_pairs_reproduce_the_grid_output_and_grads():
    torch.manual_seed(0)
    idx, weights = _grid(12, seed=3)
    idx[:, :] = torch.where(
        idx[:, :] == 2, torch.full_like(idx, E), idx
    )  # expert 2 idle
    weights = weights.masked_fill(idx >= E, 0)
    x = torch.randn(12, H, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(E, 2 * INTER, H, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(E, H, INTER, dtype=torch.float64, requires_grad=True)
    weights.requires_grad_(True)

    ref = _reference_grid(x, idx, weights, w1, w2)
    grads_ref = torch.autograd.grad(ref.square().sum(), (x, w1, w2, weights))

    tokens, expert_ids, scores = experts.compact_local_pairs(
        idx, weights, E, True, torch.float64
    )
    out = _kernel_on_pairs(x, tokens, expert_ids, scores, w1, w2)
    grads = torch.autograd.grad(out.square().sum(), (x, w1, w2, weights))

    torch.testing.assert_close(out, ref)
    for got, want in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(got, want)
    assert torch.equal(out[1], torch.zeros(H, dtype=torch.float64))
    assert torch.equal(grads[0][1], torch.zeros(H, dtype=torch.float64))
    assert not grads[1][2].any() and not grads[2][2].any()
    assert not grads[3][idx >= E].any()


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = E
        self.num_experts_global = 4 * E
        self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * INTER, H))
        self.down_proj = torch.nn.Parameter(torch.randn(E, H, INTER))


def _scattermoe_stub():
    package = ModuleType("axolotl.integrations.kernels.libs.scattermoe_lora")
    package.__path__ = []
    stub = ModuleType("axolotl.integrations.kernels.libs.scattermoe_lora.experts")
    stub._get_base_param = lambda param: param
    stub._w1_name = lambda module: "gate_up_proj"
    stub.is_nvfp4_param = lambda param: False
    stub.scattermoe_experts_forward_ep = None
    return {package.__name__: package, stub.__name__: stub}


def test_ep_forward_pads_tokens_then_passes_pairs(monkeypatch):
    seen = {}

    def fake_forward(module, hidden, index, weights, pairs=None):
        seen.update(hidden=hidden, index=index, pairs=pairs)
        return hidden * 3

    monkeypatch.setattr(experts, "_sonicmoe_forward", fake_forward)
    module = _Module()
    idx, weights = _grid(1100)
    x = torch.randn(1100, H)
    with patch.dict(sys.modules, _scattermoe_stub()):
        out = experts._sonicmoe_ep_forward(module, x, idx, weights.float())
    assert seen["hidden"].shape[0] == 2048
    tokens, expert_ids, _scores = seen["pairs"]
    assert int(tokens.max()) < 2048
    # every padded token keeps one sentinel pair, like its all-sentinel grid row
    assert int((tokens >= 1100).sum()) == 2048 - 1100
    assert torch.equal(out, x * 3)


def test_compaction_is_skipped_where_it_does_not_apply(monkeypatch):
    module = _Module()
    assert experts._compact_routing_applies(module)
    module.has_gate = False
    assert not experts._compact_routing_applies(module)
    module.has_gate = True
    module._axolotl_multilora_sonic_forward = object()
    assert not experts._compact_routing_applies(module)
    del module._axolotl_multilora_sonic_forward
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not experts._compact_routing_applies(module)
    experts.configure_ep_compact_routing(False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert not experts._compact_routing_applies(module)


@pytest.mark.parametrize("transposed", [False, True])
def test_pairs_forward_calls_the_kernel_like_transformers(transposed):
    calls = {}

    def moe_general_routing_inputs(*args, **kwargs):
        calls.update(args=args, kwargs=kwargs)
        return args[0] + 1, None

    kernel = SimpleNamespace(
        moe_general_routing_inputs=moe_general_routing_inputs,
        activation_type_enum=SimpleNamespace(SWIGLU="swiglu-enum"),
    )
    shape1 = (E, H, 2 * INTER) if transposed else (E, 2 * INTER, H)
    shape2 = (E, INTER, H) if transposed else (E, H, INTER)
    facade = SimpleNamespace(
        has_post_expert_norm=False,
        config=SimpleNamespace(hidden_act="silu"),
        is_transposed=transposed,
        gate_up_proj=torch.randn(shape1),
        down_proj=torch.randn(shape2),
        gate_up_proj_bias=None,
        down_proj_bias=None,
        has_bias=False,
        num_experts=E,
        is_concatenated=True,
    )
    x = torch.randn(5, H)
    pairs = (torch.zeros(4, dtype=torch.int32),) * 2 + (torch.zeros(4),)
    with patch(
        "transformers.integrations.sonicmoe.load_sonicmoe_kernel", return_value=kernel
    ):
        out = experts._sonicmoe_pairs_forward(facade, x, *pairs)
    assert torch.equal(out, x + 1)
    args, kwargs = calls["args"], calls["kwargs"]
    assert args[0] is x and args[1] is pairs[2] and args[2] is pairs[0]
    assert args[3] is pairs[1]
    assert args[4].shape == (2 * INTER, H, E) and args[6].shape == (H, INTER, E)
    assert kwargs["E"] == E and kwargs["concat_layout"] is True
    assert kwargs["activation_type"] == "swiglu-enum"
