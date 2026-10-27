# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for the SonicMoE LoRA split path against materialized ``W + s * B @ A``.

The split autograd function runs on ``TorchOps`` (plain-torch versions of the quack / sonic-moe
primitives with the kernels' semantics) in float64, so output, d_hidden, d_router_scores and the
PEFT ``lora_A`` / ``lora_B`` grads of both linears must match the materialized reference to
float64 precision. Varlen outputs leave sentinel rows NaN, so any read of them would show.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe import experts, lora_split
from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize
from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
    TorchOps,
    glu_backward,
    lora_split_factors,
    sonicmoe_lora_split_forward,
)

E, K, H, INTER, T, RANK = 5, 3, 12, 8, 14, 3
IDLE_EXPERT, NO_LOCAL_TOKEN = 3, 6
S1, S2 = 0.75, 1.5


@pytest.fixture(autouse=True)
def reset_flag():
    lora_split.configure_lora_split(False)
    yield
    lora_split.configure_lora_split(False)


def _interleave(t):
    return t.unflatten(-1, (2, t.shape[-1] // 2)).transpose(-1, -2).flatten(-2)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("concat", [False, True])
def test_factors_reproduce_the_materialized_delta(transposed, concat):
    gen = torch.Generator().manual_seed(0)
    n_in, n_out, r, s = 6, 10, 2, 0.5
    shape = (E, n_in, n_out) if transposed else (E, n_out, n_in)
    base = torch.randn(shape, generator=gen, dtype=torch.float64)
    lora_A = torch.randn(r * E, shape[2], generator=gen, dtype=torch.float64)
    lora_B = torch.randn(shape[1], r * E, generator=gen, dtype=torch.float64)
    w_eff = MoELoRAMaterialize.apply(base, lora_A, lora_B, s)
    p, q = lora_split_factors(
        lora_A,
        lora_B,
        E,
        input_dim_last=not transposed,
        interleave=concat,
        dtype=torch.float64,
    )
    assert p.shape == (E, n_in, r) and q.shape == (E, r, n_out)
    x = torch.randn(7, n_in, generator=gen, dtype=torch.float64)
    for e in range(E):
        if transposed:
            delta = x @ w_eff[e] - x @ base[e]
        else:
            delta = x @ w_eff[e].T - x @ base[e].T
        if concat:
            delta = _interleave(delta)
        torch.testing.assert_close(s * (x @ p[e]) @ q[e], delta)


@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
def test_glu_backward_is_the_glu_vjp(activation):
    gen = torch.Generator().manual_seed(1)
    gate = torch.randn(9, 4, generator=gen, dtype=torch.float64) * 3
    up = torch.randn(9, 4, generator=gen, dtype=torch.float64)
    d = torch.randn(9, 4, generator=gen, dtype=torch.float64)
    gate.requires_grad_(True)
    up.requires_grad_(True)
    act = F.silu(gate) if activation == "swiglu" else F.gelu(gate, approximate="tanh")
    want = torch.autograd.grad(act * up, (gate, up), d)
    got = glu_backward(gate.detach(), up.detach(), d, activation)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w)


def _routing(seed=0):
    """Pairs with sentinels (expert E, nonzero score), an idle expert and a token with none."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, E - 1, (T, K), generator=gen)
    idx = torch.where(idx >= IDLE_EXPERT, idx + 1, idx)
    sentinel = torch.rand(T, K, generator=gen) < 0.35
    sentinel[NO_LOCAL_TOKEN] = True
    idx = idx.masked_fill(sentinel, E)
    weights = torch.rand(T, K, generator=gen, dtype=torch.float64) + 0.1
    x = torch.randn(T, H, generator=gen, dtype=torch.float64)
    tokens, expert_ids, scores = experts._grid_pairs(x, idx, weights)
    return x, tokens, expert_ids, scores


def test_routing_metadata_groups_valid_pairs_and_marks_sentinels():
    x, tokens, expert_ids, _ = _routing()
    meta = TorchOps().routing_metadata(tokens, expert_ids, T, E)
    num_slots = tokens.numel()
    valid = expert_ids < E
    n_valid = int(valid.sum())
    assert int(meta.cu_seqlens[-1]) == n_valid
    assert meta.cu_seqlens.diff()[IDLE_EXPERT] == 0
    rev = meta.s_reverse_scatter_idx
    assert (rev[~valid] == num_slots).all()
    grouped = meta.s_scatter_idx[:n_valid].long()
    assert torch.equal(rev[grouped], torch.arange(n_valid, dtype=torch.int32))
    assert torch.equal(meta.x_gather_idx[:n_valid], tokens[grouped])
    owner = torch.repeat_interleave(torch.arange(E), meta.cu_seqlens.diff().long())
    assert torch.equal(expert_ids[grouped].long(), owner)
    assert meta.token_offset[-1] == num_slots
    assert torch.equal(meta.token_offset.diff(), torch.full((T,), K, dtype=torch.int32))


class _Experts(torch.nn.Module):
    def __init__(self, transposed, concat, act, gen):
        super().__init__()
        self.num_experts = E
        self.is_transposed = transposed
        self.is_concatenated = concat
        self.has_gate = True
        self.config = SimpleNamespace(hidden_act=act)
        shape1 = (E, H, 2 * INTER) if transposed else (E, 2 * INTER, H)
        shape2 = (E, INTER, H) if transposed else (E, H, INTER)
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(shape1, generator=gen, dtype=torch.float64) / math.sqrt(H),
            requires_grad=False,
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(shape2, generator=gen, dtype=torch.float64) / math.sqrt(INTER),
            requires_grad=False,
        )


def _lora_pair(weight, gen):
    _, dim1, dim2 = weight.shape
    a = torch.randn(RANK * E, dim2, generator=gen, dtype=torch.float64) * 0.3
    b = torch.randn(dim1, RANK * E, generator=gen, dtype=torch.float64) * 0.3
    return a.requires_grad_(True), b.requires_grad_(True)


def _materialized(module, x, tokens, expert_ids, scores, lora1, lora2, act):
    w1, w2 = module.gate_up_proj, module.down_proj
    if lora1 is not None:
        w1 = MoELoRAMaterialize.apply(w1, *lora1, S1)
    if lora2 is not None:
        w2 = MoELoRAMaterialize.apply(w2, *lora2, S2)
    out = torch.zeros_like(x)
    for e in range(E):
        slots = (expert_ids == e).nonzero().squeeze(1)
        if slots.numel() == 0:
            continue
        rows = x[tokens[slots].long()]
        hp = rows @ w1[e] if module.is_transposed else rows @ w1[e].T
        if module.is_concatenated:
            gate, up = hp.chunk(2, dim=-1)
        else:
            gate, up = hp[:, ::2], hp[:, 1::2]
        glu = (F.silu(gate) if act == "silu" else F.gelu(gate, approximate="tanh")) * up
        y = glu @ w2[e] if module.is_transposed else glu @ w2[e].T
        out = out.index_add(0, tokens[slots].long(), y * scores[slots, None])
    return out


LAYOUTS = [(False, True), (False, False), (True, False)]


@pytest.mark.parametrize(("transposed", "concat"), LAYOUTS)
@pytest.mark.parametrize("act", ["silu", "gelu_pytorch_tanh"])
@pytest.mark.parametrize("targets", ["both", "gate_up", "down"])
def test_split_matches_materialized_forward_and_grads(transposed, concat, act, targets):
    gen = torch.Generator().manual_seed(2)
    module = _Experts(transposed, concat, act, gen)
    x, tokens, expert_ids, scores = _routing()
    lora1 = _lora_pair(module.gate_up_proj, gen) if targets != "down" else None
    lora2 = _lora_pair(module.down_proj, gen) if targets != "gate_up" else None
    grad_out = torch.randn(T, H, generator=gen, dtype=torch.float64)
    ref_act = "silu" if act == "silu" else "gelu"
    leaves = [t for pair in (lora1, lora2) if pair is not None for t in pair]

    def run(fn):
        xx = x.clone().requires_grad_(True)
        ss = scores.clone().requires_grad_(True)
        out = fn(xx, ss)
        grads = torch.autograd.grad((out * grad_out).sum(), [xx, ss, *leaves])
        return out.detach(), grads

    want_out, want = run(
        lambda xx, ss: _materialized(
            module, xx, tokens, expert_ids, ss, lora1, lora2, ref_act
        )
    )
    facade = experts._LoRAExpertsFacade(
        module, module.gate_up_proj, None, module.down_proj, None
    )
    got_out, got = run(
        lambda xx, ss: sonicmoe_lora_split_forward(
            facade,
            xx,
            tokens,
            expert_ids,
            ss,
            None if lora1 is None else (*lora1, S1),
            None if lora2 is None else (*lora2, S2),
            ops=TorchOps(),
        )
    )

    assert torch.isfinite(got_out).all()
    torch.testing.assert_close(got_out, want_out, rtol=1e-10, atol=1e-10)
    for g, w in zip(got, want, strict=True):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, w, rtol=1e-9, atol=1e-10)

    dx, dscores = got[0], got[1]
    assert not got_out[NO_LOCAL_TOKEN].any() and not dx[NO_LOCAL_TOKEN].any()
    assert not dscores[expert_ids == E].any()
    for d_a, d_b in zip(leaves[::2], got[3::2], strict=True):
        del d_a
        assert not d_b.reshape(d_b.shape[0], RANK, E)[:, :, IDLE_EXPERT].any()
    for d_a in got[2::2]:
        assert not d_a[IDLE_EXPERT * RANK : (IDLE_EXPERT + 1) * RANK].any()


def test_separate_addend_buffers_match_the_aliased_ones(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    module = _Experts(False, True, "silu", gen)
    x, tokens, expert_ids, scores = _routing(seed=1)
    lora1 = _lora_pair(module.gate_up_proj, gen)
    lora2 = _lora_pair(module.down_proj, gen)
    facade = experts._LoRAExpertsFacade(
        module, module.gate_up_proj, None, module.down_proj, None
    )

    def run():
        xx = x.clone().requires_grad_(True)
        out = sonicmoe_lora_split_forward(
            facade,
            xx,
            tokens,
            expert_ids,
            scores,
            (*lora1, S1),
            (*lora2, S2),
            ops=TorchOps(),
        )
        return out, torch.autograd.grad(out.square().sum(), [xx, *lora1, *lora2])

    aliased = run()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    separate = run()
    torch.testing.assert_close(aliased[0], separate[0], rtol=0, atol=0)
    for a, b in zip(aliased[1], separate[1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class _Gated(SimpleNamespace):
    pass


def _gate_case(**overrides):
    module = _Gated(
        has_gate=True,
        is_transposed=False,
        is_concatenated=True,
        config=SimpleNamespace(hidden_act="silu"),
    )
    w1 = torch.zeros(E, 2 * INTER, H, dtype=torch.bfloat16)
    w2 = torch.zeros(E, H, INTER, dtype=torch.bfloat16)
    case = {
        "module": module,
        "hidden_states": torch.zeros(T, H, dtype=torch.bfloat16),
        "w1": w1,
        "b1": None,
        "w2": w2,
        "b2": None,
    }
    for key, value in overrides.items():
        if key in case:
            case[key] = value
        else:
            setattr(module, key, value)
    return case


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({}, None),
        ({"has_gate": False}, "non-gated experts"),
        ({"hidden_states": torch.zeros(T, H)}, "non-bf16 experts"),
        (
            {
                "w1": torch.nn.Parameter(
                    torch.zeros(E, 2 * INTER, H, dtype=torch.bfloat16)
                )
            },
            "trainable base expert weights",
        ),
        ({"b1": torch.zeros(E, 2 * INTER)}, "expert bias"),
        ({"has_post_expert_norm": True}, "post-expert norm"),
        ({"config": SimpleNamespace(hidden_act="relu")}, "'relu' activation"),
        ({"config": SimpleNamespace(hidden_activation="gelu_pytorch_tanh")}, None),
        ({"is_transposed": True}, "transposed concatenated gate_up layout"),
        ({"is_transposed": True, "is_concatenated": False}, None),
    ],
)
def test_unsupported_reasons(overrides, reason):
    case = _gate_case(**overrides)
    assert lora_split.lora_split_unsupported_reason(**case) == reason


def test_flag_gates_the_split_path_and_warns_once_per_reason():
    supported = _gate_case()
    biased = _gate_case(b1=torch.zeros(E, 2 * INTER))
    assert not lora_split.lora_split_applies(**supported)
    assert not lora_split.lora_split_applies(**biased)
    assert not lora_split._LORA_SPLIT_WARNED

    lora_split.configure_lora_split(True)
    assert lora_split.lora_split_applies(**supported)
    assert not lora_split.lora_split_applies(**biased)
    assert not lora_split.lora_split_applies(**biased)
    assert lora_split._LORA_SPLIT_WARNED == {"expert bias"}
    lora_split.configure_lora_split(True)
    assert not lora_split._LORA_SPLIT_WARNED


def test_kernels_args_accepts_the_flag():
    from axolotl.integrations.kernels.args import KernelsArgs

    assert KernelsArgs(sonicmoe_lora_split=True).sonicmoe_lora_split is True
    assert KernelsArgs().sonicmoe_lora_split is None
