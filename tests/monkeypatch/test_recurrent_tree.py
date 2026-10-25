"""The tree driver against a toy recurrence: packed trees equal sequences alone."""

import numpy as np
import pytest
import torch

from axolotl.core.trainers.grpo.async_scoring import pack_sequences
from axolotl.monkeypatch.models import recurrent_tree
from axolotl.monkeypatch.models.recurrent_tree import (
    RecurrentComponent,
    tree_conv,
    tree_plan,
    tree_scan,
)

WIDTH = 4
DIM = 3


def decayed_sum(inputs, cu_seqlens, initial_state):
    """h_t = a_t * h_{t-1} + x_t per sequence, from ``initial_state`` or zeros."""
    x, a = inputs
    bounds = cu_seqlens.tolist()
    outs, finals = [], []
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        h = x.new_zeros(DIM) if initial_state is None else initial_state[i]
        for t in range(start, end):
            h = a[0, t] * h + x[0, t]
            outs.append(h)
        finals.append(h)
    return torch.stack(outs)[None], torch.stack(finals)


def causal_conv(x, cu_seqlens, weight):
    bounds = cu_seqlens.tolist()
    outs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        seq = torch.nn.functional.pad(x[0, start:end].T[None], (WIDTH - 1, 0))
        outs.append(torch.nn.functional.conv1d(seq, weight[:, None], groups=DIM)[0].T)
    return torch.cat(outs)[None]


def _sequences(rng):
    """Rollouts sharing a system prefix and per-group prompts, some branching late."""
    system = rng.integers(1, 50, size=5).tolist()
    sequences = []
    for _ in range(3):
        prompt = system + rng.integers(1, 50, size=int(rng.integers(1, 4))).tolist()
        for _ in range(3):
            completion = rng.integers(1, 50, size=int(rng.integers(1, 6))).tolist()
            sequences.append(prompt + completion)
        # a branch of the last completion after its first tokens
        sequences.append(sequences[-1][: len(prompt) + 1] + [7, 8])
    return [np.asarray(s) for s in sequences]


@pytest.mark.parametrize(
    "initial_state, min_saved, forks",
    [(True, 0, True), (True, 10**9, False), (False, 0, False)],
    ids=["fork", "paths-by-cost", "paths-without-initial-state"],
)
def test_tree_matches_each_sequence_alone(monkeypatch, initial_state, min_saved, forks):
    monkeypatch.setattr(recurrent_tree, "FORK_MIN_SAVED_TOKENS", min_saved)
    rng = np.random.default_rng(0)
    sequences = _sequences(rng)
    torch.manual_seed(0)
    embed = torch.randn(50, DIM, requires_grad=True)
    decay = torch.randn(50, DIM, requires_grad=True)
    weight = torch.randn(DIM, WIDTH, requires_grad=True)
    component = RecurrentComponent("toy", initial_state=initial_state)
    initial_states = []

    def scan(inputs, cu_seqlens, initial):
        initial_states.append(initial is not None)
        return decayed_sum(inputs, cu_seqlens, initial)

    def run(ids, plan=None, cu_seqlens=None):
        x = embed[ids][None]
        a = torch.sigmoid(decay[ids])[None]
        if plan is None:
            mixed = causal_conv(x, cu_seqlens, weight)
            return decayed_sum([mixed, a], cu_seqlens, None)[0]
        mixed = tree_conv(lambda v, cu: causal_conv(v, cu, weight), x, plan, WIDTH)
        return tree_scan(scan, [mixed, a], plan, component)

    alone = [
        run(torch.as_tensor(s), cu_seqlens=torch.tensor([0, len(s)])) for s in sequences
    ]
    alone_out = torch.cat([o[0] for o in alone])
    alone_out.square().sum().backward()
    alone_grads = [t.grad.clone() for t in (embed, decay, weight)]
    for t in (embed, decay, weight):
        t.grad = None

    packed = pack_sequences(sequences)
    assert len(packed["input_ids"]) < sum(map(len, sequences))
    plan = tree_plan(
        {
            k: packed[k]
            for k in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
        }
    )
    assert len(plan.levels) >= 3 and min(plan.lengths) == 1
    out = run(packed["input_ids"], plan=plan)
    tree_out = out[0, packed["positions"]]
    torch.testing.assert_close(tree_out, alone_out)
    tree_out.square().sum().backward()
    for tensor, grad in zip((embed, decay, weight), alone_grads, strict=True):
        torch.testing.assert_close(tensor.grad, grad)
    assert any(initial_states) == forks
    assert len(initial_states) == (len(plan.levels) if forks else 1)


def test_plan_is_built_once_per_row():
    packed = pack_sequences([np.array([1, 2, 3]), np.array([1, 2, 4])])
    kwargs = {
        k: packed[k]
        for k in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    assert tree_plan(kwargs) is tree_plan(dict(kwargs))
    assert tree_plan({}) is None
