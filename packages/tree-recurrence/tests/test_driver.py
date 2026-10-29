# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The driver against a toy recurrence: packed trees equal sequences alone."""

import pytest
import torch
from tree_recurrence import Policy, TreePlan, padded_scan, tree_conv, tree_scan
from tree_testing import branching_sequences, pack

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


def batched_decayed_sum(inputs, initial_state):
    x, a = inputs
    h = x.new_zeros(x.shape[0], DIM)
    outs = []
    for t in range(x.shape[1]):
        h = a[:, t] * h + x[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


def causal_conv(x, cu_seqlens, weight):
    bounds = cu_seqlens.tolist()
    outs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        seq = torch.nn.functional.pad(x[0, start:end].T[None], (WIDTH - 1, 0))
        outs.append(torch.nn.functional.conv1d(seq, weight[:, None], groups=DIM)[0].T)
    return torch.cat(outs)[None]


@pytest.mark.parametrize("fmt", ["trl-1.16", "trl-7686", "parents"])
@pytest.mark.parametrize(
    "scan_kind, min_saved, forks",
    [("varlen", 0, True), ("varlen", 10**9, False), ("padded", 0, False)],
    ids=["fork", "paths-by-cost", "paths-padded"],
)
def test_tree_matches_each_sequence_alone(fmt, scan_kind, min_saved, forks):
    sequences = branching_sequences(vocab_size=50)
    torch.manual_seed(0)
    embed = torch.randn(50, DIM, requires_grad=True)
    decay = torch.randn(50, DIM, requires_grad=True)
    weight = torch.randn(DIM, WIDTH, requires_grad=True)
    calls = []

    def scan(inputs, cu_seqlens, initial):
        calls.append(initial is not None)
        return decayed_sum(inputs, cu_seqlens, initial)

    if scan_kind == "padded":
        scan = padded_scan(batched_decayed_sum)

    def run(ids, plan=None, cu_seqlens=None):
        x = embed[ids][None]
        a = torch.sigmoid(decay[ids])[None]
        if plan is None:
            mixed = causal_conv(x, cu_seqlens, weight)
            return decayed_sum([mixed, a], cu_seqlens, None)[0]
        mixed = tree_conv(lambda v, cu: causal_conv(v, cu, weight), x, plan, WIDTH)
        return tree_scan(
            scan,
            [mixed, a],
            plan,
            forkable=scan_kind == "varlen",
            policy=Policy(fork_min_saved_tokens=min_saved),
        )

    alone = [
        run(torch.as_tensor(s), cu_seqlens=torch.tensor([0, len(s)])) for s in sequences
    ]
    alone_out = torch.cat([o[0] for o in alone])
    alone_out.square().sum().backward()
    alone_grads = [t.grad.clone() for t in (embed, decay, weight)]
    for t in (embed, decay, weight):
        t.grad = None

    packed = pack(sequences)
    assert len(packed.input_ids) < sum(map(len, sequences))
    plan = TreePlan.from_kwargs(packed.kwargs(fmt))
    assert plan.num_levels >= 3 and min(plan.lengths) == 1
    out = run(packed.input_ids, plan=plan)
    tree_out = out[0, torch.cat(packed.positions)]
    torch.testing.assert_close(tree_out, alone_out)
    tree_out.square().sum().backward()
    for tensor, grad in zip((embed, decay, weight), alone_grads, strict=True):
        torch.testing.assert_close(tensor.grad, grad)
    if scan_kind == "varlen":
        assert any(calls) == forks
        assert len(calls) == (plan.num_levels if forks else 1)


def test_collation_padding_gets_zeros():
    plan = TreePlan.from_kwargs(pack([[1, 2, 3], [1, 2, 4]]).kwargs())
    x = torch.randn(1, plan.num_tokens + 2, DIM)
    conv = tree_conv(lambda v, cu: v, x, plan, WIDTH)
    scan = tree_scan(lambda inputs, cu, s: (inputs[0], None), [x], plan, forkable=False)
    for out in (conv, scan):
        assert out.shape == x.shape
        assert not out[:, plan.num_tokens :].any()
        torch.testing.assert_close(out[:, : plan.num_tokens], x[:, : plan.num_tokens])


def test_unbranched_rows_run_in_one_call():
    plan = TreePlan.from_cu_seqlens(torch.tensor([0, 2, 5]))
    calls = []

    def scan(inputs, cu_seqlens, initial):
        calls.append((cu_seqlens.tolist(), initial))
        return inputs[0], None

    x = torch.randn(1, 5, DIM)
    torch.testing.assert_close(tree_scan(scan, [x], plan), x)
    assert calls == [([0, 2, 5], None)]


def test_rows_are_checked():
    plan = TreePlan.from_cu_seqlens(torch.tensor([0, 2, 5]))
    for x in (torch.randn(2, 5, DIM), torch.randn(1, 4, DIM)):
        with pytest.raises(ValueError, match="tree-packed row"):
            tree_scan(lambda inputs, cu, s: (inputs[0], None), [x], plan)
        with pytest.raises(ValueError, match="tree-packed row"):
            tree_conv(lambda v, cu: v, x, plan, WIDTH)
