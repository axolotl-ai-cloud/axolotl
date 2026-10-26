# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Binding the causal-conv1d hub kernel beneath the tree hooks."""

import types

import pytest
from test_gated_delta_net import modeling, qwen3_5
from tree_recurrence import install, reference, uninstall
from tree_recurrence._hooks import hook_of
from tree_recurrence.integrations import hub_kernels
from tree_testing import assert_tree_matches_alone, branching_sequences


@pytest.fixture
def hub_kernel(monkeypatch):
    calls = []

    def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
        calls.append(x.shape[-1])
        return reference.causal_conv1d(x, weight, bias, seq_idx, activation)

    kernel = types.SimpleNamespace(
        causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=lambda *a, **k: None
    )
    monkeypatch.setattr(hub_kernels, "hub_causal_conv1d", lambda: kernel)
    originals = {name: getattr(modeling, name) for name in hub_kernels.NAMES}
    yield calls
    uninstall()
    for name, original in originals.items():
        setattr(modeling, name, original)


def test_bound_beneath_the_hook(hub_kernel):
    model = qwen3_5()
    install(model)
    hook = hook_of(modeling, "causal_conv1d_fn")
    assert hub_kernels.use_hub_causal_conv1d(model)
    assert hook_of(modeling, "causal_conv1d_fn") is hook
    assert hook.original.__wrapped__.__name__ == "causal_conv1d_fn"
    assert_tree_matches_alone(model, branching_sequences())
    assert hub_kernel


def test_without_the_kernel(monkeypatch):
    monkeypatch.setattr(hub_kernels, "hub_causal_conv1d", lambda: None)
    assert not hub_kernels.use_hub_causal_conv1d(modeling)
