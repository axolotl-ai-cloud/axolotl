# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Qwen3.5 (Gated DeltaNet + attention) on tree rows scores every sample as it scores
alone, through transformers' own forward with the module functions hooked."""

import sys
import types

import pytest
import torch
from tree_recurrence import Policy, install, reference, uninstall
from tree_recurrence._hooks import hook_of
from tree_recurrence.integrations import transformers as integration
from tree_testing import assert_tree_matches_alone, branching_sequences

transformers = pytest.importorskip("transformers")
modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

GDN_FUNCTIONS = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")


def qwen3_5(gradient_checkpointing=False):
    config = transformers.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
    )
    torch.manual_seed(0)
    model = modeling.Qwen3_5ForCausalLM(config).float()
    for name, param in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            torch.nn.init.normal_(param, std=0.5)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model.train()


@pytest.fixture
def driver_calls(monkeypatch):
    """The tree driver calls the hooks make, by kind."""
    calls = {"conv": 0, "scan": 0}

    def spy(kind, function):
        def call(*args, **kwargs):
            calls[kind] += 1
            return function(*args, **kwargs)

        return call

    monkeypatch.setattr(integration, "tree_conv", spy("conv", integration.tree_conv))
    monkeypatch.setattr(integration, "tree_scan", spy("scan", integration.tree_scan))
    return calls


@pytest.mark.parametrize("fmt", ["trl-1.16", "trl-7686"])
@pytest.mark.parametrize(
    "gradient_checkpointing", [False, True], ids=["", "checkpointed"]
)
def test_torch_reference_kernels(driver_calls, fmt, gradient_checkpointing):
    """Without FLA the scan runs every path as one padded batch."""
    model = qwen3_5(gradient_checkpointing)
    support = install(model)
    assert support.mode == "tree", support.reason
    assert support.owners == {"Qwen3_5GatedDeltaNet": "tree_recurrence"}
    assert_tree_matches_alone(model, branching_sequences(), fmt=fmt)
    # the forward really calls the hooks: two GDN layers, twice when checkpointed
    expected = 2 * (2 if gradient_checkpointing else 1)
    assert driver_calls == {"conv": expected, "scan": expected}


@pytest.mark.parametrize(
    "min_saved, forks", [(0, True), (10**9, False)], ids=["fork", "paths"]
)
def test_varlen_kernel(monkeypatch, min_saved, forks):
    """A kernel with cu_seqlens and initial states (FLA's API) forks by policy."""
    initial_states = []

    def chunk_gated_delta_rule(*args, initial_state=None, **kwargs):
        if kwargs.get("cu_seqlens") is not None:
            initial_states.append(initial_state is not None)
        return reference.chunk_gated_delta_rule(
            *args, initial_state=initial_state, **kwargs
        )

    monkeypatch.setattr(
        modeling, "torch_chunk_gated_delta_rule", chunk_gated_delta_rule
    )
    model = qwen3_5()
    assert install(model, policy=Policy(fork_min_saved_tokens=min_saved)).mode == "tree"
    assert_tree_matches_alone(model, branching_sequences(), rtol=1e-4)
    assert initial_states and any(initial_states) == forks


def test_chains_and_collation_padding():
    model = qwen3_5()
    install(model)
    assert_tree_matches_alone(model, branching_sequences(), share=False)
    assert_tree_matches_alone(model, branching_sequences(), collation_padding=3)


def test_install_is_idempotent_and_reversible():
    originals = {name: getattr(modeling, name) for name in GDN_FUNCTIONS}
    model = qwen3_5()
    install(model)
    wrappers = {name: getattr(modeling, name) for name in GDN_FUNCTIONS}
    assert all(wrappers[name] is not originals[name] for name in GDN_FUNCTIONS)
    install(model, policy=Policy(fork_min_saved_tokens=1))
    assert {name: getattr(modeling, name) for name in GDN_FUNCTIONS} == wrappers
    assert integration._INSTALLED[modeling.__name__].policy.fork_min_saved_tokens == 1
    uninstall(model)
    assert {name: getattr(modeling, name) for name in GDN_FUNCTIONS} == originals


def test_a_hook_wrapped_by_someone_else_passes_through():
    original = modeling.torch_chunk_gated_delta_rule
    model = qwen3_5()
    install(model)
    ours = modeling.torch_chunk_gated_delta_rule
    modeling.torch_chunk_gated_delta_rule = lambda *a, **k: ours(*a, **k)
    try:
        uninstall(model)
        assert not ours.__tree_recurrence_hook__.enabled
        # and a later install wraps the outer function, not a second hook of ours
        install(model)
        assert hook_of(modeling, "torch_chunk_gated_delta_rule").original is not ours
    finally:
        uninstall(model)
        modeling.torch_chunk_gated_delta_rule = original


def test_a_stale_hook_left_on_the_module_is_replaced():
    original = modeling.torch_chunk_gated_delta_rule
    model = qwen3_5()
    install(model)
    stale = modeling.torch_chunk_gated_delta_rule
    uninstall(model)
    modeling.torch_chunk_gated_delta_rule = stale  # as a monkeypatch undo would
    try:
        install(model)
        assert hook_of(modeling, "torch_chunk_gated_delta_rule").original is original
        assert_tree_matches_alone(model, branching_sequences())
    finally:
        uninstall(model)
        modeling.torch_chunk_gated_delta_rule = original


def test_missing_seam_is_refused(monkeypatch):
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return hidden_states  # calls neither module function

    monkeypatch.setattr(modeling.Qwen3_5GatedDeltaNet, "forward", forward)
    support = install(qwen3_5())
    assert support.mode == "none" and "does not call" in support.reason


def test_a_replaced_instance_forward_is_refused():
    model = qwen3_5()
    mixer = model.model.layers[0].linear_attn
    mixer.forward = types.MethodType(
        lambda self, hidden_states, **kwargs: hidden_states, mixer
    )
    support = install(model)
    assert support.mode == "none" and "replaced" in support.reason


@pytest.fixture
def fake_trl(monkeypatch):
    """A stand-in for TRL's packing module with huggingface/trl#7686's
    ``patch_tree_gdn``, which wraps a module's functions once per module."""
    packing = types.ModuleType(integration.TRL_PACKING)
    packing._PATCHED = set()
    packing.calls = 0

    def patch_tree_gdn(model):
        packing.calls += 1
        module = sys.modules[type(model).__module__]
        if module in packing._PATCHED:
            return True
        for name in GDN_FUNCTIONS:
            original = getattr(module, name)
            setattr(
                module, name, lambda *a, _original=original, **k: _original(*a, **k)
            )
        packing._PATCHED.add(module)
        return True

    patch_tree_gdn.__module__ = integration.TRL_PACKING
    packing.patch_tree_gdn = patch_tree_gdn
    monkeypatch.setitem(sys.modules, integration.TRL_PACKING, packing)
    originals = {name: getattr(modeling, name) for name in GDN_FUNCTIONS}
    yield packing
    for name, original in originals.items():
        setattr(modeling, name, original)


def test_a_module_trl_patched_is_left_to_trl(fake_trl):
    model = qwen3_5()
    fake_trl.patch_tree_gdn(model)
    support = install(model)
    assert support.mode == "tree"
    assert support.owners == {"Qwen3_5GatedDeltaNet": "trl"}
    assert hook_of(modeling, "torch_chunk_gated_delta_rule") is None


def test_trl_does_not_patch_a_module_patched_here(fake_trl):
    model = qwen3_5()
    install(model)
    assert modeling in fake_trl._PATCHED
    assert fake_trl.patch_tree_gdn(model)
    assert hook_of(modeling, "torch_chunk_gated_delta_rule") is not None
    uninstall(model)
    assert modeling not in fake_trl._PATCHED


def test_trl_backend(fake_trl):
    model = qwen3_5()
    install(model)
    support = install(model, backends={"gated_delta_net": "trl"})
    assert support.owners == {"Qwen3_5GatedDeltaNet": "trl"} and fake_trl.calls == 1
    assert hook_of(modeling, "torch_chunk_gated_delta_rule") is None
    assert modeling in fake_trl._PATCHED


def test_trl_backend_without_trl_support(monkeypatch, caplog):
    packing = types.ModuleType(integration.TRL_PACKING)  # TRL 1.16: no patch_tree_gdn
    monkeypatch.setitem(sys.modules, integration.TRL_PACKING, packing)
    support = install(qwen3_5(), backends={"gated_delta_net": "trl"})
    assert support.owners == {"Qwen3_5GatedDeltaNet": "tree_recurrence"}
    assert "patch_tree_gdn" in caplog.text


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        install(qwen3_5(), backends={"gated_delta_net": "fla"})


def test_without_hooks_packed_samples_leak():
    with pytest.raises(AssertionError):
        assert_tree_matches_alone(qwen3_5(), branching_sequences())
