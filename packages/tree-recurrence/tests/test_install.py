# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""What ``install`` reports for models with and without recurrent layers."""

import pytest
import torch
from tree_recurrence import (
    TreeFamily,
    families,
    install,
    register_family,
    tree_family,
    unregister_family,
)
from tree_recurrence._hooks import dispatches_to_kernel

transformers = pytest.importorskip("transformers")


def test_attention_only_models_share_prefixes():
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    support = install(transformers.LlamaForCausalLM(config))
    assert support.mode == "tree" and support.shares_prefixes
    assert support.reason == "no recurrent layers"


def tiny_mamba():
    config = transformers.MambaConfig(
        vocab_size=64, hidden_size=32, state_size=4, num_hidden_layers=1
    )
    return transformers.MambaForCausalLM(config)


def test_recurrent_models_without_a_family_are_refused():
    support = install(tiny_mamba())
    assert support.mode == "none" and not support.shares_prefixes
    assert "mamba" in support.reason


def test_a_chains_family():
    register_family(
        TreeFamily(
            "mamba1",
            ("mamba",),
            ("MambaMixer",),
            layer_types=("linear_attention",),
            support="chains",
        )
    )
    try:
        support = install(tiny_mamba())
    finally:
        unregister_family("mamba1")
    assert support.mode == "chains" and support.owners == {"MambaMixer": "model"}


def test_a_family_whose_mixers_are_missing():
    register_family(TreeFamily("renamed", ("mamba",), ("RenamedMixer",)))
    try:
        support = install(tiny_mamba())
    finally:
        unregister_family("renamed")
    assert support.mode == "none" and "RenamedMixer" in support.reason


def test_built_in_families():
    names = {family.name for family in families()}
    assert {"gated_delta_net", "mamba2", "short_conv"} <= names
    assert tree_family(["qwen3_5_text"]).name == "gated_delta_net"
    assert tree_family(["nemotron_h"]).name == "mamba2"
    assert tree_family(["llama"]) is None


def test_kernel_dispatch_detection():
    from transformers.integrations.hub_kernels import (
        use_kernel_func_from_hub_with_fallback,
    )

    def reference_add(x, y, **kwargs):
        return x + y

    fallback = use_kernel_func_from_hub_with_fallback("missing", "no_such_package")(
        reference_add
    )
    resolved = use_kernel_func_from_hub_with_fallback("dumps", "json")(reference_add)
    assert not dispatches_to_kernel(fallback)
    assert dispatches_to_kernel(resolved)
    assert dispatches_to_kernel(reference_add)  # anything else is taken for a kernel

    # kernels >= 0.17 turns the dispatcher into a module whose forward closes over it,
    # and kernelize swaps an instance's forward for a hub kernel's
    def module_of(func):
        class Func(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return func(*args, **kwargs)

        return Func()

    dispatcher = module_of(fallback)
    assert not dispatches_to_kernel(dispatcher)
    dispatcher.forward = lambda *args, **kwargs: None
    assert dispatches_to_kernel(dispatcher)


def test_tree_rows_build_no_cache():
    from test_gated_delta_net import qwen3_5
    from tree_recurrence import uninstall
    from tree_testing import pack

    model = qwen3_5()
    install(model)
    packed = pack([[1, 2, 3], [1, 2, 4]])
    tree = packed.kwargs()
    out = model(input_ids=packed.input_ids[None], **tree)
    assert out.past_key_values is None
    plain = model(input_ids=packed.input_ids[None])
    assert plain.past_key_values is not None  # untouched without a tree
    uninstall(model)
    assert not any(module._forward_pre_hooks for module in model.modules())


def test_an_uncovered_recurrent_layer_beside_a_covered_one_is_refused():
    from test_gated_delta_net import qwen3_5

    model = qwen3_5()
    model.model.layers[1].extra = tiny_mamba().backbone.layers[0].mixer
    support = install(model)
    assert support.mode == "none"
    assert "MambaMixer at model.layers.1.extra" in support.reason


def test_an_uncovered_layer_type_is_refused():
    from test_gated_delta_net import qwen3_5

    model = qwen3_5()
    model.config.layer_types = ["linear_attention", "full_attention", "conv"]
    support = install(model)
    assert support.mode == "none" and "conv layers" in support.reason


def test_models_sharing_an_architecture_share_its_hooks():
    from test_gated_delta_net import modeling, qwen3_5
    from tree_recurrence import uninstall
    from tree_recurrence._hooks import hook_of
    from tree_testing import assert_tree_matches_alone, branching_sequences

    first, second = qwen3_5(), qwen3_5()
    install(first)
    install(second)
    uninstall(first)
    assert hook_of(modeling, "torch_chunk_gated_delta_rule") is not None
    assert_tree_matches_alone(second, branching_sequences())
    uninstall(second)
    assert hook_of(modeling, "torch_chunk_gated_delta_rule") is None
